import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.seqllm4rec import *
from sentence_transformers import SentenceTransformer
from datetime import datetime

from tqdm import trange, tqdm

try:
    import habana_frameworks.torch.core as htcore
except:
    0


class llmrec_model(nn.Module):
    def __init__(self, args, user_train=None):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device

        self.distill_loss_type = args.distill_loss if hasattr(args, 'distill_loss') else 'mse'
        self.history_sampling = getattr(args, 'history_sampling', 'recent')
        print(f"✅ History sampling strategy: {self.history_sampling}")

        self.user_train = user_train  # ✅ 전달받은 user_train 저장
        print(f"DEBUG in __init__: received user_train = {user_train is None}")  # ✅
        print(f"DEBUG in __init__: self.user_train = {self.user_train is None}")  # ✅

        with open(f'./SeqRec/data_{args.rec_pre_trained_data}/text_name_dict.json.gz','rb') as ft:
            self.text_name_dict = pickle.load(ft)

        import os
        category_file = f'./SeqRec/data_{args.rec_pre_trained_data}/item_categories.pkl'
        if os.path.exists(category_file):
            try:
                with open(category_file, 'rb') as f:
                    self.item_to_category = pickle.load(f)
                print(f"✅ Loaded categories for {len(self.item_to_category):,} items")
            except Exception as e:
                print(f"⚠️ Error loading categories: {e}")
                self.item_to_category = {}
        else:
            self.item_to_category = {}
            print(f"⚠️ No category file found")
            print(f"ℹ️ Run: python extract_categories.py {args.rec_pre_trained_data}")
        
        id2asin_file = f'./SeqRec/data_{args.rec_pre_trained_data}/id_to_asin.pkl'
        if os.path.exists(id2asin_file):
            with open(id2asin_file, 'rb') as f:
                self.id_to_asin = pickle.load(f)
            print(f"✅ Loaded id_to_asin for {len(self.id_to_asin):,} items")
        else:
            self.id_to_asin = {}
            print("⚠️ No id_to_asin file found")

        rating_file = f'./SeqRec/data_{args.rec_pre_trained_data}/item_ratings_by_id.pkl'
        if os.path.exists(rating_file):
            with open(rating_file, 'rb') as f:
                self.item_ratings = pickle.load(f)
            print(f"✅ Loaded ratings for {len(self.item_ratings):,} items")
        else:
            self.item_ratings = {}
            print("⚠️ No rating file found")

        recsys_checkpoint = args.recsys_data if hasattr(args, 'recsys_data') else rec_pre_trained_data
        self.recsys = RecSys(args.recsys, recsys_checkpoint, self.device, args, user_train=self.user_train)


        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.all_embs = None
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.NDCG_20 = 0
        self.HIT_20 = 0


        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG=0
        self.lan_HIT=0
        self.num_user = 0
        self.yes = 0

        self.extract_embs_list = []

        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

        self.llm = llm4rec(device=self.device, llm_model=args.llm, args = self.args)

        self.projection_dim = 128

        # Stage 1: 64 → 128
        self.item_emb_proj = nn.Linear(self.rec_sys_dim, self.projection_dim)
        self.user_emb_proj = nn.Linear(self.rec_sys_dim, self.projection_dim)

        self.to_llm_hidden = nn.Linear(self.projection_dim, self.llm.llm_model.config.hidden_size)

        # Initialization
        nn.init.xavier_normal_(self.item_emb_proj.weight)
        nn.init.xavier_normal_(self.user_emb_proj.weight)
        nn.init.xavier_normal_(self.to_llm_hidden.weight)

        self.users = 0.0
        self.NDCG = 0.0
        self.HT = 0.0
        

        if hasattr(args, 'use_category') and args.use_category and len(self.item_to_category) > 0:
            unique_categories = sorted(set(self.item_to_category.values()))
            self.category_to_id = {cat: idx for idx, cat in enumerate(unique_categories)}
            self.num_categories = len(unique_categories)
            
            category_dim = args.category_dim if hasattr(args, 'category_dim') else 32
            self.category_emb = nn.Embedding(self.num_categories, category_dim)
            nn.init.xavier_normal_(self.category_emb.weight)
            
            # ✅ 수정: 128 → 64 (LightGCN과 같은 크기!)
            self.category_proj = nn.Linear(category_dim, self.rec_sys_dim)  # 32 → 64
            nn.init.xavier_normal_(self.category_proj.weight)
    
            print(f"✅ Category Embedding: {self.num_categories} categories, {category_dim}→{self.rec_sys_dim}")
        else:
            self.category_emb = None
            self.category_proj = None

    def get_rating_weight(self, item_id):
        """Get rating-based weight for loss scaling"""
        if item_id not in self.item_ratings:
            return 0.5  # neutral for unknown
        
        rating = self.item_ratings[item_id]['avg_rating']
        
        if rating >= 4.5:
            return 1.0  # loved
        elif rating >= 3.5:
            return 0.7  # enjoyed
        elif rating >= 2.5:
            return 0.4  # okay
        else:
            return 0.2  # tried

    def get_rating_text(self, item_id):
        """Get rating emphasis text for prompt"""
        if item_id not in self.item_ratings:
            return "purchased"
        
        rating = self.item_ratings[item_id]['avg_rating']
        
        if rating >= 4.5:
            return "absolutely loved"
        elif rating >= 4.0:
            return "really enjoyed"
        elif rating >= 3.5:
            return "enjoyed"
        elif rating >= 3.0:
            return "found acceptable"
        elif rating >= 2.5:
            return "tried"
        else:
            return "found unsatisfactory"

    def sample_user_history(self, user_id, total=10):
        """Sample user history with different strategies"""
        user_history = list(self.user_train[user_id])
        
        if len(user_history) <= total:
            return user_history
        
        if self.history_sampling == 'recent':
            # 기존: 최근 N개만
            return user_history[-total:]
        
        elif self.history_sampling == 'hybrid':
            # Hybrid: Recent 5 + Diverse 5
            recent_num = 5
            diverse_num = total - recent_num
            
            # Recent items
            recent_items = user_history[-recent_num:]
            older_items = user_history[:-recent_num]
            
            # Category-based diversity
            category_items = {}
            for item_id in older_items:
                if item_id in self.id_to_asin:
                    asin = self.id_to_asin[item_id]
                    if asin in self.item_to_category:
                        cat = self.item_to_category[asin]
                        if cat not in category_items:
                            category_items[cat] = item_id
            
            diverse_items = list(category_items.values())
            
            # 부족하면 랜덤 추가
            if len(diverse_items) < diverse_num:
                remaining = [x for x in older_items if x not in diverse_items]
                if remaining:
                    need = diverse_num - len(diverse_items)
                    diverse_items.extend(random.sample(remaining, min(need, len(remaining))))
            
            random.shuffle(diverse_items)
            diverse_items = diverse_items[:diverse_num]
            
            # Diverse first, then recent (순서 유지)
            return diverse_items + recent_items
        
        else:
            return user_history[-total:]



    def save_model(self, args, epoch2=None, best=False, optimizer=None):
        out_dir = f'./models/{args.save_dir}/'
        if best:
            out_dir = out_dir[:-1] + 'best/'

        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_'

        out_dir += f'{args.llm}_{epoch2}_'
        if args.train:
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            torch.save(self.llm.pred_user.state_dict(), out_dir + 'pred_user.pt')
            torch.save(self.llm.pred_item.state_dict(), out_dir + 'pred_item.pt')
            if not args.token:
                if args.nn_parameter:
                    torch.save(self.llm.CLS.data, out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.data, out_dir + 'CLS_item.pt')
                else:
                    torch.save(self.llm.CLS.state_dict(), out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.state_dict(), out_dir + 'CLS_item.pt')
            if args.token:
                torch.save(self.llm.llm_model.model.embed_tokens.state_dict(), out_dir + 'token.pt')

            if optimizer is not None:
                torch.save(optimizer.state_dict(), out_dir + 'optimizer.pt')

    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./models/{args.save_dir}/{args.rec_pre_trained_data}_'

        out_dir += f'{args.llm}_{phase2_epoch}_'


        item_emb_proj = torch.load(out_dir + 'item_proj.pt', map_location = self.device)
        self.item_emb_proj.load_state_dict(item_emb_proj)
        del item_emb_proj


        pred_user = torch.load(out_dir + 'pred_user.pt', map_location = self.device)
        self.llm.pred_user.load_state_dict(pred_user)
        del pred_user

        pred_item = torch.load(out_dir + 'pred_item.pt', map_location = self.device)
        self.llm.pred_item.load_state_dict(pred_item)
        del pred_item

        if not args.token:
            CLS = torch.load(out_dir + 'CLS.pt', map_location = self.device)
            self.llm.CLS.load_state_dict(CLS)
            del CLS

            CLS_item = torch.load(out_dir + 'CLS_item.pt', map_location = self.device)
            self.llm.CLS_item.load_state_dict(CLS_item)
            del CLS_item

        if args.token:
            token = torch.load(out_dir + 'token.pt', map_location = self.device)
            self.llm.llm_model.model.embed_tokens.load_state_dict(token)
            del token


    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        if title_flag and description_flag:
            return [f'"{(self.text_name_dict[t].get(i) or t_)[:100]}, {(self.text_name_dict[d].get(i) or d_)[:200]}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{(self.text_name_dict[t].get(i) or t_)[:100]}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{(self.text_name_dict[d].get(i) or d_)[:200]}"' for i in item]

    def find_item_time(self, item, user, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        l = [datetime.utcfromtimestamp(int(self.text_name_dict['time'][i][user])/1000) for i in item]
        return [l_.strftime('%Y-%m-%d') for l_ in l]


    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        if title_flag and description_flag:
            return f'"{(self.text_name_dict[t].get(item) or t_)[:100]}, {(self.text_name_dict[d].get(item) or d_)[:200]}"'
        elif title_flag and not description_flag:
            return f'"{(self.text_name_dict[t].get(item) or t_)[:100]}"'
        elif not title_flag and description_flag:
            return f'"{(self.text_name_dict[d].get(item) or d_)[:200]}"'

    def get_item_emb(self, item_ids):
        """Get item embeddings with optional category enhancement"""
        with torch.no_grad():
            _, item_all = self.recsys.model.get_all_embeddings()
            item_embs = item_all[(torch.LongTensor(item_ids).to(self.device) - 1)]
            
            if self.category_emb is not None:
                category_ids = []
                for item_id in item_ids:
                    cat_id = 0  # default unknown
                    if item_id in self.id_to_asin:  # ✅ 수정
                        asin = self.id_to_asin[item_id]
                        if asin in self.item_to_category:
                            cat = self.item_to_category[asin]
                            cat_id = self.category_to_id.get(cat, 0)
                    category_ids.append(cat_id)
                
                category_ids = torch.LongTensor(category_ids).to(self.device)
                cat_embs = self.category_emb(category_ids)
                cat_proj = self.category_proj(cat_embs)
                item_embs = item_embs + cat_proj
        
        return item_embs

    # ✅ 여기에 추가!
    def cosine_distill_loss(self, llm_user, gcn_user):
        """
        Cosine similarity loss for LightGCN-LLM distillation
        LightGCN produces L2-normalized embeddings, so direction matters more than magnitude
        """
        import torch.nn.functional as F
        
        # L2 normalize both embeddings
        llm_user_norm = F.normalize(llm_user, p=2, dim=-1)
        gcn_user_norm = F.normalize(gcn_user, p=2, dim=-1)
        
        # Cosine similarity (higher = more similar)
        cos_sim = F.cosine_similarity(llm_user_norm, gcn_user_norm, dim=-1)
        
        # Convert to loss (1 - similarity, so lower is better)
        loss = (1 - cos_sim).mean()
        
        return loss


    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode=='generate_batch':
            self.generate_batch(data)
            print(self.args.save_dir, self.args.rec_pre_trained_data)
            print('test (NDCG@10: %.4f, HR@10: %.4f), Num User: %.4f'
                    % (self.NDCG/self.users, self.HT/self.users, self.users))
            print('test (NDCG@20: %.4f, HR@20: %.4f), Num User: %.4f'
                    % (self.NDCG_20/self.users, self.HIT_20/self.users, self.users))
        if mode=='extract':
            self.extract_emb(data)

    def make_interact_text(self, interact_ids, interact_max_num, user):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        interact_text = []
        count = 1
        
        # Interaction 수 제한
        if interact_max_num == 'all':
            times = self.find_item_time(interact_ids, user)
            items_to_use = interact_item_titles_
            ids_to_use = interact_ids
        else:
            times = self.find_item_time(interact_ids[-interact_max_num:], user)
            items_to_use = interact_item_titles_[-interact_max_num:]
            ids_to_use = interact_ids[-interact_max_num:]
        
        total_items = len(items_to_use)
        
        # 각 아이템 처리
        for idx, (title, time, item_id) in enumerate(zip(items_to_use, times, ids_to_use)):
            
            # ✅ Rating 텍스트 추가
            rating_text = self.get_rating_text(item_id)
            
            # Category
            if item_id in self.id_to_asin:
                asin = self.id_to_asin[item_id]
                if asin in self.item_to_category:
                    category = self.item_to_category[asin]
                    category_text = f', Category: {category}'
                else:
                    category_text = ''
            else:
                category_text = ''
            
            
            # ✅ 최종 텍스트 (Rating 포함!)
            interact_text.append(
                f'{rating_text}: Item No.{count}, Time: {time}{category_text}, {title}[HistoryEmb]'
            )
            count += 1
        
        if interact_max_num != 'all':
            interact_ids = ids_to_use
        
        interact_text = ','.join(interact_text)
        return interact_text, interact_ids


    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set = None, task = 'ItemTask'):
        neg_item_id = []
        if candi_set == None:
            neg_item_id = []
            while len(neg_item_id)<99:
                t = np.random.randint(1, self.item_num+1)
                if not (t in interact_ids or t in neg_item_id):
                    neg_item_id.append(t)
        else:
            his = set(interact_ids)
            items = list(candi_set.difference(his))
            if len(items) >99:
                neg_item_id = random.sample(items, 99)
            else:
                while len(neg_item_id)<49:
                    t = np.random.randint(1, self.item_num+1)
                    if not (t in interact_ids or t in neg_item_id):
                        neg_item_id.append(t)
        random.shuffle(neg_item_id)

        candidate_ids = [target_item_id]

        candidate_text = [f'The item title and item embedding are as follows: ' + target_item_title + "[HistoryEmb], then generate item representation token:[ItemOut]"]


        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append(f'The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut]")

            candidate_ids.append(neg_candidate)

        return candidate_text, candidate_ids


    def make_candidate(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set = None, task = 'ItemTask'):
        neg_item_id = []
        neg_item_id = []
        while len(neg_item_id)<99:
            t = np.random.randint(1, self.item_num+1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)

        random.shuffle(neg_item_id)

        candidate_ids = [target_item_id]

        candidate_ids = candidate_ids + neg_item_id[:candidate_num - 1]

        return candidate_ids


    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        print(self.args.save_dir, self.args.rec_pre_trained_data, self.args.llm)
        optimizer.zero_grad()
        u, seq, pos, neg = data
        
        rating_weights = []
        for pos_item in pos:
            weight = self.get_rating_weight(pos_item)
            rating_weights.append(weight)
        rating_weights = torch.tensor(rating_weights, dtype=torch.float32).to(self.device)


        original_seq = seq.copy()


        mean_loss = 0

        text_input = []
        candidates_pos = []
        candidates_neg = []
        interact_embs = []
        candidate_embs_pos = []
        candidate_embs_neg = []
        candidate_embs = []

        loss_rm_mode1 = 0
        loss_rm_mode2 = 0

        with torch.no_grad():
            u_tensor = torch.tensor(u, device=self.device).long()
            user_all, item_all = self.recsys.model.get_all_embeddings()
            log_emb = user_all[u_tensor]

        for i in range(len(u)):
            user_id = u[i].item()

            # 🔹 LightGCN에서는 seq 대신 user_train에서 직접 가져옴
            # (user_train은 dict: {user_id: [item1, item2, ...]} 형태)
            user_history = self.sample_user_history(user_id, total=10)

            target_item_id = pos[i]
            target_item_title = self.find_item_text_single(
                target_item_id, title_flag=True, description_flag=False
            )

            # 🔹 history 기반 문장 생성
            interact_text, interact_ids = self.make_interact_text(user_history, 10, user_id)
            candidate_num = 4
            candidate_text, candidate_ids = self.make_candidate_text(
                user_history, candidate_num, target_item_id, target_item_title, task="RecTask"
            )

                #no user
            input_text = ''


            input_text += 'This user has made a series of purchases in the following order: '

            input_text += interact_text

            input_text +=". Based on this sequence of purchases, generate user representation token:[UserOut]"

            text_input.append(input_text)

            candidates_pos += candidate_text


            proj_emb = self.item_emb_proj(self.get_item_emb(interact_ids))
            interact_embs.append(self.to_llm_hidden(proj_emb))

            proj_emb = self.item_emb_proj(self.get_item_emb(candidate_ids))  # ← [candidate_ids] 제거!
            candidate_embs_pos.append(proj_emb)    

        candidate_embs = torch.cat(candidate_embs_pos)
        candidate_embs = self.to_llm_hidden(candidate_embs) 


        samples = {'text_input': text_input, 'log_emb':log_emb, 'candidates_pos': candidates_pos, 'interact': interact_embs, 'candidate_embs':candidate_embs,}

        loss, rec_loss, match_loss = self.llm(samples, mode=0)

        # ✅ Rating-weighted loss
        weighted_rec_loss = (rec_loss * rating_weights).mean()
        total_loss = weighted_rec_loss + match_loss
        
        # ✅ NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️ Skipping NaN loss at iteration {step}")
            return
        
        # Print
        print(f"LLMRec loss in epoch {epoch}/{total_epoch} iter {step}/{total_step}: "
            f"rec={weighted_rec_loss:.4f}, match={match_loss:.4f}, total={total_loss:.4f}")
        
        # ✅ Backward (한 번만!)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        if self.args.nn_parameter:
            htcore.mark_step()
        
        optimizer.step()

    def split_into_batches(self,itemnum, m):
        numbers = list(range(1, itemnum+1))

        batches = [numbers[i:i + m] for i in range(0, itemnum, m)]

        return batches


    def generate_batch(self, data):
        if self.all_embs == None:
            batch_ = 128
            if self.args.llm == 'llama':
                batch_ = 64
            if self.args.rec_pre_trained_data == 'Electronics' or self.args.rec_pre_trained_data == 'Books':
                batch_ = 64
                if self.args.llm == 'llama':
                    batch_ = 32
            
            batches = self.split_into_batches(self.item_num, batch_)
            self.all_embs = []
            max_input_length = 1024
            
            for bat in tqdm(batches, desc="Generating item embeddings"):
                candidate_text = []
                candidate_ids = []
                
                for neg_candidate in bat:
                    candidate_text.append('The item title and item embedding are as follows: ' + 
                                        self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + 
                                        "[HistoryEmb], then generate item representation token:[ItemOut]")
                    candidate_ids.append(neg_candidate)
                
                with torch.no_grad():
                    candi_tokens = self.llm.llm_tokenizer(
                        candidate_text,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=max_input_length,
                    ).to(self.device)
                    
                    proj_emb = self.item_emb_proj(self.get_item_emb(candidate_ids))
                    candidate_embs = self.to_llm_hidden(proj_emb)
                    
                    candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                    candi_embeds = self.llm.replace_out_token_all_infer(
                        candi_tokens, candi_embeds, 
                        token=['[ItemOut]', '[HistoryEmb]'], 
                        embs={'[HistoryEmb]': candidate_embs}
                    )
                    
                    with torch.amp.autocast('cuda'):
                        candi_outputs = self.llm.llm_model.forward(
                            inputs_embeds=candi_embeds,
                            output_hidden_states=True
                        )
                        
                        indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]')
                        item_outputs = torch.cat([
                            candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) 
                            for i in range(len(indx))
                        ])
                        item_outputs = self.llm.pred_item(item_outputs)
                    
                    # ✅ CPU로 이동 (GPU 메모리 절약)
                    self.all_embs.append(item_outputs.cpu())
                    
                    # ✅ 모든 중간 변수 삭제
                    del candi_outputs
                    del item_outputs
                    del candi_embeds
                    del candi_tokens
                    del candidate_embs
                    del proj_emb
                    
                    # ✅ GPU 메모리 즉시 해제
                    torch.cuda.empty_cache()
            
            # ✅ 다시 GPU로 이동
            self.all_embs = torch.cat(self.all_embs).to(self.device)

        u, seq, pos, neg, rank, candi_set, files = data
        original_seq = seq.copy()

        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            for i in range(len(u)):

                candidate_embs = []
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

                user_id = u[i]
                # ✅ 수정: 전달받은 seq 사용 (LLM-SRec 방식)
                # seq[i]는 evaluate_model에서 이미 train+valid 합쳐서 전달됨
                if isinstance(seq[i], np.ndarray):
                    full_history = seq[i][seq[i] > 0].tolist()  # 0 제거
                else:
                    full_history = [x for x in seq[i] if x > 0]
                
                # ✅ Prompt용: 최근 10개만 (프롬프트 길이 제한)
                prompt_history = full_history[-10:] if len(full_history) > 10 else full_history
                
                interact_text, interact_ids = self.make_interact_text(prompt_history, 10, user_id)

                candidate_num = 100
                # ✅ Neg 제외: 전체 history 제외 (LLM-SRec 방식)
                candidate_ids = self.make_candidate(full_history, candidate_num, target_item_id, target_item_title, candi_set)

                candidate.append(candidate_ids)


                #no user
                input_text = ''


                input_text += 'This user has made a series of purchases in the following order: '

                input_text += interact_text


                input_text +=". Based on this sequence of purchases, generate user representation token:[UserOut]"

                text_input.append(input_text)


                proj_emb = self.item_emb_proj(self.get_item_emb(interact_ids))
                interact_embs.append(self.to_llm_hidden(proj_emb))


            max_input_length = 1024

            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

                #no user
            inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[HistoryEmb]'], embs= { '[HistoryEmb]':interact_embs})

            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,

                    output_hidden_states=True
                )

                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)

                for i in range(len(candidate)):

                    item_outputs = self.all_embs[np.array(candidate[i])-1]

                    logits= torch.mm(item_outputs, user_outputs[i].unsqueeze(0).T).squeeze(-1)

                    logits = -1*logits

                    rank = logits.argsort().argsort()[0].item()

                    if rank < 10:
                        self.NDCG += 1 / np.log2(rank + 2)
                        self.HT += 1
                    if rank < 20:
                        self.NDCG_20 += 1 / np.log2(rank + 2)
                        self.HIT_20 += 1
                    self.users +=1
        return self.NDCG

    def extract_emb(self,data):
        u, seq, pos, neg, original_seq, rank, files = data

        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            for i in range(len(u)):

                user_id = u[i]
                user_history = self.sample_user_history(user_id, total=10)
                interact_text, interact_ids = self.make_interact_text(user_history, 10, user_id)

                input_text = ''


                input_text += 'This user has made a series of purchases in the following order: '

                input_text += interact_text


                input_text +=". Based on this sequence of purchases, generate user representation token:[UserOut]"

                text_input.append(input_text)

                proj_emb = self.item_emb_proj(self.get_item_emb(interact_ids))
                interact_embs.append(self.to_llm_hidden(proj_emb))


            max_input_length = 1024

            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

            inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[HistoryEmb]'], embs= { '[HistoryEmb]':interact_embs})

            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,

                    output_hidden_states=True
                )

                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)

                self.extract_embs_list.append(user_outputs.detach().cpu())

        return 0