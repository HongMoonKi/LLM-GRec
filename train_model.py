RESUME_FROM_EPOCH = 0
# train_model.py (논문 방식으로 수정)
import os
import sys
import time
import torch
import random
import numpy as np
from tqdm import tqdm

from models.seqllm_model import llmrec_model
from SeqRec.lightgcn.utils import data_partition


class Sampler:
    """
    Data sampler for LLM-SRec training (논문 방식)
    
    논문에서의 training:
    - Input: train items (전체)
    - Target: valid의 첫 번째 item
    """
    def __init__(self, user_train, user_valid, usernum, itemnum, 
                 batch_size=20, maxlen=128):
        self.user_train = user_train
        self.user_valid = user_valid
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen

        # ✅ valid가 있는 user만 사용
        self.user_ids = [u for u in user_train.keys() 
                         if len(user_train[u]) > 0 and 
                            u in user_valid and len(user_valid[u]) > 0]
        np.random.shuffle(self.user_ids)
        self.index = 0
        
        print(f"✅ Sampler initialized:")
        print(f"   Total users: {usernum:,}")
        print(f"   Valid users (with train & valid): {len(self.user_ids):,}")
        print(f"   Batch size: {batch_size}")

    def next_batch(self):
        batch_u, batch_seq, batch_pos, batch_neg = [], [], [], []

        attempts = 0
        max_attempts = self.batch_size * 3
        
        while len(batch_u) < self.batch_size and attempts < max_attempts:
            if self.index >= len(self.user_ids):
                np.random.shuffle(self.user_ids)
                self.index = 0

            u = self.user_ids[self.index]
            self.index += 1
            attempts += 1

            # ✅ Train history (전체 사용) - 논문 방식
            seq = list(self.user_train[u])
            
            # ✅ Valid의 첫 번째 item을 positive로 - 논문 방식
            if u not in self.user_valid or len(self.user_valid[u]) == 0:
                continue
            pos = self.user_valid[u][0]

            # ✅ Negative sampling (train + valid 모두 제외)
            history_set = set(seq)
            if u in self.user_valid:
                history_set.update(self.user_valid[u])
            
            neg = np.random.randint(1, self.itemnum + 1)
            max_neg_attempts = 100
            neg_attempts = 0
            while neg in history_set and neg_attempts < max_neg_attempts:
                neg = np.random.randint(1, self.itemnum + 1)
                neg_attempts += 1
            
            if neg in history_set:
                continue  # skip if couldn't find valid negative

            batch_u.append(u)
            batch_seq.append(seq)
            batch_pos.append(pos)
            batch_neg.append(neg)

        # ✅ 배치가 비어있으면 재시도
        if len(batch_u) == 0:
            print("⚠️ Warning: Empty batch, retrying...")
            return self.next_batch()

        return (
            np.array(batch_u),
            batch_seq,  # list of lists (각 user의 history 길이가 다름)
            np.array(batch_pos),
            np.array(batch_neg)
        )


def evaluate_model(model, user_train, user_valid, user_data, 
                   desc="Eval", is_test=False, args=None):  # ← args 추가
    """평가 함수
    
    Args:
        args: Arguments with eval_min_rating
    """
    model.eval()
    model.users = 0.0
    model.NDCG = 0.0
    model.HT = 0.0
    model.NDCG_20 = 0.0
    model.HIT_20 = 0.0
    model.all_embs = None
    
    # ✅ Rating filtering 설정
    min_rating = args.eval_min_rating if (args and hasattr(args, 'eval_min_rating')) else 0.0
    item_ratings = model.item_ratings if hasattr(model, 'item_ratings') else {}
    
    # ✅ Target user 선택 (rating filtering 적용)
    test_users = []
    for u in user_data.keys():
        if u not in user_train:
            continue
        
        if len(user_data[u]) > 0:
            target_item = user_data[u][0]
            
            # ✅ Rating 필터링
            if min_rating > 0.0:
                # Rating 기준 필터링
                if target_item in item_ratings:
                    rating = item_ratings[target_item]['avg_rating']
                    if rating >= min_rating:
                        test_users.append(u)
                # Rating 정보 없으면 제외
            else:
                # 모든 user 포함 (default)
                test_users.append(u)
    
    if len(test_users) > 10000:
        test_users = random.sample(test_users, 10000)
    
    rating_msg = f"≥{min_rating}★" if min_rating > 0 else "all"
    print(f"\n{'─'*70}")
    print(f"{desc}: {len(test_users):,} users (rating {rating_msg})")
    print(f"{'─'*70}")

    
    with torch.no_grad():
        for u in tqdm(test_users, desc=desc, leave=False):
            # ✅ 핵심 수정!
            if is_test:
                # Test: train + valid
                user_history = list(user_train[u])
                if u in user_valid and len(user_valid[u]) > 0:
                    user_history = user_history + user_valid[u]
            else:
                # Validation: train만
                user_history = list(user_train[u])
            
            pos_item = user_data[u][0]
            
            model(
                [np.array([u]), [user_history], [pos_item], [0], 0, None, 'original'],
                mode='generate_batch'
            )
    
    ndcg_10 = model.NDCG / model.users if model.users > 0 else 0
    hr_10 = model.HT / model.users if model.users > 0 else 0
    ndcg_20 = model.NDCG_20 / model.users if model.users > 0 else 0
    hr_20 = model.HIT_20 / model.users if model.users > 0 else 0
    
    return ndcg_10, hr_10, ndcg_20, hr_20


def train_model(args):
    """LLM-SRec training (논문 설정 - 수정됨)"""
    
    print("\n" + "="*70)
    print("LLM-SRec with LightGCN Training (Paper Configuration - Fixed)")
    print("="*70)
    print("📌 Training Strategy:")
    print("   Input:  train items (전체)")
    print("   Target: valid[0] (논문 방식)")
    print("="*70)

    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = data_partition(
        args.rec_pre_trained_data, 
        args,
        path=f'./SeqRec/data_{args.rec_pre_trained_data}/{args.rec_pre_trained_data}'
    )
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset

    print(f"✅ Dataset loaded:")
    print(f"   Users: {usernum:,}")
    print(f"   Items: {itemnum:,}")
    
    # ✅ Train/Valid/Test 통계
    avg_train = np.mean([len(v) for v in user_train.values() if len(v) > 0])
    valid_users = [u for u in user_valid.keys() if len(user_valid[u]) > 0]
    test_users = [u for u in user_test.keys() if len(user_test[u]) > 0]
    
    print(f"   Avg train items/user: {avg_train:.2f}")
    print(f"   Users with valid: {len(valid_users):,}")
    print(f"   Users with test: {len(test_users):,}")

    # Initialize model
    print("\n🤖 Initializing model...")
    model = llmrec_model(args, user_train=user_train).to(args.device)

    start_epoch = 1
    if RESUME_FROM_EPOCH > 0:
        print(f"\n{'='*70}")
        print(f"🔄 Loading checkpoint from epoch {RESUME_FROM_EPOCH}")
        print(f"{'='*70}")
        
        dataset_name = args.rec_pre_trained_data
        
        # ✅ 체크포인트 파일명
        ckpt_files = {
            'CLS': f'{dataset_name}_llama-3b_{RESUME_FROM_EPOCH}_CLS.pt',
            'CLS_item': f'{dataset_name}_llama-3b_{RESUME_FROM_EPOCH}_CLS_item.pt',  # ← 추가!
            'item_proj': f'{dataset_name}_llama-3b_{RESUME_FROM_EPOCH}_item_proj.pt',
            'pred_item': f'{dataset_name}_llama-3b_{RESUME_FROM_EPOCH}_pred_item.pt',
            'pred_user': f'{dataset_name}_llama-3b_{RESUME_FROM_EPOCH}_pred_user.pt'
        }
        
        # 파일 존재 확인
        all_exist = True
        for name, filename in ckpt_files.items():
            if os.path.exists(filename):
                ckpt_files[name] = filename
            elif os.path.exists(f'./models/{args.save_dir}/{filename}'):
                ckpt_files[name] = f'./models/{args.save_dir}/{filename}'
            else:
                print(f"❌ File not found: {filename}")
                all_exist = False
        
        if all_exist:
            print(f"✅ All checkpoint files found!")
            
            try:
                # 1. item_proj
                model.item_emb_proj.load_state_dict(
                    torch.load(ckpt_files['item_proj'], map_location=args.device)
                )
                print(f"   ✅ Loaded item_emb_proj")
                
                # 2. pred_user
                model.llm.pred_user.load_state_dict(
                    torch.load(ckpt_files['pred_user'], map_location=args.device)
                )
                print(f"   ✅ Loaded pred_user")
                
                # 3. pred_item
                model.llm.pred_item.load_state_dict(
                    torch.load(ckpt_files['pred_item'], map_location=args.device)
                )
                print(f"   ✅ Loaded pred_item")
                
                # 4. CLS
                if not args.token:
                    if args.nn_parameter:
                        model.llm.CLS.data = torch.load(
                            ckpt_files['CLS'], map_location=args.device
                        )
                        model.llm.CLS_item.data = torch.load(  # ← 추가!
                            ckpt_files['CLS_item'], map_location=args.device
                        )
                    else:
                        model.llm.CLS.load_state_dict(
                            torch.load(ckpt_files['CLS'], map_location=args.device)
                        )
                        model.llm.CLS_item.load_state_dict(  # ← 추가!
                            torch.load(ckpt_files['CLS_item'], map_location=args.device)
                        )
                    print(f"   ✅ Loaded CLS")
                    print(f"   ✅ Loaded CLS_item")  # ← 추가!

                # ✅ 5. Optimizer
                optimizer_file = f'{dataset_name}_llama-3b_{RESUME_FROM_EPOCH}_optimizer.pt'
                optimizer_path = None
                
                if os.path.exists(optimizer_file):
                    optimizer_path = optimizer_file
                elif os.path.exists(f'./models/{args.save_dir}/{optimizer_file}'):
                    optimizer_path = f'./models/{args.save_dir}/{optimizer_file}'
                
                if optimizer_path:
                    adam_optimizer.load_state_dict(
                        torch.load(optimizer_path, map_location=args.device)
                    )
                    print(f"   ✅ Loaded optimizer state")
                else:
                    print(f"   ⚠️ Optimizer state not found (starting fresh)")
                
                start_epoch = RESUME_FROM_EPOCH + 1
                print(f"\n✅ All checkpoints loaded successfully!")
                print(f"✅ Resuming from epoch {start_epoch}")
                print(f"{'='*70}\n")
                
            except Exception as e:
                print(f"\n❌ Error loading checkpoints: {e}")
                print(f"   Starting from scratch...\n")
                start_epoch = 1
        else:
            print(f"\n❌ Some checkpoint files missing.")
            print(f"   Starting from scratch...\n")
            start_epoch = 1

    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    print(f"✅ Trainable parameters: {total_params:,}")
    
    # Optimizer
    adam_optimizer = torch.optim.Adam(
        trainable_params, 
        lr=args.stage2_lr, 
        betas=(0.9, 0.98)
    )
    
    # ✅ Learning rate scheduler (논문)
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(adam_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    if RESUME_FROM_EPOCH > 0:
    # Scheduler를 올바른 epoch로 이동
        for _ in range(RESUME_FROM_EPOCH):
            scheduler.step()

    # ✅ Setup sampler (user_valid 추가!)
    print(f"\n📊 Setting up data sampler...")
    sampler = Sampler(
        user_train, 
        user_valid,  # ✅ 추가!
        usernum, 
        itemnum, 
        batch_size=args.batch_size, 
        maxlen=args.maxlen
    )
    
    num_batch = len(sampler.user_ids) // args.batch_size
    print(f"✅ Batches per epoch: {num_batch}")
    
    # ✅ Validation interval (epoch당 1번)
    val_interval = num_batch
    print(f"✅ Validation every {val_interval} steps (100% of epoch)")

    # Training setup
    best_perform = 0.0
    best_epoch = 0
    best_step = 0
    early_stop = 0
    early_thres = 10  # ✅ 논문: patience of 10
    
    start_time = time.time()

    print("\n" + "="*70)
    print("Training Configuration")
    print("="*70)
    print(f"Epochs:              {args.num_epochs}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Learning Rate:       {args.stage2_lr}")
    print(f"Early Stop Patience: {early_thres}")
    print(f"LR Scheduler:        0.95^epoch")
    print(f"Validation:          Every epoch")
    print("="*70 + "\n")

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        epoch_start = time.time()
        
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch}/{args.num_epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'─'*70}")

        # Training loop
        for step in tqdm(range(num_batch), desc="Training", ncols=100):
            u, seq, pos, neg = sampler.next_batch()
            
            # Forward & backward
            model(
                [u, seq, pos, neg],
                optimizer=adam_optimizer, 
                batch_iter=[epoch, args.num_epochs, step, num_batch], 
                mode='phase2'
            )
            
            # ✅ Validation at epoch end
            if step == num_batch - 1:
                print(f"\n{'─'*70}")
                print(f"📊 Validation at epoch {epoch}")
                print(f"{'─'*70}")
                
                # ✅ Validation: train → valid[0]
                val_ndcg10, val_hr10, val_ndcg20, val_hr20 = evaluate_model(
                    model, user_train, user_valid, user_valid, desc="Validating", is_test=False, args=args
                )
                
                print(f"\nValidation Results:")
                print(f"  HR@10:   {val_hr10:.4f}")
                print(f"  NDCG@10: {val_ndcg10:.4f}")
                print(f"  HR@20:   {val_hr20:.4f}")
                print(f"  NDCG@20: {val_ndcg20:.4f}")
                
                perform = val_hr10
                
                # Best model check
                if perform >= best_perform:
                    improvement = perform - best_perform
                    print(f"\n🎉 New Best Model! HR@10: {perform:.4f} (+{improvement:.4f})")
                    
                    best_perform = perform
                    best_epoch = epoch
                    best_step = step
                    early_stop = 0
                    
                    # Save best model
                    print(f"💾 Saving best model...")
                    model.save_model(args, epoch2=epoch, best=True, optimizer=adam_optimizer)
                    
                    # ✅ Test on best model: train+valid → test[0]
                    print(f"\n📈 Testing best model...")
                    test_ndcg10, test_hr10, test_ndcg20, test_hr20 = evaluate_model(
                        model, user_train, user_valid, user_test, desc="Testing", is_test=True, args=args
                    )
                    
                    print(f"\nTest Results (Best Model):")
                    print(f"  HR@10:   {test_hr10:.4f}")
                    print(f"  NDCG@10: {test_ndcg10:.4f}")
                    print(f"  HR@20:   {test_hr20:.4f}")
                    print(f"  NDCG@20: {test_ndcg20:.4f}")
                    
                    # Save results to file
                    # Save results to file (append mode - 모든 best epoch 기록)
                    result_dir = f'./models/{args.save_dir}/best/'
                    os.makedirs(result_dir, exist_ok=True)
                    result_file = os.path.join(result_dir, 
                        f'{args.rec_pre_trained_data}_{args.llm}_results_all.txt')
                    
                    # 파일이 없으면 헤더 작성
                    if not os.path.exists(result_file):
                        with open(result_file, 'w') as f:
                            f.write(f"All Best Model Results (Training History)\n")
                            f.write(f"="*70 + "\n\n")
                    
                    # Append mode로 결과 추가
                    with open(result_file, 'a') as f:
                        f.write(f"[Epoch {epoch}] New Best Model (Step {step})\n")
                        f.write(f"{'-'*70}\n")
                        f.write(f"Validation:\n")
                        f.write(f"  NDCG@10: {val_ndcg10:.4f}  |  HR@10: {val_hr10:.4f}\n")
                        f.write(f"  NDCG@20: {val_ndcg20:.4f}  |  HR@20: {val_hr20:.4f}\n")
                        f.write(f"\nTest:\n")
                        f.write(f"  NDCG@10: {test_ndcg10:.4f}  |  HR@10: {test_hr10:.4f}\n")
                        f.write(f"  NDCG@20: {test_ndcg20:.4f}  |  HR@20: {test_hr20:.4f}\n")
                        f.write(f"="*70 + "\n\n")
                    
                    print(f"✅ Results appended to {result_file}")
                else:
                    early_stop += 1
                    print(f"\n⚠️ No improvement. Early stop counter: {early_stop}/{early_thres}")
                    print(f"   Best: {best_perform:.4f} (epoch {best_epoch})")
                
                # Early stopping check
                if early_stop >= early_thres:
                    print(f"\n{'='*70}")
                    print(f"⛔ Early Stopping Triggered!")
                    print(f"{'='*70}")
                    print(f"Best Epoch: {best_epoch}")
                    print(f"Best HR@10: {best_perform:.4f}")
                    print(f"{'='*70}\n")
                    return
                
                model.train()
                print(f"{'─'*70}\n")

        epoch_time = time.time() - epoch_start
        
        # Save regular checkpoint
        print(f"\n💾 Saving epoch {epoch} checkpoint...")
        model.save_model(args, epoch2=epoch, optimizer=adam_optimizer)
        print(f"✅ Epoch {epoch} completed in {epoch_time/60:.1f} minutes")
        
        # ✅ Learning rate decay
        scheduler.step()

    # Training complete
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("🎉 Training Completed!")
    print("="*70)
    print(f"Total Epochs:    {args.num_epochs}")
    print(f"Best Epoch:      {best_epoch}")
    print(f"Best HR@10:      {best_perform:.4f}")
    print(f"Total Time:      {total_time/60:.1f} minutes")
    print(f"Avg Time/Epoch:  {total_time/args.num_epochs/60:.1f} minutes")
    print(f"Models Saved:    ./models/{args.save_dir}/")
    print("="*70 + "\n")