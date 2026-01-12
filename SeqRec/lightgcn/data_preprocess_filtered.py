import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random
import pickle
from datasets import load_dataset


def preprocess_with_rating_filter(fname, min_rating=3.0):
    """
    Rating filtering version
    - Filters out low ratings (1-2★) as noise
    - Only keeps interactions with rating >= min_rating
    """

    random.seed(0)
    np.random.seed(0)

    print(f"\n{'='*70}")
    print(f"🔥 RATING FILTERING ENABLED: >= {min_rating}★")
    print(f"{'='*70}\n")

    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"5core_last_out_{fname}", trust_remote_code=True)
    meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{fname}", trust_remote_code=True)

    print("Load Meta Data")
    meta_dict = {}
    for l in tqdm(meta_dataset['full']):
        meta_dict[l['parent_asin']] = [l['title'], l['description']]
    del meta_dataset

    # ✅ Rating 정보 추출을 위한 사전 처리
    print("\n📊 Loading rating information...")
    item_ratings_dict = {}  # {(user_id, asin): rating}
    
    for t in ['train', 'valid', 'test']:
        d = dataset[t]
        for l in tqdm(d, desc=f"Loading {t} ratings"):
            user_id = l['user_id']
            asin = l['parent_asin']
            rating = l.get('rating', 5.0)  # Default 5.0 if no rating field
            
            # ✅ Convert to float if string
            if isinstance(rating, str):
                rating = float(rating)
            
            item_ratings_dict[(user_id, asin)] = rating

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = defaultdict(list)
    User_s = {'train': defaultdict(list), 'valid': defaultdict(list), 'test': defaultdict(list)}
    id2asin = dict()
    time_dict = defaultdict(dict)
    
    # ✅ Rating filtering statistics
    total_interactions = 0
    filtered_interactions = 0
    rating_distribution = defaultdict(int)

    for t in ['train', 'valid', 'test']:
        d = dataset[t]

        for l in tqdm(d, desc=f"Processing {t}"):
            user_id = l['user_id']
            asin = l['parent_asin']
            rating = l.get('rating', 5.0)
            
            # ✅ Convert to float if string
            if isinstance(rating, str):
                rating = float(rating)
            
            total_interactions += 1
            rating_distribution[round(rating)] += 1  # ✅ round로 변경 (5.0 → 5)
            
            # ✅ Rating filtering
            if rating < min_rating:
                filtered_interactions += 1
                continue  # Skip low ratings!

            if user_id in usermap:
                userid = usermap[user_id]
            else:
                usernum += 1
                userid = usernum
                usermap[user_id] = userid

            if asin in itemmap:
                itemid = itemmap[asin]
            else:
                itemnum += 1
                itemid = itemnum
                itemmap[asin] = itemid

            User[userid].append(itemid)
            User_s[t][userid].append(itemid)
            id2asin[itemid] = asin
            time_dict[itemid][userid] = l['timestamp']

    # ✅ Print filtering statistics
    print(f"\n{'='*70}")
    print(f"📊 RATING FILTERING RESULTS")
    print(f"{'='*70}")
    print(f"Total interactions:     {total_interactions:,}")
    print(f"Filtered out (<{min_rating}★): {filtered_interactions:,} ({filtered_interactions/total_interactions*100:.1f}%)")
    print(f"Remaining (>={min_rating}★):   {total_interactions - filtered_interactions:,} ({(1-filtered_interactions/total_interactions)*100:.1f}%)")
    print(f"\nRating distribution:")
    for r in sorted(rating_distribution.keys()):
        count = rating_distribution[r]
        pct = count / total_interactions * 100
        status = "❌ FILTERED" if r < min_rating else "✅ KEPT"
        print(f"  {r}★: {count:8,} ({pct:5.1f}%) {status}")
    print(f"{'='*70}\n")

    sample_size = int(len(User.keys()))
    print(f'Num users after filtering: {sample_size:,}')
    
    sample_rate = {
        'Movies_and_TV': 0.05,
        'Electronics': 0.05,
        'Industrial_and_Scientific': 1.0,
        'CDs_and_Vinyl': 0.33,
    }

    sample_ratio = sample_rate[fname]
    use_key = random.sample(list(User.keys()), int(sample_size*sample_ratio))
    print(f'Num sample users: {len(use_key):,}')

    CountU = defaultdict(int)
    CountI = defaultdict(int)

    usermap_final = dict()
    itemmap_final = dict()
    usernum_final = 0
    itemnum_final = 0
    use_key_dict = defaultdict(int)
    use_train_dict = defaultdict(int)
    
    for key in use_key:
        use_key_dict[key] = 1

        for t in ['train', 'valid', 'test']:
            for i_ in User_s[t][key]:
                CountI[i_] += 1
                CountU[key] += 1

    text_dict = {'time': defaultdict(dict), 'description': {}, 'title': {}}
    
    # ✅ Output directory with filtering info
    dataset_name = f'{fname}_filtered{int(min_rating)}'
    output_dir = f'./../data_{dataset_name}/'
    os.makedirs(output_dir, exist_ok=True)
    
    for t in ['train', 'valid', 'test']:
        d = dataset[t]
        use_id = defaultdict(int)
        # ✅ 파일명도 dataset_name으로 통일
        f = open(f'{output_dir}{dataset_name}_{t}.txt', 'w')
        
        for l in tqdm(d, desc=f"Writing {t}"):
            user_id = l['user_id']
            asin = l['parent_asin']
            rating = l.get('rating', 5.0)
            
            # ✅ Convert to float if string
            if isinstance(rating, str):
                rating = float(rating)
            
            # ✅ Skip filtered ratings
            if rating < min_rating:
                continue
                
            user_id_ = usermap[user_id]
            
            if use_id[user_id_] == 0:
                use_id[user_id_] = 1
            else:
                continue
                
            if use_key_dict[user_id_] == 1 and CountU[user_id_] > 4:
                use_items = []
                for it in User_s[t][user_id_]:
                    if CountI[it] > 4:
                        use_items.append(it)
                        
                if t == 'train':
                    if len(use_items) > 4:
                        use_train_dict[user_id_] = 1
                        if user_id_ in usermap_final:
                            userid = usermap_final[user_id_]
                        else:
                            usernum_final += 1
                            userid = usernum_final
                            usermap_final[user_id_] = userid
                            
                        for it in use_items:
                            if it in itemmap_final:
                                itemid = itemmap_final[it]
                            else:
                                itemnum_final += 1
                                itemid = itemnum_final
                                itemmap_final[it] = itemid

                            d = meta_dict[id2asin[it]][1]
                            if d == None:
                                text_dict['description'][itemid] = 'Empty description'
                            elif len(d) == 0:
                                text_dict['description'][itemid] = 'Empty description'
                            else:
                                text_dict['description'][itemid] = d[0]
                            texts = meta_dict[id2asin[it]][0]

                            if texts == None:
                                text_dict['title'][itemid] = 'Empty title'
                            elif len(texts) == 0:
                                text_dict['title'][itemid] = 'Empty title'
                            else:
                                texts_ = texts
                                text_dict['title'][itemid] = texts_
                            text_dict['time'][itemid][userid] = time_dict[it][user_id_]

                            f.write('%d %d\n' % (userid, itemid))
                else:
                    if use_train_dict[user_id_] == 1:
                        for it in User_s[t][user_id_]:
                            if CountI[it] > 4:
                                if user_id_ in usermap_final:
                                    userid = usermap_final[user_id_]
                                else:
                                    usernum_final += 1
                                    userid = usernum_final
                                    usermap_final[user_id_] = userid
                                if it in itemmap_final:
                                    itemid = itemmap_final[it]
                                else:
                                    itemnum_final += 1
                                    itemid = itemnum_final
                                    itemmap_final[it] = itemid

                                d = meta_dict[id2asin[it]][1]
                                if d == None:
                                    text_dict['description'][itemid] = 'Empty description'
                                elif len(d) == 0:
                                    text_dict['description'][itemid] = 'Empty description'
                                else:
                                    text_dict['description'][itemid] = d[0]
                                texts = meta_dict[id2asin[it]][0]

                                if texts == None:
                                    text_dict['title'][itemid] = 'Empty title'
                                elif len(texts) == 0:
                                    text_dict['title'][itemid] = 'Empty title'
                                else:
                                    texts_ = texts
                                    text_dict['title'][itemid] = texts_
                                text_dict['time'][itemid][userid] = time_dict[it][user_id_]

                                f.write('%d %d\n' % (userid, itemid))
        f.close()
        
    with open(f'{output_dir}text_name_dict.json.gz', 'wb') as tf:
        pickle.dump(text_dict, tf)

    # ✅ Save id_to_asin mapping
    print(f"\n💾 Saving id_to_asin mapping...")
    with open(f'{output_dir}id_to_asin.pkl', 'wb') as f:
        pickle.dump(id2asin, f)
    print(f"✅ Saved id_to_asin.pkl with {len(id2asin):,} items")

    # ✅ Link category and rating files from original dataset
    print(f"\n🔗 Linking category/rating files from original dataset...")
    original_data_dir = f'./../data_{fname}/'
    
    for file in ['item_categories.pkl', 'item_ratings.pkl', 'item_ratings_by_id.pkl']:
        src = os.path.join(original_data_dir, file)
        dst = os.path.join(output_dir, file)
        
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.symlink(src, dst)
                print(f"   ✅ Linked: {file}")
            except OSError:
                # Windows에서 symlink 실패 시 복사
                import shutil
                shutil.copy2(src, dst)
                print(f"   ✅ Copied: {file}")
        else:
            print(f"   ⚠️ Not found: {file} (will need to extract separately)")

    print(f"\n✅ Filtered data saved to: {output_dir}")
    print(f"✅ Final users: {usernum_final:,}")
    print(f"✅ Final items: {itemnum_final:,}\n")

    del text_dict
    del meta_dict
    del dataset


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CDs_and_Vinyl',
                       choices=['Movies_and_TV', 'Electronics', 'Industrial_and_Scientific', 'CDs_and_Vinyl'])
    parser.add_argument('--min_rating', type=float, default=3.0,
                       help='Minimum rating threshold (default: 3.0)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Min Rating: {args.min_rating}★")
    print(f"{'='*70}\n")
    
    preprocess_with_rating_filter(args.dataset, args.min_rating)