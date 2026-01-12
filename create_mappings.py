import pickle
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

dataset_name = 'Electronics'

print(f"📥 Loading dataset: {dataset_name}")

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
id2asin = dict()

print("\n🔍 Building mappings...")
for t in ['train', 'valid', 'test']:
    print(f"\n📥 Loading {t} split...")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"0core_timestamp_w_his_{dataset_name}",
        split=t,
        trust_remote_code=True
    )

    for l in tqdm(dataset, desc=f"Processing {t}"):
        user_id = l['user_id']
        asin = l['parent_asin']

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
            id2asin[itemid] = asin

print(f"\n✅ Created mappings:")
print(f"   Users: {usernum:,}")
print(f"   Items: {itemnum:,}")
print(f"   ID→ASIN: {len(id2asin):,}")

with open(f'./SeqRec/data_{dataset_name}/id_to_asin.pkl', 'wb') as f:
    pickle.dump(id2asin, f)

with open(f'./SeqRec/data_{dataset_name}/item_ratings.pkl', 'rb') as f:
    asin_ratings = pickle.load(f)

id_ratings = {}
for item_id, asin in id2asin.items():
    if asin in asin_ratings:
        id_ratings[item_id] = asin_ratings[asin]

print(f"\n✅ Matched {len(id_ratings):,}/{len(id2asin):,} ratings ({len(id_ratings)/len(id2asin)*100:.1f}%)")

with open(f'./SeqRec/data_{dataset_name}/item_ratings_by_id.pkl', 'wb') as f:
    pickle.dump(id_ratings, f)

print(f"\n📦 Sample (ID → ASIN → Rating):")
for item_id in list(id_ratings.keys())[:5]:
    asin = id2asin[item_id]
    rating = id_ratings[item_id]['avg_rating']
    count = id_ratings[item_id]['num_ratings']
    print(f"   ID {item_id:5d} ({asin}): {rating:.2f}⭐ ({count:4d} reviews)")

