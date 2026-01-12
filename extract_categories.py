import os
import pickle
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict


def extract_categories(dataset_name='Electronics'):
    """Extract categories from Amazon-Reviews-2023"""
    
    print(f"\n{'='*70}")
    print(f"📦 Extracting Categories: {dataset_name}")
    print(f"{'='*70}")
    
    output_file = f'./SeqRec/data_{dataset_name}/item_categories.pkl'
    
    # Load dataset
    print(f"\n📥 Downloading meta dataset from HuggingFace...")
    try:
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            f"raw_meta_{dataset_name}", 
            trust_remote_code=True,
            split='full'
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None
    
    print(f"✅ Loaded {len(meta_dataset):,} items")
    
    # Extract categories
    print(f"\n🔍 Extracting categories...")
    item_to_category = {}
    category_counts = defaultdict(int)
    no_category_count = 0
    
    for item in tqdm(meta_dataset, desc="Processing"):
        parent_asin = item.get('parent_asin')
        categories = item.get('categories', [])
        
        # ✅ 수정: categories는 list of strings
        # categories[0] = 'Industrial & Scientific' (메인 카테고리)
        # categories[1] = 'Food Service Equipment' (서브 카테고리) ← 이걸 사용!
        if parent_asin and isinstance(categories, list) and len(categories) >= 2:
            category = categories[1]  # ✅ 두 번째 카테고리 사용
            item_to_category[parent_asin] = category
            category_counts[category] += 1
        else:
            no_category_count += 1
    
    # Statistics
    print(f"\n{'='*70}")
    print(f"📊 Extraction Results")
    print(f"{'='*70}")
    print(f"Total items:            {len(meta_dataset):,}")
    print(f"Items with category:    {len(item_to_category):,}")
    print(f"Items without category: {no_category_count:,}")
    print(f"Unique categories:      {len(category_counts):,}")
    
    # Top categories
    print(f"\n📈 Top 10 Categories:")
    for i, (cat, count) in enumerate(
        sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10], 1
    ):
        print(f"   {i:2d}. {cat:50s} {count:6,} items")
    
    # Sample items
    print(f"\n📦 Sample Items:")
    for i, (asin, cat) in enumerate(list(item_to_category.items())[:5], 1):
        print(f"   {i}. {asin}: {cat}")
    
    # Save
    if len(item_to_category) > 0:
        print(f"\n💾 Saving categories...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(item_to_category, f)
        
        print(f"✅ Saved to: {output_file}")
        print(f"{'='*70}\n")
        return item_to_category
    else:
        print(f"\n❌ No categories to save")
        return None


if __name__ == "__main__":
    import sys
    
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'Electronics'
    
    print(f"\n🚀 Starting category extraction...")
    print(f"   Dataset: {dataset_name}")
    
    result = extract_categories(dataset_name)
    
    if result and len(result) > 0:
        print(f"\n🎉 Category extraction completed!")
        print(f"   You can now use: --use_category --category_dim 32")
    else:
        print(f"\n❌ Category extraction failed")