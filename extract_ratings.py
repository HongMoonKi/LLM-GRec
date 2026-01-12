import os
import pickle
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict


def extract_ratings(dataset_name='Industrial_and_Scientific'):
    """Extract ratings from Amazon-Reviews-2023"""
    
    print(f"\n{'='*70}")
    print(f"⭐ Extracting Ratings: {dataset_name}")
    print(f"{'='*70}")
    
    output_file = f'./SeqRec/data_{dataset_name}/item_ratings.pkl'
    
    # Load REVIEW dataset (not meta!)
    print(f"\n📥 Downloading review dataset from HuggingFace...")
    try:
        review_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            f"raw_review_{dataset_name}",  
            trust_remote_code=True,
            split='full'
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None
    
    print(f"✅ Loaded {len(review_dataset):,} reviews")
    
    # Extract ratings per item
    print(f"\n🔍 Extracting ratings...")
    item_ratings = defaultdict(list)  # {parent_asin: [rating1, rating2, ...]}
    
    for review in tqdm(review_dataset, desc="Processing"):
        parent_asin = review.get('parent_asin')
        rating = review.get('rating')
        
        if parent_asin and rating is not None:
            item_ratings[parent_asin].append(rating)
    
    # Calculate average ratings
    item_avg_ratings = {}
    for asin, ratings in item_ratings.items():
        item_avg_ratings[asin] = {
            'avg_rating': sum(ratings) / len(ratings),
            'num_ratings': len(ratings),
            'rating_std': (sum((r - sum(ratings)/len(ratings))**2 for r in ratings) / len(ratings))**0.5
        }
    
    # Statistics
    print(f"\n{'='*70}")
    print(f"📊 Extraction Results")
    print(f"{'='*70}")
    print(f"Total reviews:          {len(review_dataset):,}")
    print(f"Unique items:           {len(item_avg_ratings):,}")
    print(f"Avg reviews per item:   {len(review_dataset) / len(item_avg_ratings):.2f}")
    
    # Rating distribution
    all_ratings = [r['avg_rating'] for r in item_avg_ratings.values()]
    print(f"\n⭐ Rating Statistics:")
    print(f"   Average rating:      {sum(all_ratings)/len(all_ratings):.2f}")
    print(f"   Min rating:          {min(all_ratings):.2f}")
    print(f"   Max rating:          {max(all_ratings):.2f}")
    
    # Sample items
    print(f"\n📦 Sample Items with Ratings:")
    for i, (asin, info) in enumerate(list(item_avg_ratings.items())[:5], 1):
        print(f"   {i}. {asin}: {info['avg_rating']:.2f}⭐ ({info['num_ratings']} reviews)")
    
    # Save
    if len(item_avg_ratings) > 0:
        print(f"\n💾 Saving ratings...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(item_avg_ratings, f)
        
        print(f"✅ Saved to: {output_file}")
        print(f"{'='*70}\n")
        return item_avg_ratings
    else:
        print(f"\n❌ No ratings to save")
        return None


if __name__ == "__main__":
    import sys
    
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'Industrial_and_Scientific'
    
    print(f"\n🚀 Starting rating extraction...")
    print(f"   Dataset: {dataset_name}")
    
    result = extract_ratings(dataset_name)
    
    if result and len(result) > 0:
        print(f"\n🎉 Rating extraction completed!")
        print(f"   You can now use ratings as node features in LightGCN")
    else:
        print(f"\n❌ Rating extraction failed")