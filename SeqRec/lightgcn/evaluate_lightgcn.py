import os
import sys
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from model import LightGCN
from utils import data_partition


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Industrial_and_Scientific')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='./CDs_and_Vinyl/lightgcn_best.pth')
    
    # Model hyperparameters (should match training)
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--maxlen', type=int, default=128)
    
    # Evaluation
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of users to evaluate')

    args = parser.parse_args()

    if args.device == 'hpu':
        args.device = torch.device('hpu')
    else:
        args.device = torch.device(f'cuda:{args.device}')

    return args


def merge_train_valid(user_train, user_valid):
    """
    Merge train and valid for fair comparison with SASRec
    
    Args:
        user_train: dict {user_id: [item_ids]}
        user_valid: dict {user_id: [item_id]}
        
    Returns:
        merged: dict {user_id: [item_ids]}
    """
    merged = defaultdict(list)
    
    # Copy train
    for u, items in user_train.items():
        merged[u] = items.copy()
    
    # Add valid
    for u, items in user_valid.items():
        if len(items) > 0:
            merged[u].extend(items)
    
    return merged


def evaluate_fair(model, user_train, user_valid, user_test, usernum, itemnum, maxlen, num_samples=10000):
    """
    Fair evaluation matching SASRec protocol
    - Use train+valid for graph
    - Exclude train+valid+test from negative sampling
    
    Args:
        model: LightGCN model
        user_train: training interactions
        user_valid: validation interactions
        user_test: test interactions
        usernum: number of users
        itemnum: number of items
        maxlen: max sequence length (not used in LightGCN)
        num_samples: number of users to evaluate
        
    Returns:
        hr@10, ndcg@10
    """
    model.eval()
    
    with torch.no_grad():
        # Get all embeddings once
        user_embeddings, item_embeddings = model.get_all_embeddings()
        
        hits, ndcgs = [], []
        
        # Get test users
        test_users = list(user_test.keys())
        if len(test_users) > num_samples:
            test_users = random.sample(test_users, num_samples)
        
        print(f"\n{'='*70}")
        print(f"Evaluating on {len(test_users):,} users...")
        print(f"{'='*70}\n")
        
        for u_id in tqdm(test_users, desc="Evaluating", ncols=100):
            if len(user_test[u_id]) == 0:
                continue

            pos_item = user_test[u_id][0]

            # ✅ Exclude train + valid + test from negative sampling (same as SASRec)
            his = set(user_train.get(u_id, [])) | set(user_valid.get(u_id, [])) | set([pos_item])
            his.add(0)  # Add padding
            
            # Generate 99 negative samples
            neg_items = []
            items = set(range(1, itemnum + 1))
            available_items = list(items.difference(his))
            
            if len(available_items) > 99:
                neg_items = random.sample(available_items, 99)
            else:
                neg_items = available_items

            candidates = [pos_item] + neg_items
            
            # Get embeddings
            user_emb = user_embeddings[u_id]  # [hidden_dim]
            candidate_embs = item_embeddings[
                torch.LongTensor([c-1 for c in candidates]).to(model.dev)
            ]  # [100, hidden_dim]
            
            # Compute scores
            logits = torch.matmul(candidate_embs, user_emb)  # [100]
            rank = (logits.cpu().numpy().argsort()[::-1] == 0).argmax()

            if rank < 10:
                hits.append(1)
                ndcgs.append(1.0 / np.log2(rank + 2))
            else:
                hits.append(0)
                ndcgs.append(0.0)

        hr = np.mean(hits) if hits else 0.0
        ndcg = np.mean(ndcgs) if ndcgs else 0.0
    
    return hr, ndcg


def main():
    args = create_args()
    
    print("\n" + "="*70)
    print("LightGCN Fair Evaluation (train+valid graph)")
    print("="*70)
    print(f"Dataset:     {args.dataset}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Device:      {args.device}")
    print(f"Num samples: {args.num_samples:,}")
    print("="*70 + "\n")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    # Load dataset
    print("Loading dataset...")
    dataset = data_partition(args.dataset, args, path=f'../data_{args.dataset}/{args.dataset}')
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset

    print(f"✅ Dataset loaded:")
    print(f"   Users: {usernum:,}")
    print(f"   Items: {itemnum:,}")
    print(f"   Test users: {len(user_test):,}\n")

    # ✅ Merge train + valid
    print("Merging train + valid...")
    user_train_valid = merge_train_valid(user_train, user_valid)
    
    train_only_edges = sum(len(items) for items in user_train.values())
    train_valid_edges = sum(len(items) for items in user_train_valid.values())
    print(f"   Train edges: {train_only_edges:,}")
    print(f"   Train+Valid edges: {train_valid_edges:,}")
    print(f"   Added: {train_valid_edges - train_only_edges:,} edges\n")

    # Initialize model
    print("Loading model...")
    model = LightGCN(usernum, itemnum, args).to(args.device)

    # ✅ Build graph with train+valid
    print("Building graph with train+valid...")
    model.build_graph(user_train_valid, usernum, itemnum)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    print("✅ Model loaded successfully!\n")

    # Evaluate on test set (fair comparison with SASRec)
    hr, ndcg = evaluate_fair(
        model, 
        user_train,
        user_valid,
        user_test, 
        usernum, 
        itemnum, 
        args.maxlen,
        num_samples=args.num_samples
    )

    print(f"\n{'='*70}")
    print("Fair Evaluation Results (train+valid graph)")
    print(f"{'='*70}")
    print(f"HR@10:   {hr:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()