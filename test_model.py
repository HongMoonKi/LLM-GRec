# test_model.py - Test specific epoch checkpoint
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

from models.seqllm_model import llmrec_model
from SeqRec.lightgcn.utils import data_partition


def evaluate_model(model, user_train, user_valid, user_test, desc="Test"):
    """평가 함수"""
    model.eval()
    model.users = 0.0
    model.NDCG = 0.0
    model.HT = 0.0
    model.NDCG_20 = 0.0
    model.HIT_20 = 0.0
    model.all_embs = None
    
    # Test 데이터가 있는 user만
    test_users = [u for u in user_test.keys() 
                  if u in user_train and len(user_test[u]) > 0]
    
    # ✅ 논문: 10,000명 샘플링
    if len(test_users) > 10000:
        import random
        test_users = random.sample(test_users, 10000)
    
    print(f"\n{'─'*70}")
    print(f"{desc}: {len(test_users):,} users")
    print(f"{'─'*70}")
    
    with torch.no_grad():
        for u in tqdm(test_users, desc=desc, leave=False):
            # Input: train + valid (전체)
            user_history = list(user_train[u])
            if u in user_valid and len(user_valid[u]) > 0:
                user_history = user_history + user_valid[u]
            
            # Target: test의 첫 번째 item
            pos_item = user_test[u][0]
            
            # generate_batch 호출
            model(
                [np.array([u]), [user_history], [pos_item], [0], 0, None, 'original'],
                mode='generate_batch'
            )
    
    ndcg_10 = model.NDCG / model.users if model.users > 0 else 0
    hr_10 = model.HT / model.users if model.users > 0 else 0
    ndcg_20 = model.NDCG_20 / model.users if model.users > 0 else 0
    hr_20 = model.HIT_20 / model.users if model.users > 0 else 0
    
    return ndcg_10, hr_10, ndcg_20, hr_20


def test_checkpoint(args, epoch):
    """특정 에포크 체크포인트 테스트"""
    
    print("\n" + "="*70)
    print(f"Testing Epoch {epoch} Checkpoint")
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
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = llmrec_model(args, user_train=user_train).to(args.device)
    
    # Load checkpoint
    print(f"\n📂 Loading checkpoint from epoch {epoch}...")
    dataset_name = args.rec_pre_trained_data
    
    # 체크포인트 파일명
    ckpt_files = {
        'CLS': f'{dataset_name}_llama-3b_{epoch}_CLS.pt',
        'CLS_item': f'{dataset_name}_llama-3b_{epoch}_CLS_item.pt',
        'item_proj': f'{dataset_name}_llama-3b_{epoch}_item_proj.pt',
        'pred_item': f'{dataset_name}_llama-3b_{epoch}_pred_item.pt',
        'pred_user': f'{dataset_name}_llama-3b_{epoch}_pred_user.pt'
    }
    
    # 파일 존재 확인 및 경로 찾기 (best 디렉토리 우선)
    all_exist = True
    for name, filename in ckpt_files.items():
        if os.path.exists(filename):
            ckpt_files[name] = filename
        elif os.path.exists(f'./models/{args.save_dir}best/{filename}'):     
            ckpt_files[name] = f'./models/{args.save_dir}best/{filename}'
        elif os.path.exists(f'./models/{args.save_dir}/{filename}'):
            ckpt_files[name] = f'./models/{args.save_dir}/{filename}'
        else:
            print(f"❌ File not found: {filename}")
            all_exist = False
    
    if not all_exist:
        print(f"\n❌ Some checkpoint files are missing!")
        return
    
    print(f"✅ All checkpoint files found!")
    
    try:
        # Load weights
        model.item_emb_proj.load_state_dict(
            torch.load(ckpt_files['item_proj'], map_location=args.device)
        )
        print(f"   ✅ Loaded item_emb_proj")
        
        model.llm.pred_user.load_state_dict(
            torch.load(ckpt_files['pred_user'], map_location=args.device)
        )
        print(f"   ✅ Loaded pred_user")
        
        model.llm.pred_item.load_state_dict(
            torch.load(ckpt_files['pred_item'], map_location=args.device)
        )
        print(f"   ✅ Loaded pred_item")
        
        if not args.token:
            if args.nn_parameter:
                model.llm.CLS.data = torch.load(
                    ckpt_files['CLS'], map_location=args.device
                )
                model.llm.CLS_item.data = torch.load(
                    ckpt_files['CLS_item'], map_location=args.device
                )
            else:
                model.llm.CLS.load_state_dict(
                    torch.load(ckpt_files['CLS'], map_location=args.device)
                )
                model.llm.CLS_item.load_state_dict(
                    torch.load(ckpt_files['CLS_item'], map_location=args.device)
                )
            print(f"   ✅ Loaded CLS tokens")
        
        print(f"\n✅ Checkpoint loaded successfully!")
        
    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")
        return
    
    # Test
    print(f"\n📊 Running evaluation...")
    test_ndcg10, test_hr10, test_ndcg20, test_hr20 = evaluate_model(
        model, user_train, user_valid, user_test, desc=f"Testing Epoch {epoch}"
    )
    
    # Results
    print(f"\n" + "="*70)
    print(f"Test Results (Epoch {epoch})")
    print(f"="*70)
    print(f"  NDCG@10: {test_ndcg10:.4f}")
    print(f"  HR@10:   {test_hr10:.4f}")
    print(f"  NDCG@20: {test_ndcg20:.4f}")
    print(f"  HR@20:   {test_hr20:.4f}")
    print(f"="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Test settings
    parser.add_argument("--epoch", type=int, required=True,
                       help='Epoch number to test (e.g., 5)')
    
    # Model settings (must match training config)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument("--llm", type=str, default='llama-3b')
    parser.add_argument("--recsys", type=str, default='lightgcn')
    parser.add_argument("--rec_pre_trained_data", type=str, 
                       default='Industrial_and_Scientific')
    parser.add_argument("--save_dir", type=str, 
                       default='model_lightgcn_rating_category_v2')
    
    # Model architecture (must match training)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument('--maxlen', default=128, type=int)
    parser.add_argument('--nn_parameter', default=False, action='store_true')
    parser.add_argument('--token', default=False, action='store_true')
    
    # Category settings (must match training)
    parser.add_argument('--use_category', default=False, action='store_true')
    parser.add_argument('--category_dim', default=32, type=int)
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'hpu':
        args.device = torch.device('hpu')
    else:
        args.device = torch.device(f'cuda:{args.device}')
    
    # Run test
    test_checkpoint(args, args.epoch)