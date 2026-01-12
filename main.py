import os
import sys
import argparse
import torch
from utils import *
from train_model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Multi-GPU settings
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument("--world_size", type=int, default=8)
    
    # Model settings
    parser.add_argument("--llm", type=str, default='llama-3b',  
                       help='llama-3b, llama (8B), flan_t5')
    parser.add_argument("--recsys", type=str, default='lightgcn', 
                       help='lightgcn (recommended), sasrec')
    parser.add_argument("--rec_pre_trained_data", type=str, default='Industrial_and_Scientific',
                       help='Industrial_and_Scientific, CDs_and_Vinyl, Movies_and_TV')
    parser.add_argument("--recsys_data", type=str, default=None,
                    help='RecSys checkpoint (e.g., CDs_and_Vinyl_filtered3)')


    
    # Training mode
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--extract", action='store_true')
    parser.add_argument("--token", action='store_true')
    
    # Save directory
    parser.add_argument("--save_dir", type=str, default='model_lightgcn_3b')  

    # ✅ Training hyperparameters (aligned with LLM-SRec paper)
    parser.add_argument('--batch_size', default=20, type=int,
                       help='Batch size for training (paper: 20)')
    parser.add_argument('--batch_size_infer', default=20, type=int,
                       help='Batch size for inference (paper: 20)')
    parser.add_argument('--num_epochs', default=10, type=int,  
                       help='Number of training epochs (paper: 10)')
    parser.add_argument("--stage2_lr", type=float, default=0.0001,
                       help='Learning rate for LLM-SRec training (paper: 1e-4)')

    parser.add_argument('--distill_loss', type=str, default='mse',
                        choices=['mse', 'cosine'],
                        help='Distillation loss type: mse or cosine')                       
    
    # ✅ LightGCN specific parameters (should match pre-trained model)
    parser.add_argument("--hidden_units", type=int, default=64,
                       help='Embedding dimension (must match LightGCN checkpoint)')
    parser.add_argument("--num_blocks", type=int, default=3,
                       help='Number of GCN layers (must match LightGCN checkpoint)')
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                       help='Dropout rate (0.0 for LightGCN)')
    
    # Additional parameters
    parser.add_argument('--l2_emb', type=float, default=0.0,
                       help='L2 regularization (optional)')
    parser.add_argument('--maxlen', default=128, type=int,
                       help='Maximum sequence length')
    parser.add_argument('--nn_parameter', default=False, action='store_true')
    
    # ✅ 추가: Category embedding 관련 (나중에 사용)
    parser.add_argument('--use_category', default=False, action='store_true',
                       help='Use category information')
    parser.add_argument('--category_dim', default=32, type=int,
                       help='Category embedding dimension')

    parser.add_argument('--history_sampling', type=str, default='recent',
                   choices=['recent', 'hybrid'],
                   help='User history sampling: recent (baseline) or hybrid (diverse)')                   

    parser.add_argument('--eval_min_rating', type=float, default=0.0,
                   help='Minimum rating for evaluation targets (0=all, 4.0=high-quality)')

    args = parser.parse_args()
    if args.recsys_data is None:
        args.recsys_data = args.rec_pre_trained_data

    # Device setup
    if args.device == 'hpu':
        args.device = torch.device('hpu')
    else:
        args.device = torch.device(f'cuda:{args.device}')

    # Print configuration
    print("\n" + "="*60)
    print("LLM-SRec with LightGCN Training Configuration")
    print("="*60)
    print(f"LLM Model:           {args.llm}")
    print(f"RecSys Encoder:      {args.recsys}")
    print(f"Dataset:             {args.rec_pre_trained_data}")
    print(f"Device:              {args.device}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Num Epochs:          {args.num_epochs}")
    print(f"Learning Rate:       {args.stage2_lr}")
    print(f"Hidden Units:        {args.hidden_units}")
    print(f"Num Blocks:          {args.num_blocks}")
    print(f"Save Directory:      {args.save_dir}")
    print(f"Use Category:        {args.use_category}")  # ✅ 추가
    print("="*60 + "\n")

    if args.train:
        train_model(args)