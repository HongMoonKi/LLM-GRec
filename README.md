# LLM-GRec: Enhancing LLM-based Recommendation with Graph Neural Networks

This repository is designed for implementing LLM-GRec.

## Overview

LLM-GRec is a method that enhances the integration of collaborative filtering information into LLMs by distilling the user representations extracted from a pre-trained LightGCN model into LLMs. Unlike sequential recommendation approaches, we leverage graph neural networks to capture user-item interaction patterns.

- We use LLaMA-3.2-3b-instruct.

## Env Setting
```
conda create -n [env name] pip
conda activate [env name]
pip install -r requirements.txt
```

## 1. Download Dataset

The data ([Amazon 2023](https://amazon-reviews-2023.github.io/)) is automatically downloaded when using the SASRec training code.
```
cd SeqRec/sasrec
python main.py --device 0 --dataset Industrial_and_Scientific
```

Available datasets: `Industrial_and_Scientific`, `CDs_and_Vinyl`, `Movies_and_TV`

## 2. Pre-train LightGCN
```
cd SeqRec/lightgcn
python main.py --device 0 --dataset Industrial_and_Scientific
```

## 3. Extract Metadata
```
python extract_categories.py
python extract_ratings.py
python create_mappings.py
```

## 4. Train LLM-GRec

The model saves when the best validation score is reached during training and performs inference on the test set.
```
python main.py --device 0 --train \
  --rec_pre_trained_data CDs_and_Vinyl \
  --eval_min_rating 0.0 \
  --history_sampling hybrid \
  --save_dir model_cds \
  --distill_loss cosine \
  --use_category --category_dim 32 \
  --batch_size 20 --num_epochs 10
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--rec_pre_trained_data` | Dataset name (`Industrial_and_Scientific`, `CDs_and_Vinyl`, `Movies_and_TV`) |
| `--history_sampling` | Sampling strategy (`hybrid`) |
| `--distill_loss` | Distillation loss type (`cosine`) |
| `--use_category` | Enable category embeddings |
| `--category_dim` | Category embedding dimension |