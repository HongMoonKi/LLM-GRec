import contextlib
import logging
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from SeqRec.lightgcn.model import LightGCN


def load_checkpoint(recsys, pre_trained):
    path = f'./SeqRec/{recsys}/{pre_trained}/'
    pth_file_path = find_filepath(path, '.pth')

    if len(pth_file_path) == 0:
        raise FileNotFoundError(
            f"No .pth found in {path}. Train LightGCN first."
        )
    if len(pth_file_path) > 1:
        raise ValueError("Multiple .pth files found. Keep only one.")

    checkpoint = torch.load(pth_file_path[0], map_location="cpu")
    return checkpoint


class RecSys(nn.Module):
    """
    RecSys wrapper for LightGCN teacher encoder
    - Loads checkpoint (E0)
    - Builds graph
    - Provides propagation-based E_final embeddings
    """
    def __init__(self, recsys_model, pre_trained_data, device, args=None, user_train=None):
        super().__init__()

        if user_train is None:
            raise ValueError("RecSys must receive user_train to build graph")

        # Load checkpoint (E0)
        checkpoint = load_checkpoint(recsys_model, pre_trained_data)

        user_num = checkpoint["user_emb.weight"].shape[0] - 1
        item_num = checkpoint["item_emb.weight"].shape[0] - 1

        # Initialize LightGCN with E0
        model = LightGCN(user_num, item_num, args).to(device)
        model.load_state_dict(checkpoint)

        # L2 normalize (teacher quality ↑)
        with torch.no_grad():
            model.user_emb.weight.data = F.normalize(model.user_emb.weight.data, p=2, dim=1)
            model.item_emb.weight.data = F.normalize(model.item_emb.weight.data, p=2, dim=1)

        # Build graph (필수!!!)
        model.build_graph(user_train, user_num, item_num)

        # Freeze
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_units = args.hidden_units if args else 64

    # ----------------------------------------
    # 🔥 반드시 필요한 teacher embedding 함수
    # ----------------------------------------
    def get_all_embeddings(self):
        """
        Returns:
            user_all: [num_users+1, hidden_dim]
            item_all: [num_items+1, hidden_dim]
        """
        return self.model.get_all_embeddings()

    def get_user_emb(self, user_ids):
        """Propagation된 E_final user embedding 반환"""
        user_all, _ = self.get_all_embeddings()
        return user_all[user_ids]

    def get_item_emb(self, item_ids):
        """Propagation된 E_final item embedding 반환"""
        _, item_all = self.get_all_embeddings()
        return item_all[item_ids]  # item_ids already 1-indexed in your pipeline

    # (optional) default forward: do not use for LLMRec
    def forward(self, user_ids, item_ids):
        return self.get_user_emb(user_ids), self.get_item_emb(item_ids)
