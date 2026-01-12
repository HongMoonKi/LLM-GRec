import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp


class LightGCN(torch.nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    
    Graph structure:
    - User nodes: 0~usernum (usernum+1 nodes, user 0 is padding)
    - Item nodes: (usernum+1)~(usernum+itemnum) (itemnum nodes)
    - Item IDs are 1-indexed (1~itemnum, item 0 is padding)
    """
    
    def __init__(self, user_num, item_num, args):
        super(LightGCN, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.embedding_dim = args.hidden_units
        self.num_layers = args.num_blocks
        self.keep_prob = 1 - args.dropout_rate
        self.args = args

        # ✅ Embedding with Xavier initialization (논문 방식)
        self.user_emb = nn.Embedding(user_num + 1, args.hidden_units, padding_idx=0)
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        
        nn.init.xavier_normal_(self.user_emb.weight.data)
        nn.init.xavier_normal_(self.item_emb.weight.data)

        self.graph = None
        self.graph_size = None

    def build_graph(self, user_train, usernum, itemnum):
        """
        Build normalized user-item bipartite graph
        
        Args:
            user_train: dict {user_id: [item_id, ...]}
            usernum: number of users (excluding padding 0)
            itemnum: number of items (excluding padding 0)
        """
        rows, cols = [], []
        data = []

        for user_id in user_train:
            for item_id in user_train[user_id]:
                # User → Item edge
                rows.append(user_id)
                cols.append(usernum + item_id)
                data.append(1.0)

                # Item → User edge (bidirectional)
                rows.append(usernum + item_id)
                cols.append(user_id)
                data.append(1.0)

        # Graph size: user 0~usernum + item (usernum+1)~(usernum+itemnum)
        n_nodes = usernum + itemnum + 1
        adj_mat = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        norm_adj_coo = norm_adj.tocoo()

        # Convert to PyTorch sparse tensor
        indices = torch.LongTensor(np.vstack((norm_adj_coo.row, norm_adj_coo.col))).to(self.dev)
        values = torch.FloatTensor(norm_adj_coo.data).to(self.dev)
        shape = torch.Size(norm_adj_coo.shape)

        self.graph = torch.sparse_coo_tensor(indices, values, shape, device=self.dev)
        self.graph_size = (usernum, itemnum)

    def _get_ego_embeddings(self):
        """
        Get initial embeddings for GCN propagation
        
        Returns:
            all_embeddings: [usernum+1+itemnum, hidden_dim]
                - [:usernum+1]: user embeddings (0~usernum)
                - [usernum+1:]: item embeddings (1~itemnum, excluding item 0)
        """
        usernum, itemnum = self.graph_size
        
        # User: 0~usernum (including padding 0)
        user_embeddings = self.user_emb.weight[:usernum+1]
        
        # Item: 1~itemnum (excluding padding 0)
        item_embeddings = self.item_emb.weight[1:itemnum+1]
        
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return all_embeddings

    def _gcn_propagate(self):
        """
        GCN propagation with layer combination
        
        Returns:
            all_embeddings: [usernum+1+itemnum, hidden_dim]
                Aggregated embeddings from all layers
        """
        all_embeddings = self._get_ego_embeddings()
        embeddings_list = [all_embeddings]

        # Multi-layer propagation
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.graph, all_embeddings)
            embeddings_list.append(all_embeddings)

        # Layer combination with uniform weights (논문: 1/(K+1))
        all_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        return all_embeddings

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode='default'):
        """
        Forward pass for training
        
        Args:
            user_ids: user IDs (0~usernum)
            log_seqs: not used in LightGCN
            pos_seqs: positive item IDs (1~itemnum)
            neg_seqs: negative item IDs (1~itemnum)
            mode: 'default' or 'item'
            
        Returns:
            pos_logits, neg_logits: BPR scores
        """
        # Convert tensor to numpy if needed
        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.cpu().numpy()
        if isinstance(pos_seqs, torch.Tensor):
            pos_seqs = pos_seqs.cpu().numpy()
        if isinstance(neg_seqs, torch.Tensor):
            neg_seqs = neg_seqs.cpu().numpy()
            
        user_ids = np.array(user_ids)
        usernum, itemnum = self.graph_size

        # GCN Propagation
        all_embeddings = self._gcn_propagate()

        # Split user and item embeddings
        user_all_embeddings = all_embeddings[:usernum+1]
        item_all_embeddings = all_embeddings[usernum+1:]

        # Clipping for safety
        user_ids = np.clip(user_ids, 0, usernum)
        pos_seqs = np.clip(pos_seqs, 1, itemnum)
        neg_seqs = np.clip(neg_seqs, 1, itemnum)

        # Get embeddings
        user_emb = user_all_embeddings[torch.LongTensor(user_ids).to(self.dev)]
        
        # Item indexing: 1-indexed → 0-indexed
        pos_embs = item_all_embeddings[torch.LongTensor(pos_seqs - 1).to(self.dev)]
        neg_embs = item_all_embeddings[torch.LongTensor(neg_seqs - 1).to(self.dev)]

        # Inner product
        pos_logits = (user_emb * pos_embs).sum(dim=-1)
        neg_logits = (user_emb * neg_embs).sum(dim=-1)

        if mode == 'item':
            return user_emb, pos_embs, neg_embs
        else:
            return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """
        Predict scores for given user-item pairs
        
        Args:
            user_ids: [batch_size]
            log_seqs: not used
            item_indices: [batch_size, num_items]
            
        Returns:
            logits: [batch_size, num_items]
        """
        if isinstance(user_ids, torch.Tensor):
            user_ids = user_ids.cpu().numpy()
        if isinstance(item_indices, torch.Tensor):
            item_indices = item_indices.cpu().numpy()
            
        user_ids = np.array(user_ids)
        usernum, itemnum = self.graph_size

        # GCN Propagation
        all_embeddings = self._gcn_propagate()

        user_all_embeddings = all_embeddings[:usernum+1]
        item_all_embeddings = all_embeddings[usernum+1:]

        user_emb = user_all_embeddings[torch.LongTensor(user_ids).to(self.dev)]

        item_indices = np.clip(item_indices, 1, itemnum)
        item_embs = item_all_embeddings[torch.LongTensor(item_indices - 1).to(self.dev)]

        logits = (user_emb.unsqueeze(1) * item_embs).sum(dim=-1)

        return logits

    def log2feats(self, log_seqs, user_ids=None):
        """
        Get user embeddings with GCN propagation
        
        ✅ Used as teacher embeddings in LLM-SRec
        
        Args:
            log_seqs: not used in LightGCN
            user_ids: user IDs (0~usernum)
            
        Returns:
            user_emb: [batch_size, hidden_dim]
                User embeddings with global graph structure
        """
        if user_ids is not None:
            if isinstance(user_ids, torch.Tensor):
                user_ids = user_ids.cpu().numpy()
            user_ids = np.array(user_ids)
        else:
            user_ids = np.array([])
            
        usernum, itemnum = self.graph_size

        # GCN Propagation
        all_embeddings = self._gcn_propagate()

        user_all_embeddings = all_embeddings[:usernum+1]
        
        user_ids = np.clip(user_ids, 0, usernum)
        user_emb = user_all_embeddings[torch.LongTensor(user_ids).to(self.dev)]

        return user_emb

    def get_all_embeddings(self):
        """
        Get all user and item embeddings (for fast evaluation)
        
        Returns:
            user_embeddings: [usernum+1, hidden_dim]
            item_embeddings: [itemnum, hidden_dim]
        """
        usernum, itemnum = self.graph_size
        
        all_embeddings = self._gcn_propagate()
        
        user_embeddings = all_embeddings[:usernum+1]
        item_embeddings = all_embeddings[usernum+1:]
        
        return user_embeddings, item_embeddings