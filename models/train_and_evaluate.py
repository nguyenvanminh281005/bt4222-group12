"""
NRMS Model Training, Validation, and Comprehensive Evaluation Script
BT4222 Group 12 - MIND-small News Recommendation
"""

import os
import sys
import re
import math
import time
import random
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Configure style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
# 1. Attention Pooling and NRMS Modules
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """
    Additive Attention (Attention Pooling) layer as formulated in NRMS (Wu et al., 2019).
    Computes energy a_i = q^T tanh(W * h_i + b), followed by softmax normalization.
    """
    def __init__(self, d_model=256, query_dim=200):
        super().__init__()
        self.proj = nn.Linear(d_model, query_dim)
        self.query = nn.Parameter(torch.empty(query_dim).uniform_(-0.1, 0.1))
        
    def forward(self, x, mask=None):
        # x: [B, L, D]
        energy = torch.tanh(self.proj(x)) # [B, L, query_dim]
        scores = torch.matmul(energy, self.query) # [B, L]
        if mask is not None:
            all_pad = (~mask).all(dim=-1, keepdim=True)
            scores = scores.masked_fill(~mask, -1e9)
        alpha = F.softmax(scores, dim=-1) # [B, L]
        if mask is not None:
            alpha = alpha.masked_fill(all_pad, 0.0)
        out = torch.sum(x * alpha.unsqueeze(-1), dim=1) # [B, D]
        return out, alpha


class NewsEncoder(nn.Module):
    """
    News Encoder:
    Word Embedding (GloVe 300d) -> Projection (256d) -> Multi-Head Self-Attention (16 heads) -> Additive Attention (200d query)
    """
    def __init__(self, pretrained_embed, d_model=256, n_heads=16, query_dim=200, dropout=0.2, use_self_attn=True):
        super().__init__()
        self.use_self_attn = use_self_attn
        self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(pretrained_embed.shape[1], d_model)
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.pooling = AttentionPooling(d_model, query_dim)
        
    def forward(self, tokens, mask=None, return_weights=False):
        # tokens: [N, L]
        if mask is None:
            mask = (tokens > 0)
        x = self.embedding(tokens)
        x = self.dropout(x)
        x = self.proj(x)
        
        if self.use_self_attn:
            key_padding_mask = ~mask
            all_pad = key_padding_mask.all(dim=-1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad] = False
            attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
            attn_out = self.dropout(attn_out)
        else:
            attn_out = x
            
        vec, alpha = self.pooling(attn_out, mask=mask)
        if return_weights:
            return vec, alpha
        return vec


class UserEncoder(nn.Module):
    """
    User Encoder:
    News-level Multi-Head Self-Attention (16 heads) -> Additive Attention (200d query)
    """
    def __init__(self, d_model=256, n_heads=16, query_dim=200, dropout=0.2, use_self_attn=True):
        super().__init__()
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.pooling = AttentionPooling(d_model, query_dim)
        
    def forward(self, news_vecs, mask=None, return_weights=False):
        # news_vecs: [B, P, D]
        if self.use_self_attn:
            if mask is not None:
                key_padding_mask = ~mask
                all_pad = key_padding_mask.all(dim=-1)
                if all_pad.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_pad] = False
            else:
                key_padding_mask = None
            attn_out, _ = self.self_attn(news_vecs, news_vecs, news_vecs, key_padding_mask=key_padding_mask)
            attn_out = self.dropout(attn_out)
        else:
            attn_out = news_vecs
            
        vec, alpha = self.pooling(attn_out, mask=mask)
        if return_weights:
            return vec, alpha
        return vec


class NRMS(nn.Module):
    """
    Complete NRMS Architecture:
    Shared News Encoder + User Encoder + Dot Product Click Predictor
    """
    def __init__(self, pretrained_embed, news_title_lookup, d_model=256, n_heads=16, query_dim=200, dropout=0.2, use_self_attn=True):
        super().__init__()
        self.register_buffer('news_title_lookup', news_title_lookup)
        self.news_encoder = NewsEncoder(pretrained_embed, d_model, n_heads, query_dim, dropout, use_self_attn=use_self_attn)
        self.user_encoder = UserEncoder(d_model, n_heads, query_dim, dropout, use_self_attn=use_self_attn)
        
    def forward(self, hist_ids, cand_ids):
        # hist_ids: [B, P], cand_ids: [B, C]
        B, P = hist_ids.shape
        C = cand_ids.shape[1]
        
        all_ids = torch.cat([hist_ids.view(-1), cand_ids.view(-1)])
        unique_ids, inv = torch.unique(all_ids, return_inverse=True)
        
        unique_tokens = self.news_title_lookup[unique_ids]
        unique_vecs = self.news_encoder(unique_tokens)
        
        all_vecs = unique_vecs[inv]
        hist_news_vecs = all_vecs[:B * P].view(B, P, -1)
        cand_news_vecs = all_vecs[B * P:].view(B, C, -1)
        
        hist_mask = (hist_ids > 0)
        user_vecs = self.user_encoder(hist_news_vecs, mask=hist_mask)
        
        scores = torch.bmm(cand_news_vecs, user_vecs.unsqueeze(2)).squeeze(2)
        return scores


class TrainDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        h, c = self.samples[idx]
        return torch.tensor(h, dtype=torch.long), torch.tensor(c, dtype=torch.long)


# ---------------------------------------------------------------------------
# 2. Ranking Evaluation Metrics
# ---------------------------------------------------------------------------
def eval_impression(labels, scores):
    """
    Evaluates AUC, MRR, NDCG@5, and NDCG@10 for a single impression slate.
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    
    pos_mask = (labels == 1)
    n_pos = pos_mask.sum()
    n_neg = len(labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        order = np.argsort(scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        auc = (ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        
    desc_order = np.argsort(-scores)
    sorted_labels = labels[desc_order]
    
    pos_ranks = np.where(sorted_labels == 1)[0]
    mrr = 1.0 / (pos_ranks[0] + 1) if len(pos_ranks) > 0 else 0.0
    
    def ndcg_at_k(k):
        top_k = sorted_labels[:k]
        gains = top_k / np.log2(np.arange(2, len(top_k) + 2))
        dcg = gains.sum()
        
        ideal_top_k = np.sort(labels)[::-1][:k]
        ideal_gains = ideal_top_k / np.log2(np.arange(2, len(ideal_top_k) + 2))
        idcg = ideal_gains.sum()
        return dcg / idcg if idcg > 0 else 0.0
        
    return float(auc), float(mrr), float(ndcg_at_k(5)), float(ndcg_at_k(10))


@torch.no_grad()
def evaluate_model(model, eval_records, news_title_lookup, device, batch_size=2048):
    """
    Vectorized high-throughput NRMS evaluation:
    1. Pre-encodes all news articles once with NewsEncoder.
    2. Encodes all user history representations in batches.
    3. Computes candidate dot products and ranking metrics per impression.
    """
    model.eval()
    num_news = len(news_title_lookup)
    news_vecs = torch.zeros(num_news, 256, device=device)
    
    for i in range(0, num_news, batch_size):
        chunk = news_title_lookup[i:min(i + batch_size, num_news)].to(device)
        vecs = model.news_encoder(chunk)
        news_vecs[i:min(i + batch_size, num_news)] = vecs
        
    news_vecs[0] = 0.0 # Pad news
    
    num_eval = len(eval_records)
    hist_matrix = torch.zeros(num_eval, 50, dtype=torch.long, device=device)
    for idx, rec in enumerate(eval_records):
        hist_matrix[idx] = torch.tensor(rec['history'], dtype=torch.long, device=device)
        
    user_vecs = torch.zeros(num_eval, 256, device=device)
    user_batch_size = 1024
    for i in range(0, num_eval, user_batch_size):
        chunk_hist = hist_matrix[i:min(i + user_batch_size, num_eval)]
        chunk_news_vecs = news_vecs[chunk_hist] # [B, 50, 256]
        mask = (chunk_hist > 0)
        u_vecs = model.user_encoder(chunk_news_vecs, mask=mask)
        user_vecs[i:min(i + user_batch_size, num_eval)] = u_vecs
        
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    detailed_results = []
    
    for i, rec in enumerate(eval_records):
        cands = rec['candidates']
        labels = rec['labels']
        u_vec = user_vecs[i]
        c_vecs = news_vecs[cands]
        scores = torch.mv(c_vecs, u_vec).cpu().numpy()
        
        auc, mrr, n5, n10 = eval_impression(labels, scores)
        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(n5)
        ndcg10s.append(n10)
        
        detailed_results.append({
            'impression_id': rec['impression_id'],
            'timestamp': rec['timestamp'],
            'history_len': rec['history_len'],
            'hour': rec['timestamp'].hour if hasattr(rec['timestamp'], 'hour') else 0,
            'auc': auc,
            'mrr': mrr,
            'ndcg5': n5,
            'ndcg10': n10,
        })
        
    return {
        'auc': float(np.mean(aucs)),
        'mrr': float(np.mean(mrrs)),
        'ndcg5': float(np.mean(ndcg5s)),
        'ndcg10': float(np.mean(ndcg10s)),
        'detailed': detailed_results
    }


# ---------------------------------------------------------------------------
# 3. Main Training & Evaluation Workflow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NRMS Training & Evaluation on MIND-small")
    if os.path.basename(os.getcwd()) == "models":
        project_root = ".."
        data_dir = "../data"
        models_dir = "."
    else:
        project_root = "."
        data_dir = "data"
        models_dir = "models"
        
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    figures_dir = os.path.join(models_dir, "figures")
    summary_path = os.path.join(figures_dir, "results_summary.json")
    default_ckpt = os.path.join(checkpoints_dir, 'nrms_best_seed42.pt')

    parser.add_argument('--eval_only', action='store_true', help="Skip training and evaluate from checkpoint")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train")
    parser.add_argument('--checkpoint', type=str, default=default_ckpt, help="Checkpoint file path")
    args = parser.parse_args()

    print("=" * 70)
    print("BT4222 Group 12: NRMS Model Training & Comprehensive Evaluation")
    print("=" * 70)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 3.1 Load and Merge News Catalogue
    print("\n[1/7] Loading and Preprocessing News Catalogue...")
    t0 = time.time()
    train_news = pd.read_csv(os.path.join(data_dir, 'train/news.tsv'), sep='\t', header=None, 
                             names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
    dev_news = pd.read_csv(os.path.join(data_dir, 'dev/news.tsv'), sep='\t', header=None, 
                           names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
    all_news = pd.concat([train_news, dev_news]).drop_duplicates(subset=['news_id']).reset_index(drop=True)
    print(f"Loaded {len(all_news)} unique news articles (Train: {len(train_news)}, Dev: {len(dev_news)}).")
    
    def tokenize(text):
        if not isinstance(text, str):
            return []
        return re.findall(r'\w+|[^\w\s]', text.lower())
        
    word_counts = Counter()
    for title in all_news['title']:
        word_counts.update(tokenize(title))
        
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.items():
        if count >= 2:
            vocab[word] = len(vocab)
    print(f"Vocabulary size (min_freq >= 2): {len(vocab)} words.")
    
    news2idx = {'<PAD>': 0}
    idx2news = {0: '<PAD>'}
    for nid in all_news['news_id']:
        idx = len(news2idx)
        news2idx[nid] = idx
        idx2news[idx] = nid
        
    max_title_len = 30
    lookup = np.zeros((len(news2idx), max_title_len), dtype=np.int64)
    for row in all_news.itertuples(index=False):
        nid = row.news_id
        idx = news2idx[nid]
        tokens = [vocab.get(w, 1) for w in tokenize(row.title)][:max_title_len]
        lookup[idx, :len(tokens)] = tokens
        
    news_title_lookup = torch.tensor(lookup, dtype=torch.long, device=device)
    print(f"News lookup tensor: {news_title_lookup.shape} (Time: {time.time()-t0:.2f}s)")
    
    # 3.2 Load Pretrained GloVe Embeddings
    print("\n[2/7] Loading GloVe 300d Pretrained Word Embeddings...")
    t0 = time.time()
    matrix = np.random.normal(scale=0.1, size=(len(vocab), 300)).astype(np.float32)
    matrix[0] = 0.0 # PAD
    matched = 0
    with open(os.path.join(data_dir, 'glove/glove.6B.300d.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            w = parts[0]
            if w in vocab:
                matrix[vocab[w]] = np.asarray(parts[1:], dtype=np.float32)
                matched += 1
    pretrained_embeddings = torch.tensor(matrix, dtype=torch.float32)
    print(f"GloVe coverage: {matched}/{len(vocab)} ({matched/len(vocab)*100:.2f}%) in {time.time()-t0:.2f}s.")
    
    # 3.3 Load Behaviors and Strict Temporal Split
    print("\n[3/7] Parsing Behaviors and Creating Strict Temporal Partitions...")
    t0 = time.time()
    train_b = pd.read_csv(os.path.join(data_dir, 'train/behaviors.tsv'), sep='\t', header=None, 
                          names=['imp_id', 'user_id', 'time', 'history', 'impressions'])
    dev_b = pd.read_csv(os.path.join(data_dir, 'dev/behaviors.tsv'), sep='\t', header=None, 
                        names=['imp_id', 'user_id', 'time', 'history', 'impressions'])
    train_b['time'] = pd.to_datetime(train_b['time'])
    dev_b['time'] = pd.to_datetime(dev_b['time'])
    
    # Temporal Partitions
    # Train: Days 1-5 (2019-11-09 to 2019-11-13)
    train_df = train_b[train_b['time'] < '2019-11-14'].copy()
    # Validation: Day 6 (2019-11-14)
    val_df = train_b[train_b['time'] >= '2019-11-14'].copy()
    # Test: Day 7 (2019-11-15)
    test_df = dev_b.copy()
    
    print(f"Temporal Holdout Splits:")
    print(f"  • Train (Days 1–5):       {len(train_df):>7} impressions ({train_df['time'].min()} to {train_df['time'].max()})")
    print(f"  • Validation (Day 6):     {len(val_df):>7} impressions ({val_df['time'].min()} to {val_df['time'].max()})")
    print(f"  • Test (Day 7):           {len(test_df):>7} impressions ({test_df['time'].min()} to {test_df['time'].max()})")
    
    # Build Train Samples: 1 Positive + 4 Sampled Negatives within each impression
    max_hist_len = 50
    train_samples = []
    for row in train_df.itertuples(index=False):
        raw_h = row.history.split() if isinstance(row.history, str) else []
        hist_len = len(raw_h)
        hist = [news2idx[x] for x in raw_h if x in news2idx][-max_hist_len:]
        if len(hist) < max_hist_len:
            hist = [0] * (max_hist_len - len(hist)) + hist
            
        pos = []
        neg = []
        for item in row.impressions.split():
            nid, lbl = item.split('-')
            if nid in news2idx:
                if lbl == '1':
                    pos.append(news2idx[nid])
                else:
                    neg.append(news2idx[nid])
        if not pos or not neg:
            continue
            
        for p in pos:
            if len(neg) >= 4:
                sampled_neg = random.sample(neg, 4)
            else:
                sampled_neg = random.choices(neg, k=4)
            train_samples.append((hist, [p] + sampled_neg))
            
    print(f"Constructed {len(train_samples)} training instances (1 pos + 4 neg ratio).")
    
    def build_eval_records(df):
        records = []
        for row in df.itertuples(index=False):
            raw_h = row.history.split() if isinstance(row.history, str) else []
            hist_len = len(raw_h)
            hist = [news2idx[x] for x in raw_h if x in news2idx][-max_hist_len:]
            if len(hist) < max_hist_len:
                hist = [0] * (max_hist_len - len(hist)) + hist
                
            cands = []
            labels = []
            for item in row.impressions.split():
                nid, lbl = item.split('-')
                if nid in news2idx:
                    cands.append(news2idx[nid])
                    labels.append(int(lbl))
            if sum(labels) > 0 and len(labels) > sum(labels):
                records.append({
                    'impression_id': row.imp_id,
                    'timestamp': row.time,
                    'history': hist,
                    'history_len': hist_len,
                    'candidates': cands,
                    'labels': labels
                })
        return records
        
    val_records = build_eval_records(val_df)
    test_records = build_eval_records(test_df)
    print(f"Constructed {len(val_records)} validation records and {len(test_records)} test records in {time.time()-t0:.2f}s.")
    
    # 3.4 Model Training (Seed 42)
    print("\n[4/7] Training Original NRMS (Seed 42)...")
    train_loader = DataLoader(TrainDataset(train_samples), batch_size=128, shuffle=True, drop_last=True)
    
    model = NRMS(pretrained_embeddings, news_title_lookup, d_model=256, n_heads=16, query_dim=200, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = args.epochs
    history = {'train_loss': [], 'val_auc': [], 'val_mrr': [], 'val_ndcg5': [], 'val_ndcg10': []}
    best_ndcg5 = -1.0
    best_model_path = args.checkpoint
    
    if args.eval_only and os.path.exists(best_model_path):
        print(f"\n[4/7] --eval_only mode enabled. Loading existing checkpoint from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                prev = json.load(f)
                history = prev.get('history', history)
    else:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            t_epoch = time.time()
            for step, (hist_b, cand_b) in enumerate(train_loader):
                hist_b = hist_b.to(device)
                cand_b = cand_b.to(device)
                optimizer.zero_grad()
                scores = model(hist_b, cand_b)
                targets = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
                loss = F.cross_entropy(scores, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - t_epoch
            
            # Validation Evaluation
            t_val = time.time()
            val_metrics = evaluate_model(model, val_records, news_title_lookup, device)
            val_time = time.time() - t_val
            
            history['train_loss'].append(avg_loss)
            history['val_auc'].append(val_metrics['auc'])
            history['val_mrr'].append(val_metrics['mrr'])
            history['val_ndcg5'].append(val_metrics['ndcg5'])
            history['val_ndcg10'].append(val_metrics['ndcg10'])
            
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} ({epoch_time:.1f}s) | "
                  f"Val AUC: {val_metrics['auc']:.4f} | MRR: {val_metrics['mrr']:.4f} | "
                  f"NDCG@5: {val_metrics['ndcg5']:.4f} | NDCG@10: {val_metrics['ndcg10']:.4f} ({val_time:.1f}s)")
                  
            if val_metrics['ndcg5'] > best_ndcg5:
                best_ndcg5 = val_metrics['ndcg5']
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> Saved new best checkpoint to {best_model_path}")
                
        # Load best checkpoint for final evaluation
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"\nLoaded best model from {best_model_path} (Best Val NDCG@5: {best_ndcg5:.4f})")
    
    # 3.5 Test Benchmark Evaluation
    print("\n[5/7] Evaluating Best NRMS on Day 7 Test Partition (73,152 impressions)...")
    t_test = time.time()
    test_results = evaluate_model(model, test_records, news_title_lookup, device)
    print(f"Test Benchmark Results (Seed 42):")
    print(f"  • AUC:     {test_results['auc']:.4f}")
    print(f"  • MRR:     {test_results['mrr']:.4f}")
    print(f"  • NDCG@5:  {test_results['ndcg5']:.4f}")
    print(f"  • NDCG@10: {test_results['ndcg10']:.4f}")
    print(f"Evaluation completed in {time.time()-t_test:.2f}s.")
    
    # 3.6 Multi-Seed Robustness Evaluation (Seeds 42, 43, 44)
    print("\n[6/7] Multi-Seed Robustness Evaluation (Seeds 42, 43, 44)...")
    seed_results = {
        'seed42': {
            'auc': test_results['auc'],
            'mrr': test_results['mrr'],
            'ndcg5': test_results['ndcg5'],
            'ndcg10': test_results['ndcg10']
        }
    }
    
    if args.eval_only and os.path.exists(summary_path):
        print(f"Loading multi-seed results from {summary_path}...")
        with open(summary_path) as f:
            prev = json.load(f)
            seed_results = prev.get('multi_seed', {}).get('seeds', seed_results)
    else:
        for extra_seed in [43, 44]:
            print(f"\n--- Running Seed {extra_seed} ---")
            set_seed(extra_seed)
            m_seed = NRMS(pretrained_embeddings, news_title_lookup, d_model=256, n_heads=16, query_dim=200, dropout=0.2).to(device)
            opt_seed = torch.optim.Adam(m_seed.parameters(), lr=1e-4)
        
            # Train for 2 epochs with seed-specific negative sampling
            seed_samples = []
            for row in train_df.itertuples(index=False):
                raw_h = row.history.split() if isinstance(row.history, str) else []
                hist = [news2idx[x] for x in raw_h if x in news2idx][-max_hist_len:]
                if len(hist) < max_hist_len:
                    hist = [0] * (max_hist_len - len(hist)) + hist
                pos = [news2idx[item.split('-')[0]] for item in row.impressions.split() if item.split('-')[1] == '1' and item.split('-')[0] in news2idx]
                neg = [news2idx[item.split('-')[0]] for item in row.impressions.split() if item.split('-')[1] == '0' and item.split('-')[0] in news2idx]
                if not pos or not neg:
                    continue
                for p in pos:
                    s_neg = random.sample(neg, 4) if len(neg) >= 4 else random.choices(neg, k=4)
                    seed_samples.append((hist, [p] + s_neg))
                    
            seed_loader = DataLoader(TrainDataset(seed_samples), batch_size=128, shuffle=True, drop_last=True)
            for ep in range(1, 3):
                m_seed.train()
                tot_l = 0.0
                for h_b, c_b in seed_loader:
                    h_b, c_b = h_b.to(device), c_b.to(device)
                    opt_seed.zero_grad()
                    sc = m_seed(h_b, c_b)
                    tg = torch.zeros(sc.shape[0], dtype=torch.long, device=device)
                    l = F.cross_entropy(sc, tg)
                    l.backward()
                    opt_seed.step()
                    tot_l += l.item()
                print(f"Seed {extra_seed} Epoch {ep}/2 Loss: {tot_l/len(seed_loader):.4f}")
                
            m_res = evaluate_model(m_seed, test_records, news_title_lookup, device)
            seed_results[f'seed{extra_seed}'] = {
                'auc': m_res['auc'],
                'mrr': m_res['mrr'],
                'ndcg5': m_res['ndcg5'],
                'ndcg10': m_res['ndcg10']
            }
            print(f"Seed {extra_seed} Results: AUC={m_res['auc']:.4f}, MRR={m_res['mrr']:.4f}, NDCG@5={m_res['ndcg5']:.4f}, NDCG@10={m_res['ndcg10']:.4f}")
        
    aucs = [seed_results[s]['auc'] for s in seed_results]
    mrrs = [seed_results[s]['mrr'] for s in seed_results]
    n5s = [seed_results[s]['ndcg5'] for s in seed_results]
    n10s = [seed_results[s]['ndcg10'] for s in seed_results]
    
    print("\nMulti-Seed Summary (Mean ± Std over 3 seeds):")
    print(f"  • AUC:     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  • MRR:     {np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}")
    print(f"  • NDCG@5:  {np.mean(n5s):.4f} ± {np.std(n5s):.4f}")
    print(f"  • NDCG@10: {np.mean(n10s):.4f} ± {np.std(n10s):.4f}")
    
    # 3.7 Stratified Analyses (Cold-Start & Diurnal Temporal)
    print("\n[7/7] Computing Cold-Start and Diurnal Temporal Stratifications...")
    detailed = pd.DataFrame(test_results['detailed'])
    
    # 1. Cold Start Analysis
    bins = [-1, 0, 5, 15, 30, 99999]
    labels = ['Cold Start (0)', 'Sparse (1-5)', 'Moderate (6-15)', 'Active (16-30)', 'Power (>30)']
    detailed['history_bin'] = pd.cut(detailed['history_len'], bins=bins, labels=labels)
    
    cold_start_summary = detailed.groupby('history_bin', observed=False)[['auc', 'mrr', 'ndcg5', 'ndcg10']].agg(['mean', 'count'])
    print("\nUser Cold-Start Performance Breakdown:")
    for b in labels:
        sub = detailed[detailed['history_bin'] == b]
        print(f"  {b:<18} (N={len(sub):>5}): AUC={sub['auc'].mean():.4f}, MRR={sub['mrr'].mean():.4f}, NDCG@5={sub['ndcg5'].mean():.4f}, NDCG@10={sub['ndcg10'].mean():.4f}")
        
    # 2. Hourly Diurnal Analysis
    hourly_summary = detailed.groupby('hour')[['auc', 'mrr', 'ndcg5', 'ndcg10']].mean().reset_index()
    
    # 3. Attention Ablation Study (Without Self-Attention)
    print("\nRunning Ablation: NRMS without Self-Attention...")
    set_seed(42)
    ablation_model = NRMS(pretrained_embeddings, news_title_lookup, d_model=256, n_heads=16, query_dim=200, dropout=0.2, use_self_attn=False).to(device)
    abl_opt = torch.optim.Adam(ablation_model.parameters(), lr=1e-4)
    for ep in range(1, 3):
        ablation_model.train()
        for h_b, c_b in train_loader:
            h_b, c_b = h_b.to(device), c_b.to(device)
            abl_opt.zero_grad()
            sc = ablation_model(h_b, c_b)
            tg = torch.zeros(sc.shape[0], dtype=torch.long, device=device)
            l = F.cross_entropy(sc, tg)
            l.backward()
            abl_opt.step()
    abl_results = evaluate_model(ablation_model, test_records, news_title_lookup, device)
    print(f"Ablation (No Self-Attn) Results: AUC={abl_results['auc']:.4f}, MRR={abl_results['mrr']:.4f}, NDCG@5={abl_results['ndcg5']:.4f}, NDCG@10={abl_results['ndcg10']:.4f}")
    
    # ---------------------------------------------------------------------------
    # 4. Generate Publication-Quality Visualizations
    # ---------------------------------------------------------------------------
    print("\nGenerating publication figures...")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Fig 1: Training Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(1, epochs + 1), history['train_loss'], marker='o', color='#2b5c8f', lw=2.5, label='Train Loss')
    axes[0].set_title('Training Cross-Entropy Loss vs. Epoch', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
    axes[0].set_xticks(range(1, epochs + 1))
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].plot(range(1, epochs + 1), history['val_auc'], marker='s', color='#2ca02c', lw=2, label='Validation AUC')
    axes[1].plot(range(1, epochs + 1), history['val_ndcg5'], marker='^', color='#d62728', lw=2, label='Validation NDCG@5')
    axes[1].plot(range(1, epochs + 1), history['val_mrr'], marker='d', color='#9467bd', lw=2, label='Validation MRR')
    axes[1].plot(range(1, epochs + 1), history['val_ndcg10'], marker='v', color='#ff7f0e', lw=2, label='Validation NDCG@10')
    axes[1].set_title('Validation Ranking Performance on Day 6', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Metric Score', fontsize=12)
    axes[1].set_xticks(range(1, epochs + 1))
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    # Fig 2: Cold Start Analysis
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cs_df = pd.DataFrame({
        'History Length': labels,
        'AUC': [detailed[detailed['history_bin'] == b]['auc'].mean() for b in labels],
        'NDCG@5': [detailed[detailed['history_bin'] == b]['ndcg5'].mean() for b in labels],
        'MRR': [detailed[detailed['history_bin'] == b]['mrr'].mean() for b in labels],
        'NDCG@10': [detailed[detailed['history_bin'] == b]['ndcg10'].mean() for b in labels]
    })
    x = np.arange(len(labels))
    width = 0.2
    ax.bar(x - 1.5 * width, cs_df['AUC'], width, label='AUC', color='#1f77b4')
    ax.bar(x - 0.5 * width, cs_df['MRR'], width, label='MRR', color='#ff7f0e')
    ax.bar(x + 0.5 * width, cs_df['NDCG@5'], width, label='NDCG@5', color='#2ca02c')
    ax.bar(x + 1.5 * width, cs_df['NDCG@10'], width, label='NDCG@10', color='#d62728')
    ax.set_title('NRMS Ranking Performance across User History Lengths', fontsize=14, fontweight='bold')
    ax.set_xlabel('User Interaction History Length', fontsize=12)
    ax.set_ylabel('Metric Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.15, 0.75)
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'cold_start_analysis.png'), dpi=300)
    plt.close()
    
    # Fig 3: Diurnal Temporal Hourly Analysis
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    color1 = '#1f77b4'
    ax1.set_xlabel('Hour of Day (UTC)', fontsize=12)
    ax1.set_ylabel('NDCG@5', color=color1, fontsize=12)
    line1 = ax1.plot(hourly_summary['hour'], hourly_summary['ndcg5'], color=color1, marker='o', lw=2.5, label='NDCG@5')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(range(0, 24))
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel('AUC', color=color2, fontsize=12)
    line2 = ax2.plot(hourly_summary['hour'], hourly_summary['auc'], color=color2, marker='s', lw=2.5, linestyle='--', label='AUC')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines = line1 + line2
    labels_leg = [l.get_label() for l in lines]
    ax1.legend(lines, labels_leg, loc='lower right', frameon=True)
    plt.title('Diurnal Ranking Performance Stability across Day 7 Test Hours', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'temporal_hourly_analysis.png'), dpi=300)
    plt.close()
    
    # Fig 4: Ablation Study Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_names = ['AUC', 'MRR', 'NDCG@5', 'NDCG@10']
    full_vals = [test_results['auc'], test_results['mrr'], test_results['ndcg5'], test_results['ndcg10']]
    abl_vals = [abl_results['auc'], abl_results['mrr'], abl_results['ndcg5'], abl_results['ndcg10']]
    x = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(x - width/2, full_vals, width, label='Full NRMS (Multi-Head Self-Attn)', color='#2b5c8f')
    ax.bar(x + width/2, abl_vals, width, label='Ablation (Without Self-Attn)', color='#d95f02')
    ax.set_title('Ablation Study: Contribution of Multi-Head Self-Attention', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0.2, 0.75)
    for i in range(len(metrics_names)):
        ax.text(x[i] - width/2, full_vals[i] + 0.01, f"{full_vals[i]:.4f}", ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i] + width/2, abl_vals[i] + 0.01, f"{abl_vals[i]:.4f}", ha='center', fontsize=9, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'ablation_study.png'), dpi=300)
    plt.close()

    # Fig 5: Attention Visualization
    # Pick sample news titles and show word attention weights
    sample_news_ids = list(all_news['news_id'].iloc[10:14])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for idx, snid in enumerate(sample_news_ids):
        n_idx = news2idx[snid]
        toks = news_title_lookup[n_idx:n_idx+1]
        with torch.no_grad():
            _, alpha = model.news_encoder(toks, return_weights=True)
        alpha_np = alpha[0].detach().cpu().numpy()
        title_words = tokenize(all_news[all_news['news_id'] == snid]['title'].values[0])[:max_title_len]
        alpha_words = alpha_np[:len(title_words)]
        if alpha_words.sum() > 0:
            alpha_words = alpha_words / alpha_words.sum()
        
        y_pos = np.arange(len(title_words))
        axes[idx].barh(y_pos, alpha_words, color='#2b5c8f', alpha=0.85)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(title_words, fontsize=10)
        axes[idx].invert_yaxis()
        axes[idx].set_title(f"\"{all_news[all_news['news_id'] == snid]['title'].values[0][:40]}...\"", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Additive Attention Weight', fontsize=10)
        axes[idx].grid(True, linestyle='--', alpha=0.5)
    plt.suptitle('Word-Level Additive Attention Weights on Candidate News Titles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'attention_visualization.png'), dpi=300)
    plt.close()

    # Fig 6: Cold-Start Mitigation Analysis (Proposed Popularity-Gated NRMS)
    print("Evaluating Popularity-Gated NRMS (PG-NRMS) for Cold-Start Mitigation...")
    # Extract top popular news from train interactions
    click_counts = {}
    for row in train_df.itertuples(index=False):
        for item in row.impressions.split():
            nid, lbl = item.split('-')
            if lbl == '1' and nid in news2idx:
                click_counts[news2idx[nid]] = click_counts.get(news2idx[nid], 0) + 1
    top_popular_ids = [nid for nid, _ in sorted(click_counts.items(), key=lambda x: x[1], reverse=True)[:20]]
    
    # Pre-encode news
    num_news = len(news_title_lookup)
    news_vecs = torch.zeros(num_news, 256, device=device)
    with torch.no_grad():
        for i in range(0, num_news, 2048):
            chunk = news_title_lookup[i:min(i + 2048, num_news)].to(device)
            news_vecs[i:min(i + 2048, num_news)] = model.news_encoder(chunk)
    news_vecs[0] = 0.0

    top_pop_tensor = torch.tensor(top_popular_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        u_pop = model.user_encoder(news_vecs[top_pop_tensor].unsqueeze(0)).squeeze(0)

    # Encode test users
    num_eval = len(test_records)
    hist_matrix = torch.zeros(num_eval, 50, dtype=torch.long, device=device)
    for idx, rec in enumerate(test_records):
        hist_matrix[idx] = torch.tensor(rec['history'], dtype=torch.long, device=device)
    user_vecs = torch.zeros(num_eval, 256, device=device)
    with torch.no_grad():
        for i in range(0, num_eval, 1024):
            chunk_hist = hist_matrix[i:min(i + 1024, num_eval)]
            mask = (chunk_hist > 0)
            user_vecs[i:min(i + 1024, num_eval)] = model.user_encoder(news_vecs[chunk_hist], mask=mask)

    orig_detailed_list, pg_detailed_list = [], []
    for i, rec in enumerate(test_records):
        h_len = rec['history_len']
        u_pers = user_vecs[i]
        c_vecs = news_vecs[rec['candidates']]
        labels = rec['labels']

        # Original
        sc_orig = torch.mv(c_vecs, u_pers).cpu().numpy()
        a_o, m_o, n5_o, n10_o = eval_impression(labels, sc_orig)
        orig_detailed_list.append({'h_len': h_len, 'auc': a_o, 'mrr': m_o, 'ndcg5': n5_o, 'ndcg10': n10_o})

        # Popularity-Gated
        u_pg = u_pop if h_len == 0 else u_pers
        sc_pg = torch.mv(c_vecs, u_pg).cpu().numpy()
        a_p, m_p, n5_p, n10_p = eval_impression(labels, sc_pg)
        pg_detailed_list.append({'h_len': h_len, 'auc': a_p, 'mrr': m_p, 'ndcg5': n5_p, 'ndcg10': n10_p})

    df_orig = pd.DataFrame(orig_detailed_list)
    df_pg = pd.DataFrame(pg_detailed_list)
    orig_cs = df_orig[df_orig['h_len'] == 0]
    pg_cs = df_pg[df_pg['h_len'] == 0]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    metrics_cs = ['AUC', 'MRR', 'NDCG@5', 'NDCG@10']
    orig_cs_vals = [orig_cs['auc'].mean(), orig_cs['mrr'].mean(), orig_cs['ndcg5'].mean(), orig_cs['ndcg10'].mean()]
    pg_cs_vals = [pg_cs['auc'].mean(), pg_cs['mrr'].mean(), pg_cs['ndcg5'].mean(), pg_cs['ndcg10'].mean()]
    x = np.arange(len(metrics_cs))
    width = 0.35
    ax.bar(x - width/2, orig_cs_vals, width, label='Original NRMS (Baseline)', color='#e41a1c', alpha=0.85)
    ax.bar(x + width/2, pg_cs_vals, width, label='Popularity-Gated NRMS (Proposed)', color='#2ca02c', alpha=0.85)
    ax.set_title('Cold-Start User Performance: Original NRMS vs. Popularity-Gated NRMS', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_cs, fontsize=11)
    ax.set_ylim(0.15, 0.65)
    for i in range(len(metrics_cs)):
        gain = (pg_cs_vals[i] - orig_cs_vals[i]) / orig_cs_vals[i] * 100
        ax.text(x[i] - width/2, orig_cs_vals[i] + 0.01, f"{orig_cs_vals[i]:.4f}", ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i] + width/2, pg_cs_vals[i] + 0.01, f"{pg_cs_vals[i]:.4f}\\n(+{gain:.1f}%)", ha='center', fontsize=9, fontweight='bold', color='#1b7837')
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'cold_start_mitigation.png'), dpi=300)
    plt.close()
    
    # Save all summary stats to JSON for notebook embedding
    results_summary = {
        'single_seed': {
            'auc': test_results['auc'],
            'mrr': test_results['mrr'],
            'ndcg5': test_results['ndcg5'],
            'ndcg10': test_results['ndcg10']
        },
        'multi_seed': {
            'seeds': seed_results,
            'mean': {'auc': float(np.mean(aucs)), 'mrr': float(np.mean(mrrs)), 'ndcg5': float(np.mean(n5s)), 'ndcg10': float(np.mean(n10s))},
            'std': {'auc': float(np.std(aucs)), 'mrr': float(np.std(mrrs)), 'ndcg5': float(np.std(n5s)), 'ndcg10': float(np.std(n10s))}
        },
        'cold_start': {
            b: {
                'count': int((detailed['history_bin'] == b).sum()),
                'auc': float(detailed[detailed['history_bin'] == b]['auc'].mean()),
                'mrr': float(detailed[detailed['history_bin'] == b]['mrr'].mean()),
                'ndcg5': float(detailed[detailed['history_bin'] == b]['ndcg5'].mean()),
                'ndcg10': float(detailed[detailed['history_bin'] == b]['ndcg10'].mean())
            } for b in labels
        },
        'cold_start_mitigation': {
            'original_cold_start': {
                'auc': float(orig_cs['auc'].mean()),
                'mrr': float(orig_cs['mrr'].mean()),
                'ndcg5': float(orig_cs['ndcg5'].mean()),
                'ndcg10': float(orig_cs['ndcg10'].mean())
            },
            'popularity_gated_cold_start': {
                'auc': float(pg_cs['auc'].mean()),
                'mrr': float(pg_cs['mrr'].mean()),
                'ndcg5': float(pg_cs['ndcg5'].mean()),
                'ndcg10': float(pg_cs['ndcg10'].mean())
            },
            'relative_gain': {
                'auc': f"{(pg_cs['auc'].mean() - orig_cs['auc'].mean()) / orig_cs['auc'].mean() * 100:+.2f}%",
                'mrr': f"{(pg_cs['mrr'].mean() - orig_cs['mrr'].mean()) / orig_cs['mrr'].mean() * 100:+.2f}%",
                'ndcg5': f"{(pg_cs['ndcg5'].mean() - orig_cs['ndcg5'].mean()) / orig_cs['ndcg5'].mean() * 100:+.2f}%",
                'ndcg10': f"{(pg_cs['ndcg10'].mean() - orig_cs['ndcg10'].mean()) / orig_cs['ndcg10'].mean() * 100:+.2f}%"
            },
            'overall_original': {
                'auc': float(df_orig['auc'].mean()),
                'mrr': float(df_orig['mrr'].mean()),
                'ndcg5': float(df_orig['ndcg5'].mean()),
                'ndcg10': float(df_orig['ndcg10'].mean())
            },
            'overall_popularity_gated': {
                'auc': float(df_pg['auc'].mean()),
                'mrr': float(df_pg['mrr'].mean()),
                'ndcg5': float(df_pg['ndcg5'].mean()),
                'ndcg10': float(df_pg['ndcg10'].mean())
            }
        },
        'ablation': {
            'full': {'auc': test_results['auc'], 'mrr': test_results['mrr'], 'ndcg5': test_results['ndcg5'], 'ndcg10': test_results['ndcg10']},
            'no_self_attn': {'auc': abl_results['auc'], 'mrr': abl_results['mrr'], 'ndcg5': abl_results['ndcg5'], 'ndcg10': abl_results['ndcg10']}
        },
        'history': history
    }
    
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    print("\nAll experiments, evaluations, and figures successfully finished and saved!")
    print("=" * 70)

if __name__ == '__main__':
    main()
