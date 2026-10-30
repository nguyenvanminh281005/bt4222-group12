"""
NRMS / Gated Two-Branch NRMS Training, Validation, and Comprehensive Evaluation Script
BT4222 Group 12 - MIND-small News Recommendation

Model variants (--model):
  nrms  : Original NRMS (Wu et al., 2019). Cold-start users are additionally evaluated
          with a hard fallback (u = u_pop when |H| = 0) as a post-hoc baseline.
  gated : Two-branch NRMS trained end-to-end. A history branch (UserEncoder over the
          click history, or a learnable [EMPTY] vector when the history is empty) and a
          popularity branch (UserEncoder over the previous day's category-stratified
          popular articles) are merged by a learned per-user gate alpha in (0, 1):
              score(c) = alpha * <u_hist, e_c> + (1 - alpha) * <u_pop, e_c>
"""

import os
import re
import math
import time
import random
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Configure style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

HIST_BINS = [-1, 0, 5, 15, 30, 99999]
HIST_LABELS = ['Cold Start (0)', 'Sparse (1-5)', 'Moderate (6-15)', 'Active (16-30)', 'Power (>30)']


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

    pop_sets (optional): [n_days, K] news indices of each day's popular set (0 = pad).
    Only used by the hard cold-start fallback at evaluation time.
    """
    def __init__(self, pretrained_embed, news_title_lookup, d_model=256, n_heads=16, query_dim=200, dropout=0.2,
                 use_self_attn=True, pop_sets=None):
        super().__init__()
        self.register_buffer('news_title_lookup', news_title_lookup)
        self.register_buffer('pop_sets', pop_sets if pop_sets is not None else torch.zeros(0, 0, dtype=torch.long),
                             persistent=False)
        self.news_encoder = NewsEncoder(pretrained_embed, d_model, n_heads, query_dim, dropout, use_self_attn=use_self_attn)
        self.user_encoder = UserEncoder(d_model, n_heads, query_dim, dropout, use_self_attn=use_self_attn)

    def forward(self, hist_ids, cand_ids, day_idx=None):
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


class GatedNRMS(NRMS):
    """
    Two-branch NRMS with a learned per-user gate:
      - History branch:    u_hist = UserEncoder(history), or UserEncoder([EMPTY]) if the history is empty
      - Popularity branch: u_pop  = UserEncoder(previous day's category-stratified popular articles)
      - Gate:              alpha  = sigmoid(MLP([u_hist ; u_pop ; log(1+|H|) / log(1+P)]))
      - Score:             s(c)   = alpha * <u_hist, e_c> + (1 - alpha) * <u_pop, e_c>
    No hand-set threshold: the [EMPTY] vector lets the gate learn when to trust the popularity branch.
    """
    def __init__(self, pretrained_embed, news_title_lookup, d_model=256, n_heads=16, query_dim=200, dropout=0.2,
                 use_self_attn=True, pop_sets=None, gate_hidden=128, max_hist_len=50):
        super().__init__(pretrained_embed, news_title_lookup, d_model, n_heads, query_dim, dropout,
                         use_self_attn=use_self_attn, pop_sets=pop_sets)
        self.max_hist_len = max_hist_len
        self.empty_vec = nn.Parameter(torch.empty(d_model).normal_(0.0, 0.1))
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )

    def encode_popular(self, pop_news_vecs):
        # pop_news_vecs: [n_days, K, D] -> u_pop per day: [n_days, D]
        return self.user_encoder(pop_news_vecs, mask=(self.pop_sets > 0))

    def gate_users(self, hist_news_vecs, hist_mask, u_pop):
        # hist_news_vecs: [B, P, D], hist_mask: [B, P], u_pop: [B, D]
        B, P, D = hist_news_vecs.shape
        n_hist = hist_mask.sum(dim=1)
        empty = (n_hist == 0)
        # Empty history -> a single learnable [EMPTY] item in the last slot
        slot = torch.zeros_like(hist_mask)
        slot[:, -1] = True
        fill = slot & empty.unsqueeze(1)
        hist_news_vecs = torch.where(fill.unsqueeze(-1), self.empty_vec.expand(B, P, D), hist_news_vecs)
        hist_mask = hist_mask | fill
        u_hist = self.user_encoder(hist_news_vecs, mask=hist_mask)

        len_feat = (torch.log1p(n_hist.float()) / math.log1p(self.max_hist_len)).unsqueeze(1)
        alpha = torch.sigmoid(self.gate(torch.cat([u_hist, u_pop, len_feat], dim=1))).squeeze(1)
        return u_hist, alpha

    def forward(self, hist_ids, cand_ids, day_idx=None, return_alpha=False):
        B, P = hist_ids.shape
        C = cand_ids.shape[1]
        n_days, K = self.pop_sets.shape

        all_ids = torch.cat([hist_ids.view(-1), cand_ids.view(-1), self.pop_sets.view(-1)])
        unique_ids, inv = torch.unique(all_ids, return_inverse=True)
        all_vecs = self.news_encoder(self.news_title_lookup[unique_ids])[inv]

        hist_news_vecs = all_vecs[:B * P].view(B, P, -1)
        cand_news_vecs = all_vecs[B * P:B * P + B * C].view(B, C, -1)
        pop_news_vecs = all_vecs[B * P + B * C:].view(n_days, K, -1)

        u_pop = self.encode_popular(pop_news_vecs)[day_idx] # [B, D]
        u_hist, alpha = self.gate_users(hist_news_vecs, hist_ids > 0, u_pop)

        s_hist = torch.bmm(cand_news_vecs, u_hist.unsqueeze(2)).squeeze(2)
        s_pop = torch.bmm(cand_news_vecs, u_pop.unsqueeze(2)).squeeze(2)
        scores = alpha.unsqueeze(1) * s_hist + (1 - alpha.unsqueeze(1)) * s_pop
        if return_alpha:
            return scores, alpha
        return scores


class TrainDataset(Dataset):
    """
    History dropout (training only): with prob p_empty the history is wiped (simulated cold-start user),
    with prob p_trunc it is cut to its last 1..max_trunc clicks (simulated sparse user).
    """
    def __init__(self, samples, p_empty=0.0, p_trunc=0.0, max_trunc=3):
        self.samples = samples
        self.p_empty = p_empty
        self.p_trunc = p_trunc
        self.max_trunc = max_trunc
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        h, c, d = self.samples[idx]
        r = random.random() if (self.p_empty > 0 or self.p_trunc > 0) else 1.0
        if r < self.p_empty:
            h = [0] * len(h)
        elif r < self.p_empty + self.p_trunc:
            valid = [x for x in h if x > 0]
            k = random.randint(1, self.max_trunc)
            if len(valid) > k:
                h = [0] * (len(h) - k) + valid[-k:]
        return torch.tensor(h, dtype=torch.long), torch.tensor(c, dtype=torch.long), torch.tensor(d, dtype=torch.long)


def ranking_loss(scores, loss_type='ce', gamma=2.0):
    """
    Listwise loss over [1 positive, K negatives] (positive at index 0).
      ce    : -log p+                    (sampled softmax, NRMS default)
      focal : -(1 - p+)^gamma * log p+   (down-weights easy impressions)
    """
    log_p = F.log_softmax(scores, dim=-1)[:, 0]
    if loss_type == 'focal':
        return -(((1.0 - log_p.exp()) ** gamma) * log_p).mean()
    return -log_p.mean()


# ---------------------------------------------------------------------------
# 2. Ranking Evaluation Metrics
# ---------------------------------------------------------------------------
_TIE_RNG = np.random.default_rng(0)


def eval_impression(labels, scores):
    """
    Evaluates AUC, MRR, NDCG@5, and NDCG@10 for a single impression slate.
    Ties are not resolved by the logged slate order (a position artifact): AUC uses average ranks,
    so fully tied scores give exactly 0.5, and MRR/NDCG break ties uniformly at random.
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    pos_mask = (labels == 1)
    n_pos = pos_mask.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        ranks = rankdata(scores)
        auc = (ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    perm = _TIE_RNG.permutation(len(scores))
    desc_order = perm[np.argsort(-scores[perm], kind='stable')]
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
def evaluate_model(model, eval_records, news_title_lookup, device, batch_size=2048, cold_fallback=False):
    """
    Vectorized high-throughput evaluation:
    1. Pre-encodes all news articles once with NewsEncoder.
    2. Encodes all user representations in batches. For GatedNRMS the effective user vector is
       alpha * u_hist + (1 - alpha) * u_pop, which is exactly equivalent to mixing the two branch scores.
       For NRMS with cold_fallback=True, users with empty history get u_pop (hard fallback baseline).
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

    gated = isinstance(model, GatedNRMS)
    use_pop = gated or (cold_fallback and model.pop_sets.numel() > 0)
    if use_pop:
        pop_sets = model.pop_sets
        u_pop_all = model.user_encoder(news_vecs[pop_sets], mask=(pop_sets > 0)) # [n_days, D]

    num_eval = len(eval_records)
    hist_matrix = torch.zeros(num_eval, 50, dtype=torch.long, device=device)
    for idx, rec in enumerate(eval_records):
        hist_matrix[idx] = torch.tensor(rec['history'], dtype=torch.long, device=device)
    day_vec = torch.tensor([rec.get('day_idx', 0) for rec in eval_records], dtype=torch.long, device=device)

    user_vecs = torch.zeros(num_eval, 256, device=device)
    alphas = torch.ones(num_eval, device=device)
    user_batch_size = 1024
    for i in range(0, num_eval, user_batch_size):
        j = min(i + user_batch_size, num_eval)
        chunk_hist = hist_matrix[i:j]
        chunk_news_vecs = news_vecs[chunk_hist] # [B, 50, 256]
        mask = (chunk_hist > 0)
        if gated:
            u_pop = u_pop_all[day_vec[i:j]]
            u_hist, alpha = model.gate_users(chunk_news_vecs, mask, u_pop)
            u_vecs = alpha.unsqueeze(1) * u_hist + (1 - alpha.unsqueeze(1)) * u_pop
            alphas[i:j] = alpha
        else:
            u_vecs = model.user_encoder(chunk_news_vecs, mask=mask)
            if use_pop:
                empty = ~mask.any(dim=1)
                u_vecs = torch.where(empty.unsqueeze(1), u_pop_all[day_vec[i:j]], u_vecs)
                alphas[i:j] = (~empty).float()
        user_vecs[i:j] = u_vecs
    alphas = alphas.cpu().numpy()

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
            'alpha': float(alphas[i]),
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


def metrics_of(res):
    return {k: float(res[k]) for k in ['auc', 'mrr', 'ndcg5', 'ndcg10']}


def stratify_by_history(detailed_df):
    detailed_df = detailed_df.copy()
    detailed_df['history_bin'] = pd.cut(detailed_df['history_len'], bins=HIST_BINS, labels=HIST_LABELS)
    out = {}
    for b in HIST_LABELS:
        sub = detailed_df[detailed_df['history_bin'] == b]
        out[b] = {
            'count': int(len(sub)),
            'auc': float(sub['auc'].mean()),
            'mrr': float(sub['mrr'].mean()),
            'ndcg5': float(sub['ndcg5'].mean()),
            'ndcg10': float(sub['ndcg10'].mean()),
            'alpha_mean': float(sub['alpha'].mean()),
            'alpha_std': float(sub['alpha'].std()) if len(sub) > 1 else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# 3. Popularity Branch: previous-day, category-stratified popular sets
# ---------------------------------------------------------------------------
def daily_click_counts(df, news2idx):
    counts = defaultdict(Counter)
    for row in df.itertuples(index=False):
        day = row.time.normalize()
        for item in row.impressions.split():
            if item.endswith('-1'):
                idx = news2idx.get(item[:-2])
                if idx:
                    counts[day][idx] += 1
    return counts


def build_popular_set(click_counts, news_cat, per_cat=2, extra=10, min_cat_clicks=20):
    """
    Every category with >= min_cat_clicks clicks contributes its top `per_cat` articles;
    `extra` additional slots are allocated proportionally to category click volume (largest remainder).
    """
    cat_clicks = Counter()
    per_cat_items = defaultdict(list)
    for idx, c in click_counts.items():
        cat = news_cat[idx]
        cat_clicks[cat] += c
        per_cat_items[cat].append((c, idx))
    eligible = sorted([c for c, n in cat_clicks.items() if n >= min_cat_clicks], key=lambda c: -cat_clicks[c])
    if not eligible:
        return [], {}
    total = sum(cat_clicks[c] for c in eligible)
    quotas = {c: extra * cat_clicks[c] / total for c in eligible}
    alloc = {c: per_cat + int(quotas[c]) for c in eligible}
    leftover = extra - sum(int(q) for q in quotas.values())
    for c in sorted(eligible, key=lambda c: quotas[c] - int(quotas[c]), reverse=True)[:leftover]:
        alloc[c] += 1
    ids = []
    for c in eligible:
        top = sorted(per_cat_items[c], reverse=True)[:alloc[c]]
        alloc[c] = len(top)
        ids += [idx for _, idx in top]
    return ids, alloc


# ---------------------------------------------------------------------------
# 4. Data Preparation & Main Training / Evaluation Workflow
# ---------------------------------------------------------------------------
def prepare_data(data_dir, device, pop_per_cat=2, pop_extra=10, pop_min_cat_clicks=20, debug=False):
    """
    Loads MIND-small, builds the title lookup / GloVe matrix, the strict 5-1-1 temporal split,
    the daily popular sets, and the validation/test records. Shared by training and evaluation scripts.
    """
    # 4.1 Load and Merge News Catalogue
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
    news_cat = ['<PAD>']
    for nid, cat in zip(all_news['news_id'], all_news['category']):
        news2idx[nid] = len(news2idx)
        news_cat.append(cat)

    max_title_len = 30
    lookup = np.zeros((len(news2idx), max_title_len), dtype=np.int64)
    for row in all_news.itertuples(index=False):
        idx = news2idx[row.news_id]
        tokens = [vocab.get(w, 1) for w in tokenize(row.title)][:max_title_len]
        lookup[idx, :len(tokens)] = tokens

    news_title_lookup = torch.tensor(lookup, dtype=torch.long, device=device)
    print(f"News lookup tensor: {news_title_lookup.shape} (Time: {time.time()-t0:.2f}s)")

    # 4.2 Load Pretrained GloVe Embeddings
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

    # 4.3 Load Behaviors and Strict Temporal Split
    print("\n[3/7] Parsing Behaviors and Creating Strict Temporal Partitions...")
    t0 = time.time()
    train_b = pd.read_csv(os.path.join(data_dir, 'train/behaviors.tsv'), sep='\t', header=None,
                          names=['imp_id', 'user_id', 'time', 'history', 'impressions'])
    dev_b = pd.read_csv(os.path.join(data_dir, 'dev/behaviors.tsv'), sep='\t', header=None,
                        names=['imp_id', 'user_id', 'time', 'history', 'impressions'])
    train_b['time'] = pd.to_datetime(train_b['time'])
    dev_b['time'] = pd.to_datetime(dev_b['time'])

    # Train: Days 1-5 (2019-11-09 to 2019-11-13) | Validation: Day 6 (2019-11-14) | Test: Day 7 (2019-11-15)
    train_df = train_b[train_b['time'] < '2019-11-14'].copy()
    val_df = train_b[train_b['time'] >= '2019-11-14'].copy()
    test_df = dev_b.copy()
    if debug:
        train_df = train_df.sample(n=min(4000, len(train_df)), random_state=0)
        val_df = val_df.sample(n=min(1500, len(val_df)), random_state=0)
        test_df = test_df.sample(n=min(3000, len(test_df)), random_state=0)

    print(f"Temporal Holdout Splits:")
    print(f"  • Train (Days 1–5):       {len(train_df):>7} impressions ({train_df['time'].min()} to {train_df['time'].max()})")
    print(f"  • Validation (Day 6):     {len(val_df):>7} impressions ({val_df['time'].min()} to {val_df['time'].max()})")
    print(f"  • Test (Day 7):           {len(test_df):>7} impressions ({test_df['time'].min()} to {test_df['time'].max()})")

    # 4.4 Popularity branch: for an impression on day d, use clicks logged on day d-1 (no same-day leakage).
    # Day 1 has no previous logged day, so it falls back to its own clicks. Clicks come from train_b only (Days 1-6).
    all_days = sorted(set(train_b['time'].dt.normalize()) | set(dev_b['time'].dt.normalize()))
    day2idx = {d: i for i, d in enumerate(all_days)}
    clicks_by_day = daily_click_counts(train_b, news2idx)
    pop_lists = []
    print("\nDaily category-stratified popular sets (source day -> size, categories):")
    for d in all_days:
        src = d - pd.Timedelta(days=1)
        if src not in clicks_by_day:
            src = d
        ids, alloc = build_popular_set(clicks_by_day[src], news_cat, pop_per_cat, pop_extra, pop_min_cat_clicks)
        pop_lists.append(ids)
        print(f"  • {d.date()} <- {src.date()}: {len(ids)} articles over {len(alloc)} categories {dict(alloc)}")
    K = max(len(p) for p in pop_lists)
    pop_sets = torch.zeros(len(all_days), K, dtype=torch.long)
    for i, ids in enumerate(pop_lists):
        pop_sets[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    pop_sets = pop_sets.to(device)

    max_hist_len = 50

    def parse_history(raw):
        raw_h = raw.split() if isinstance(raw, str) else []
        hist = [news2idx[x] for x in raw_h if x in news2idx][-max_hist_len:]
        return [0] * (max_hist_len - len(hist)) + hist, len(raw_h)


    def build_eval_records(df):
        records = []
        for row in df.itertuples(index=False):
            hist, hist_len = parse_history(row.history)
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
                    'day_idx': day2idx[row.time.normalize()],
                    'history': hist,
                    'history_len': hist_len,
                    'candidates': cands,
                    'labels': labels
                })
        return records

    val_records = build_eval_records(val_df)
    test_records = build_eval_records(test_df)
    print(f"\nConstructed {len(val_records)} validation records and {len(test_records)} test records in {time.time()-t0:.2f}s.")

    return argparse.Namespace(
        all_news=all_news, tokenize=tokenize, max_title_len=max_title_len, news2idx=news2idx, news_cat=news_cat,
        news_title_lookup=news_title_lookup, pretrained_embeddings=pretrained_embeddings,
        train_df=train_df, val_df=val_df, test_df=test_df, all_days=all_days, day2idx=day2idx,
        pop_lists=pop_lists, pop_sets=pop_sets, max_hist_len=max_hist_len, parse_history=parse_history,
        val_records=val_records, test_records=test_records,
    )

# ---------------------------------------------------------------------------
# Main Training & Evaluation Workflow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NRMS / Gated NRMS Training & Evaluation on MIND-small")
    if os.path.basename(os.getcwd()) == "models":
        data_dir = "../data"
        models_dir = "."
    else:
        data_dir = "data"
        models_dir = "models"

    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    figures_dir = os.path.join(models_dir, "figures")

    parser.add_argument('--model', choices=['nrms', 'gated'], default='nrms', help="Model variant")
    parser.add_argument('--tag', type=str, default=None, help="Run name used for checkpoint/summary/figure names (default: --model)")
    parser.add_argument('--eval_only', action='store_true', help="Skip training and evaluate from checkpoint")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train (all seeds and ablation)")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_neg', type=int, default=4, help="Negatives sampled per click (within the same impression)")
    parser.add_argument('--loss', choices=['ce', 'focal'], default='ce')
    parser.add_argument('--gamma', type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument('--hist_dropout', type=float, default=0.0, help="Prob. of wiping a training history (simulated cold-start)")
    parser.add_argument('--hist_trunc', type=float, default=0.0, help="Prob. of truncating a training history to 1..3 clicks")
    parser.add_argument('--pop_per_cat', type=int, default=2, help="Popular articles per category")
    parser.add_argument('--pop_extra', type=int, default=10, help="Extra popular slots allocated by category click share")
    parser.add_argument('--pop_min_cat_clicks', type=int, default=20, help="Minimum daily clicks for a category to be included")
    parser.add_argument('--skip_seeds', action='store_true', help="Skip multi-seed robustness runs (seeds 43, 44)")
    parser.add_argument('--skip_ablation', action='store_true', help="Skip the no-self-attention ablation run")
    parser.add_argument('--debug', action='store_true', help="Tiny subsample smoke test")
    args = parser.parse_args()
    tag = args.tag or args.model
    if args.debug:
        tag = f'debug_{tag}'
    file_prefix = '' if tag == 'nrms' else f'{tag}_'
    summary_path = os.path.join(figures_dir, 'results_summary.json' if tag == 'nrms' else f'results_summary_{tag}.json')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print(f"BT4222 Group 12: {args.model.upper()} Training & Comprehensive Evaluation (run tag: {tag})")
    print(f"Config: {vars(args)}")
    print("=" * 70)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data = prepare_data(data_dir, device, args.pop_per_cat, args.pop_extra, args.pop_min_cat_clicks, args.debug)
    all_news, tokenize, max_title_len = data.all_news, data.tokenize, data.max_title_len
    news2idx, news_title_lookup, pretrained_embeddings = data.news2idx, data.news_title_lookup, data.pretrained_embeddings
    train_df, all_days, day2idx = data.train_df, data.all_days, data.day2idx
    pop_lists, pop_sets, parse_history = data.pop_lists, data.pop_sets, data.parse_history
    val_records, test_records = data.val_records, data.test_records

    def build_train_samples(df):
        # 1 positive + num_neg negatives sampled within each impression
        samples = []
        n_cold = 0
        for row in df.itertuples(index=False):
            hist, hist_len = parse_history(row.history)
            day = day2idx[row.time.normalize()]
            pos, neg = [], []
            for item in row.impressions.split():
                nid, lbl = item.split('-')
                if nid in news2idx:
                    (pos if lbl == '1' else neg).append(news2idx[nid])
            if not pos or not neg:
                continue
            for p in pos:
                if len(neg) >= args.num_neg:
                    sampled_neg = random.sample(neg, args.num_neg)
                else:
                    sampled_neg = random.choices(neg, k=args.num_neg)
                samples.append((hist, [p] + sampled_neg, day))
                n_cold += (hist_len == 0)
        return samples, n_cold

    model_cls = GatedNRMS if args.model == 'gated' else NRMS

    def make_model(use_self_attn=True):
        return model_cls(pretrained_embeddings, news_title_lookup, d_model=256, n_heads=16, query_dim=200, dropout=0.2,
                         use_self_attn=use_self_attn, pop_sets=pop_sets).to(device)

    def train_model(seed, ckpt_path, use_self_attn=True):
        """Trains for args.epochs epochs and keeps the checkpoint with the best validation NDCG@5."""
        set_seed(seed)
        samples, n_cold = build_train_samples(train_df)
        print(f"Constructed {len(samples)} training instances (1 pos + {args.num_neg} neg), "
              f"{n_cold} ({n_cold / max(len(samples), 1) * 100:.2f}%) with empty history.")
        loader = DataLoader(TrainDataset(samples, args.hist_dropout, args.hist_trunc),
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
        model = make_model(use_self_attn)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        history = {'train_loss': [], 'train_alpha': [], 'val_auc': [], 'val_mrr': [], 'val_ndcg5': [], 'val_ndcg10': []}
        best_ndcg5 = -1.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, total_alpha = 0.0, 0.0
            t_epoch = time.time()
            for hist_b, cand_b, day_b in loader:
                hist_b, cand_b, day_b = hist_b.to(device), cand_b.to(device), day_b.to(device)
                optimizer.zero_grad()
                if args.model == 'gated':
                    scores, alpha = model(hist_b, cand_b, day_b, return_alpha=True)
                    total_alpha += alpha.mean().item()
                else:
                    scores = model(hist_b, cand_b, day_b)
                loss = ranking_loss(scores, args.loss, args.gamma)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            avg_alpha = total_alpha / len(loader) if args.model == 'gated' else 1.0
            epoch_time = time.time() - t_epoch

            t_val = time.time()
            val_metrics = evaluate_model(model, val_records, news_title_lookup, device)
            history['train_loss'].append(avg_loss)
            history['train_alpha'].append(avg_alpha)
            for k in ['auc', 'mrr', 'ndcg5', 'ndcg10']:
                history[f'val_{k}'].append(val_metrics[k])
            print(f"[seed {seed}] Epoch {epoch}/{args.epochs} | Train Loss: {avg_loss:.4f} | mean alpha: {avg_alpha:.3f} ({epoch_time:.1f}s) | "
                  f"Val AUC: {val_metrics['auc']:.4f} | MRR: {val_metrics['mrr']:.4f} | "
                  f"NDCG@5: {val_metrics['ndcg5']:.4f} | NDCG@10: {val_metrics['ndcg10']:.4f} ({time.time()-t_val:.1f}s)")
            if val_metrics['ndcg5'] > best_ndcg5:
                best_ndcg5 = val_metrics['ndcg5']
                torch.save(model.state_dict(), ckpt_path)
                print(f"  --> Saved new best checkpoint to {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best model from {ckpt_path} (Best Val NDCG@5: {best_ndcg5:.4f})")
        return model, history

    prev = {}
    if args.eval_only and os.path.exists(summary_path):
        with open(summary_path) as f:
            prev = json.load(f)

    # 4.5 Model Training (Seed 42)
    print(f"\n[4/7] Training {args.model.upper()} (Seed 42)...")
    best_model_path = os.path.join(checkpoints_dir, f'{tag}_best_seed42.pt')
    if args.eval_only and os.path.exists(best_model_path):
        print(f"--eval_only: loading checkpoint {best_model_path}")
        model = make_model()
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        history = prev.get('history', {})
    else:
        model, history = train_model(42, best_model_path)

    # 4.6 Test Benchmark Evaluation
    print(f"\n[5/7] Evaluating best model on Day 7 Test Partition ({len(test_records)} impressions)...")
    t_test = time.time()
    test_results = evaluate_model(model, test_records, news_title_lookup, device)
    print(f"Test Benchmark Results (Seed 42):")
    print(f"  • AUC:     {test_results['auc']:.4f}")
    print(f"  • MRR:     {test_results['mrr']:.4f}")
    print(f"  • NDCG@5:  {test_results['ndcg5']:.4f}")
    print(f"  • NDCG@10: {test_results['ndcg10']:.4f}")
    print(f"Evaluation completed in {time.time()-t_test:.2f}s.")

    # 4.7 Multi-Seed Robustness Evaluation (Seeds 42, 43, 44) - same epochs and checkpoint selection as seed 42
    print("\n[6/7] Multi-Seed Robustness Evaluation (Seeds 42, 43, 44)...")
    seed_results = {'seed42': metrics_of(test_results)}
    if args.skip_seeds:
        print("Skipped (--skip_seeds).")
    elif args.eval_only:
        seed_results = prev.get('multi_seed', {}).get('seeds', seed_results)
    else:
        for extra_seed in [43, 44]:
            print(f"\n--- Running Seed {extra_seed} ---")
            m_seed, _ = train_model(extra_seed, os.path.join(checkpoints_dir, f'{tag}_best_seed{extra_seed}.pt'))
            seed_results[f'seed{extra_seed}'] = metrics_of(evaluate_model(m_seed, test_records, news_title_lookup, device))
            r = seed_results[f'seed{extra_seed}']
            print(f"Seed {extra_seed} Results: AUC={r['auc']:.4f}, MRR={r['mrr']:.4f}, NDCG@5={r['ndcg5']:.4f}, NDCG@10={r['ndcg10']:.4f}")
            del m_seed
            torch.cuda.empty_cache()

    aucs = [seed_results[s]['auc'] for s in seed_results]
    mrrs = [seed_results[s]['mrr'] for s in seed_results]
    n5s = [seed_results[s]['ndcg5'] for s in seed_results]
    n10s = [seed_results[s]['ndcg10'] for s in seed_results]

    print(f"\nMulti-Seed Summary (Mean ± Std over {len(seed_results)} seed(s)):")
    print(f"  • AUC:     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  • MRR:     {np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}")
    print(f"  • NDCG@5:  {np.mean(n5s):.4f} ± {np.std(n5s):.4f}")
    print(f"  • NDCG@10: {np.mean(n10s):.4f} ± {np.std(n10s):.4f}")

    # 4.8 Stratified Analyses (Cold-Start & Diurnal Temporal)
    print("\n[7/7] Computing Cold-Start and Diurnal Temporal Stratifications...")
    detailed = pd.DataFrame(test_results['detailed'])
    cold_start = stratify_by_history(detailed)
    print("\nUser Cold-Start Performance Breakdown:")
    for b in HIST_LABELS:
        s = cold_start[b]
        alpha_txt = f", alpha={s['alpha_mean']:.3f}±{s['alpha_std']:.3f}" if args.model == 'gated' else ""
        print(f"  {b:<18} (N={s['count']:>5}): AUC={s['auc']:.4f}, MRR={s['mrr']:.4f}, NDCG@5={s['ndcg5']:.4f}, NDCG@10={s['ndcg10']:.4f}{alpha_txt}")

    hourly_summary = detailed.groupby('hour')[['auc', 'mrr', 'ndcg5', 'ndcg10']].mean().reset_index()
    print(f"Hourly NDCG@5 range: [{hourly_summary['ndcg5'].min():.4f}, {hourly_summary['ndcg5'].max():.4f}] (std {hourly_summary['ndcg5'].std():.4f}), "
          f"AUC range: [{hourly_summary['auc'].min():.4f}, {hourly_summary['auc'].max():.4f}] (std {hourly_summary['auc'].std():.4f})")

    # Hard fallback baseline (post-hoc): only meaningful for the plain NRMS model
    fallback = None
    if args.model == 'nrms':
        print("\nEvaluating hard cold-start fallback (u = u_pop if |H| = 0, same popular sets as the gated model)...")
        fb_results = evaluate_model(model, test_records, news_title_lookup, device, cold_fallback=True)
        fb_cold = stratify_by_history(pd.DataFrame(fb_results['detailed']))
        fallback = {'overall': metrics_of(fb_results), 'cold_start': fb_cold}
        cs_o, cs_f = cold_start[HIST_LABELS[0]], fb_cold[HIST_LABELS[0]]
        print(f"  Cold-start  NRMS:     AUC={cs_o['auc']:.4f}, MRR={cs_o['mrr']:.4f}, NDCG@5={cs_o['ndcg5']:.4f}, NDCG@10={cs_o['ndcg10']:.4f}")
        print(f"  Cold-start  Fallback: AUC={cs_f['auc']:.4f}, MRR={cs_f['mrr']:.4f}, NDCG@5={cs_f['ndcg5']:.4f}, NDCG@10={cs_f['ndcg10']:.4f}")
        print(f"  Overall     Fallback: AUC={fb_results['auc']:.4f}, MRR={fb_results['mrr']:.4f}, NDCG@5={fb_results['ndcg5']:.4f}, NDCG@10={fb_results['ndcg10']:.4f}")

    # Attention Ablation Study (Without Self-Attention)
    abl_results = None
    if args.skip_ablation:
        print("\nAblation skipped (--skip_ablation).")
    elif args.eval_only:
        abl_results = prev.get('ablation', {}).get('no_self_attn')
    else:
        print("\nRunning Ablation: same model without Self-Attention...")
        ablation_model, _ = train_model(42, os.path.join(checkpoints_dir, f'{tag}_noattn_best_seed42.pt'), use_self_attn=False)
        abl_results = metrics_of(evaluate_model(ablation_model, test_records, news_title_lookup, device))
        print(f"Ablation (No Self-Attn) Results: AUC={abl_results['auc']:.4f}, MRR={abl_results['mrr']:.4f}, NDCG@5={abl_results['ndcg5']:.4f}, NDCG@10={abl_results['ndcg10']:.4f}")
        del ablation_model
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # 5. Generate Publication-Quality Visualizations
    # ---------------------------------------------------------------------------
    print("\nGenerating publication figures...")
    fig_path = lambda name: os.path.join(figures_dir, f'{file_prefix}{name}')
    loss_name = 'Focal Loss' if args.loss == 'focal' else 'Cross-Entropy Loss'

    # Fig 1: Training Curves
    if history.get('train_loss'):
        ep_range = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(ep_range, history['train_loss'], marker='o', color='#2b5c8f', lw=2.5, label='Train Loss')
        axes[0].set_title(f'Training {loss_name} vs. Epoch', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel(loss_name, fontsize=12)
        axes[0].set_xticks(list(ep_range))
        axes[0].legend(frameon=True)
        axes[0].grid(True, linestyle='--', alpha=0.6)

        axes[1].plot(ep_range, history['val_auc'], marker='s', color='#2ca02c', lw=2, label='Validation AUC')
        axes[1].plot(ep_range, history['val_ndcg5'], marker='^', color='#d62728', lw=2, label='Validation NDCG@5')
        axes[1].plot(ep_range, history['val_mrr'], marker='d', color='#9467bd', lw=2, label='Validation MRR')
        axes[1].plot(ep_range, history['val_ndcg10'], marker='v', color='#ff7f0e', lw=2, label='Validation NDCG@10')
        axes[1].set_title('Validation Ranking Performance on Day 6', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Metric Score', fontsize=12)
        axes[1].set_xticks(list(ep_range))
        axes[1].legend(frameon=True)
        axes[1].grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(fig_path('training_curves.png'), dpi=300)
        plt.close()

    # Fig 2: Cold Start Analysis
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(HIST_LABELS))
    width = 0.2
    for off, key, name, color in [(-1.5, 'auc', 'AUC', '#1f77b4'), (-0.5, 'mrr', 'MRR', '#ff7f0e'),
                                  (0.5, 'ndcg5', 'NDCG@5', '#2ca02c'), (1.5, 'ndcg10', 'NDCG@10', '#d62728')]:
        ax.bar(x + off * width, [cold_start[b][key] for b in HIST_LABELS], width, label=name, color=color)
    ax.set_title(f'{args.model.upper()} Ranking Performance across User History Lengths', fontsize=14, fontweight='bold')
    ax.set_xlabel('User Interaction History Length', fontsize=12)
    ax.set_ylabel('Metric Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(HIST_LABELS, fontsize=11)
    ax.set_ylim(0.15, 0.75)
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path('cold_start_analysis.png'), dpi=300)
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
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower right', frameon=True)
    plt.title('Diurnal Ranking Performance Stability across Day 7 Test Hours', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_path('temporal_hourly_analysis.png'), dpi=300)
    plt.close()

    # Fig 4: Ablation Study Comparison
    if abl_results is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics_names = ['AUC', 'MRR', 'NDCG@5', 'NDCG@10']
        full_vals = [test_results['auc'], test_results['mrr'], test_results['ndcg5'], test_results['ndcg10']]
        abl_vals = [abl_results['auc'], abl_results['mrr'], abl_results['ndcg5'], abl_results['ndcg10']]
        x = np.arange(len(metrics_names))
        width = 0.35
        ax.bar(x - width/2, full_vals, width, label='Full (Multi-Head Self-Attn)', color='#2b5c8f')
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
        plt.savefig(fig_path('ablation_study.png'), dpi=300)
        plt.close()

    # Fig 5: Attention Visualization
    sample_news_ids = list(all_news['news_id'].iloc[10:14])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    model.eval()
    for idx, snid in enumerate(sample_news_ids):
        n_idx = news2idx[snid]
        toks = news_title_lookup[n_idx:n_idx+1]
        with torch.no_grad():
            _, alpha = model.news_encoder(toks, return_weights=True)
        alpha_np = alpha[0].detach().cpu().numpy()
        title = all_news[all_news['news_id'] == snid]['title'].values[0]
        title_words = tokenize(title)[:max_title_len]
        alpha_words = alpha_np[:len(title_words)]
        if alpha_words.sum() > 0:
            alpha_words = alpha_words / alpha_words.sum()
        y_pos = np.arange(len(title_words))
        axes[idx].barh(y_pos, alpha_words, color='#2b5c8f', alpha=0.85)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(title_words, fontsize=10)
        axes[idx].invert_yaxis()
        axes[idx].set_title(f"\"{title[:40]}...\"", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Additive Attention Weight', fontsize=10)
        axes[idx].grid(True, linestyle='--', alpha=0.5)
    plt.suptitle('Word-Level Additive Attention Weights on Candidate News Titles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_path('attention_visualization.png'), dpi=300)
    plt.close()

    # Fig 6 (nrms): Cold-start hard fallback vs. original | (gated): learned gate weight per history bin
    if fallback is not None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        metrics_cs = ['AUC', 'MRR', 'NDCG@5', 'NDCG@10']
        keys = ['auc', 'mrr', 'ndcg5', 'ndcg10']
        orig_vals = [cold_start[HIST_LABELS[0]][k] for k in keys]
        fb_vals = [fallback['cold_start'][HIST_LABELS[0]][k] for k in keys]
        x = np.arange(len(metrics_cs))
        width = 0.35
        ax.bar(x - width/2, orig_vals, width, label='Original NRMS', color='#e41a1c', alpha=0.85)
        ax.bar(x + width/2, fb_vals, width, label='NRMS + Hard Popularity Fallback', color='#2ca02c', alpha=0.85)
        ax.set_title('Cold-Start Users: Original NRMS vs. Hard Popularity Fallback', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_cs, fontsize=11)
        ax.set_ylim(0.15, 0.7)
        for i in range(len(metrics_cs)):
            ax.text(x[i] - width/2, orig_vals[i] + 0.01, f"{orig_vals[i]:.4f}", ha='center', fontsize=9, fontweight='bold')
            ax.text(x[i] + width/2, fb_vals[i] + 0.01, f"{fb_vals[i]:.4f}", ha='center', fontsize=9, fontweight='bold')
        ax.legend(frameon=True, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(fig_path('cold_start_mitigation.png'), dpi=300)
        plt.close()
    if args.model == 'gated':
        fig, ax = plt.subplots(figsize=(9, 5))
        means = [cold_start[b]['alpha_mean'] for b in HIST_LABELS]
        stds = [cold_start[b]['alpha_std'] for b in HIST_LABELS]
        x = np.arange(len(HIST_LABELS))
        ax.bar(x, means, yerr=stds, capsize=5, color='#2b5c8f', alpha=0.85, label='History-branch weight α')
        ax.bar(x, [1 - m for m in means], bottom=means, color='#ff7f0e', alpha=0.6, label='Popularity-branch weight 1 − α')
        for i, m in enumerate(means):
            ax.text(x[i], m / 2, f"{m:.2f}", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(HIST_LABELS, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Gate weight', fontsize=12)
        ax.set_title('Learned Gate: Branch Contribution by User History Length (Day 7)', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.tight_layout()
        plt.savefig(fig_path('gate_alpha_by_history.png'), dpi=300)
        plt.close()

    # Save all summary stats to JSON
    results_summary = {
        'config': {**vars(args), 'tag': tag},
        'popular_sets': {str(d.date()): len(p) for d, p in zip(all_days, pop_lists)},
        'single_seed': metrics_of(test_results),
        'multi_seed': {
            'seeds': seed_results,
            'mean': {'auc': float(np.mean(aucs)), 'mrr': float(np.mean(mrrs)), 'ndcg5': float(np.mean(n5s)), 'ndcg10': float(np.mean(n10s))},
            'std': {'auc': float(np.std(aucs)), 'mrr': float(np.std(mrrs)), 'ndcg5': float(np.std(n5s)), 'ndcg10': float(np.std(n10s))}
        },
        'cold_start': cold_start,
        'hourly': hourly_summary.to_dict(orient='list'),
        'history': history,
    }
    if fallback is not None:
        results_summary['hard_fallback'] = fallback
    if abl_results is not None:
        results_summary['ablation'] = {'full': metrics_of(test_results), 'no_self_attn': abl_results}

    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    print("\nAll experiments, evaluations, and figures successfully finished and saved!")
    print("=" * 70)

if __name__ == '__main__':
    main()
