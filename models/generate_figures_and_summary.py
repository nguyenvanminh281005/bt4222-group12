"""
Generate figures and summary JSON using the trained best checkpoint,
including the Popularity-Gated NRMS (PG-NRMS) cold-start mitigation experiment.
"""
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('models')
from train_and_evaluate import (
    NRMS, NewsEncoder, UserEncoder, AttentionPooling,
    evaluate_model, eval_impression, set_seed
)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Load News
print("Loading News Catalogue...")
train_news = pd.read_csv('data/train/news.tsv', sep='\t', header=None, 
                         names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
dev_news = pd.read_csv('data/dev/news.tsv', sep='\t', header=None, 
                       names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
all_news = pd.concat([train_news, dev_news]).drop_duplicates(subset=['news_id']).reset_index(drop=True)

def tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r'\w+|[^\w\s]', text.lower())

word_counts = {}
for title in all_news['title']:
    for w in tokenize(title):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = {'<PAD>': 0, '<UNK>': 1}
for w, c in word_counts.items():
    if c >= 2:
        vocab[w] = len(vocab)

news2idx = {'<PAD>': 0}
for nid in all_news['news_id']:
    news2idx[nid] = len(news2idx)

max_title_len = 30
lookup = np.zeros((len(news2idx), max_title_len), dtype=np.int64)
for row in all_news.itertuples(index=False):
    idx = news2idx[row.news_id]
    tokens = [vocab.get(w, 1) for w in tokenize(row.title)][:max_title_len]
    lookup[idx, :len(tokens)] = tokens

news_title_lookup = torch.tensor(lookup, dtype=torch.long, device=device)

# 2. Extract Top Popular Articles from Train Interactions
print("Extracting Top Popular News Articles from Train Interactions...")
train_b = pd.read_csv('data/train/behaviors.tsv', sep='\t', header=None, 
                      names=['imp_id', 'user_id', 'time', 'history', 'impressions'])
train_b['time'] = pd.to_datetime(train_b['time'])
train_df = train_b[train_b['time'] < '2019-11-14']

click_counts = {}
for row in train_df.itertuples(index=False):
    for item in row.impressions.split():
        nid, lbl = item.split('-')
        if lbl == '1' and nid in news2idx:
            click_counts[news2idx[nid]] = click_counts.get(news2idx[nid], 0) + 1

top_popular_ids = [nid for nid, _ in sorted(click_counts.items(), key=lambda x: x[1], reverse=True)[:20]]
print(f"Top 20 popular articles identified in training interactions.")

# 3. Load Test Behaviors
print("Loading Test Behaviors...")
dev_b = pd.read_csv('data/dev/behaviors.tsv', sep='\t', header=None, 
                    names=['imp_id', 'user_id', 'time', 'history', 'impressions'])
dev_b['time'] = pd.to_datetime(dev_b['time'])

max_hist_len = 50
test_records = []
for row in dev_b.itertuples(index=False):
    raw_h = row.history.split() if isinstance(row.history, str) else []
    hist_len = len(raw_h)
    hist = [news2idx[x] for x in raw_h if x in news2idx][-max_hist_len:]
    if len(hist) < max_hist_len:
        hist = [0] * (max_hist_len - len(hist)) + hist
        
    cands, labels = [], []
    for item in row.impressions.split():
        nid, lbl = item.split('-')
        if nid in news2idx:
            cands.append(news2idx[nid])
            labels.append(int(lbl))
    if sum(labels) > 0 and len(labels) > sum(labels):
        test_records.append({
            'impression_id': row.imp_id,
            'timestamp': row.time,
            'history': hist,
            'history_len': hist_len,
            'candidates': cands,
            'labels': labels
        })

# 4. Model Pre-Encoding
dummy_embed = torch.zeros(len(vocab), 300)
model = NRMS(dummy_embed, news_title_lookup, d_model=256, n_heads=16, query_dim=200, dropout=0.2).to(device)
best_model_path = 'models/checkpoints/nrms_best_seed42.pt'
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print(f"Loaded weights from {best_model_path}")

print("Pre-encoding all unique news articles...")
num_news = len(news_title_lookup)
news_vecs = torch.zeros(num_news, 256, device=device)
batch_size = 2048
with torch.no_grad():
    for i in range(0, num_news, batch_size):
        chunk = news_title_lookup[i:min(i + batch_size, num_news)].to(device)
        vecs = model.news_encoder(chunk)
        news_vecs[i:min(i + batch_size, num_news)] = vecs
news_vecs[0] = 0.0

# Encode Global Popular Prior Vector
top_pop_tensor = torch.tensor(top_popular_ids, dtype=torch.long, device=device)
top_pop_vecs = news_vecs[top_pop_tensor].unsqueeze(0)
with torch.no_grad():
    u_pop = model.user_encoder(top_pop_vecs).squeeze(0)

# Batch encode test users
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

# 5. Evaluate Original NRMS vs Popularity-Gated NRMS
print("Evaluating Original vs Popularity-Gated NRMS on Day 7...")
orig_detailed = []
pg_detailed = []

for i, rec in enumerate(test_records):
    h_len = rec['history_len']
    u_pers = user_vecs[i]
    c_vecs = news_vecs[rec['candidates']]
    labels = rec['labels']
    
    # 1. Original
    sc_orig = torch.mv(c_vecs, u_pers).cpu().numpy()
    auc_o, mrr_o, n5_o, n10_o = eval_impression(labels, sc_orig)
    orig_detailed.append({
        'impression_id': rec['impression_id'],
        'timestamp': rec['timestamp'],
        'history_len': h_len,
        'hour': rec['timestamp'].hour if hasattr(rec['timestamp'], 'hour') else 0,
        'auc': auc_o, 'mrr': mrr_o, 'ndcg5': n5_o, 'ndcg10': n10_o
    })
    
    # 2. Popularity-Gated
    u_pg = u_pop if h_len == 0 else u_pers
    sc_pg = torch.mv(c_vecs, u_pg).cpu().numpy()
    auc_p, mrr_p, n5_p, n10_p = eval_impression(labels, sc_pg)
    pg_detailed.append({
        'impression_id': rec['impression_id'],
        'timestamp': rec['timestamp'],
        'history_len': h_len,
        'hour': rec['timestamp'].hour if hasattr(rec['timestamp'], 'hour') else 0,
        'auc': auc_p, 'mrr': mrr_p, 'ndcg5': n5_p, 'ndcg10': n10_p
    })

orig_df = pd.DataFrame(orig_detailed)
pg_df = pd.DataFrame(pg_detailed)

bins = [-1, 0, 5, 15, 30, 99999]
labels = ['Cold Start (0)', 'Sparse (1-5)', 'Moderate (6-15)', 'Active (16-30)', 'Power (>30)']
orig_df['history_bin'] = pd.cut(orig_df['history_len'], bins=bins, labels=labels)
pg_df['history_bin'] = pd.cut(pg_df['history_len'], bins=bins, labels=labels)

orig_cs = orig_df[orig_df['history_bin'] == 'Cold Start (0)']
pg_cs = pg_df[pg_df['history_bin'] == 'Cold Start (0)']

print("\n" + "=" * 75)
print("EXPERIMENT RESULTS: ORIGINAL NRMS vs POPULARITY-GATED NRMS (PG-NRMS)")
print("=" * 75)
print(f"Overall Test AUC:       {orig_df['auc'].mean():.4f}  ──►  {pg_df['auc'].mean():.4f} (Δ: +{pg_df['auc'].mean()-orig_df['auc'].mean():.4f})")
print(f"Overall Test MRR:       {orig_df['mrr'].mean():.4f}  ──►  {pg_df['mrr'].mean():.4f}")
print(f"Overall Test NDCG@5:    {orig_df['ndcg5'].mean():.4f}  ──►  {pg_df['ndcg5'].mean():.4f}")
print(f"Overall Test NDCG@10:   {orig_df['ndcg10'].mean():.4f}  ──►  {pg_df['ndcg10'].mean():.4f}")
print("-" * 75)
print(f"COLD-START (0) AUC:     {orig_cs['auc'].mean():.4f}  ──►  {pg_cs['auc'].mean():.4f} (GAIN: +{(pg_cs['auc'].mean()-orig_cs['auc'].mean())/orig_cs['auc'].mean()*100:+.2f}%)")
print(f"COLD-START (0) MRR:     {orig_cs['mrr'].mean():.4f}  ──►  {pg_cs['mrr'].mean():.4f} (GAIN: +{(pg_cs['mrr'].mean()-orig_cs['mrr'].mean())/orig_cs['mrr'].mean()*100:+.2f}%)")
print(f"COLD-START (0) NDCG@5:  {orig_cs['ndcg5'].mean():.4f}  ──►  {pg_cs['ndcg5'].mean():.4f} (GAIN: +{(pg_cs['ndcg5'].mean()-orig_cs['ndcg5'].mean())/orig_cs['ndcg5'].mean()*100:+.2f}%)")
print(f"COLD-START (0) NDCG@10: {orig_cs['ndcg10'].mean():.4f}  ──►  {pg_cs['ndcg10'].mean():.4f} (GAIN: +{(pg_cs['ndcg10'].mean()-orig_cs['ndcg10'].mean())/orig_cs['ndcg10'].mean()*100:+.2f}%)")
print("=" * 75)

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
os.makedirs('models/figures', exist_ok=True)

# 1. Training Curves
history = {
    'train_loss': [1.4516, 1.3688, 1.3418],
    'val_auc': [0.6193, 0.6350, 0.6435],
    'val_mrr': [0.3083, 0.3230, 0.3294],
    'val_ndcg5': [0.2849, 0.2988, 0.3047],
    'val_ndcg10': [0.3452, 0.3603, 0.3667]
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, 4), history['train_loss'], marker='o', color='#2b5c8f', lw=2.5, label='Train Loss')
axes[0].set_title('Training Cross-Entropy Loss vs. Epoch', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
axes[0].set_xticks(range(1, 4))
axes[0].legend(frameon=True)
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].plot(range(1, 4), history['val_auc'], marker='s', color='#2ca02c', lw=2, label='Validation AUC')
axes[1].plot(range(1, 4), history['val_ndcg5'], marker='^', color='#d62728', lw=2, label='Validation NDCG@5')
axes[1].plot(range(1, 4), history['val_mrr'], marker='d', color='#9467bd', lw=2, label='Validation MRR')
axes[1].plot(range(1, 4), history['val_ndcg10'], marker='v', color='#ff7f0e', lw=2, label='Validation NDCG@10')
axes[1].set_title('Validation Ranking Performance on Day 6', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Metric Score', fontsize=12)
axes[1].set_xticks(range(1, 4))
axes[1].legend(frameon=True)
axes[1].grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('models/figures/training_curves.png', dpi=300)
plt.close()

# 2. Cold Start Analysis (Original NRMS)
fig, ax = plt.subplots(figsize=(10, 5.5))
cs_df = pd.DataFrame({
    'History Length': labels,
    'AUC': [orig_df[orig_df['history_bin'] == b]['auc'].mean() for b in labels],
    'MRR': [orig_df[orig_df['history_bin'] == b]['mrr'].mean() for b in labels],
    'NDCG@5': [orig_df[orig_df['history_bin'] == b]['ndcg5'].mean() for b in labels],
    'NDCG@10': [orig_df[orig_df['history_bin'] == b]['ndcg10'].mean() for b in labels]
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
plt.savefig('models/figures/cold_start_analysis.png', dpi=300)
plt.close()

# 3. Diurnal Temporal Hourly Analysis
hourly_summary = orig_df.groupby('hour')[['auc', 'mrr', 'ndcg5', 'ndcg10']].mean().reset_index()

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
plt.savefig('models/figures/temporal_hourly_analysis.png', dpi=300)
plt.close()

# 4. Ablation Study
abl_results = {'auc': 0.6305, 'mrr': 0.3393, 'ndcg5': 0.3215, 'ndcg10': 0.3847}
fig, ax = plt.subplots(figsize=(8, 5))
metrics_names = ['AUC', 'MRR', 'NDCG@5', 'NDCG@10']
full_vals = [orig_df['auc'].mean(), orig_df['mrr'].mean(), orig_df['ndcg5'].mean(), orig_df['ndcg10'].mean()]
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
plt.savefig('models/figures/ablation_study.png', dpi=300)
plt.close()

# 5. Attention Visualization
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
plt.savefig('models/figures/attention_visualization.png', dpi=300)
plt.close()

# 6. NEW FIGURE: Cold-Start Mitigation (Original NRMS vs Popularity-Gated NRMS)
fig, ax = plt.subplots(figsize=(9, 5.5))
metrics_cs = ['AUC', 'MRR', 'NDCG@5', 'NDCG@10']
orig_cs_vals = [orig_cs['auc'].mean(), orig_cs['mrr'].mean(), orig_cs['ndcg5'].mean(), orig_cs['ndcg10'].mean()]
pg_cs_vals = [pg_cs['auc'].mean(), pg_cs['mrr'].mean(), pg_cs['ndcg5'].mean(), pg_cs['ndcg10'].mean()]

x = np.arange(len(metrics_cs))
width = 0.35
rects1 = ax.bar(x - width/2, orig_cs_vals, width, label='Original NRMS (Baseline)', color='#e41a1c', alpha=0.85)
rects2 = ax.bar(x + width/2, pg_cs_vals, width, label='Popularity-Gated NRMS (Proposed)', color='#2ca02c', alpha=0.85)

ax.set_title('Cold-Start User Performance: Original NRMS vs. Popularity-Gated NRMS', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics_cs, fontsize=11)
ax.set_ylim(0.15, 0.65)

for i in range(len(metrics_cs)):
    gain = (pg_cs_vals[i] - orig_cs_vals[i]) / orig_cs_vals[i] * 100
    ax.text(x[i] - width/2, orig_cs_vals[i] + 0.01, f"{orig_cs_vals[i]:.4f}", ha='center', fontsize=9, fontweight='bold')
    ax.text(x[i] + width/2, pg_cs_vals[i] + 0.01, f"{pg_cs_vals[i]:.4f}\n(+{gain:.1f}%)", ha='center', fontsize=9, fontweight='bold', color='#1b7837')

ax.legend(frameon=True, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('models/figures/cold_start_mitigation.png', dpi=300)
plt.close()
print("Saved models/figures/cold_start_mitigation.png successfully!")

# Multi-seed metrics
seed_results = {
    'seed42': {'auc': 0.6369, 'mrr': 0.3413, 'ndcg5': 0.3256, 'ndcg10': 0.3894},
    'seed43': {'auc': 0.6273, 'mrr': 0.3362, 'ndcg5': 0.3186, 'ndcg10': 0.3815},
    'seed44': {'auc': 0.6317, 'mrr': 0.3384, 'ndcg5': 0.3211, 'ndcg10': 0.3843}
}
aucs = [seed_results[s]['auc'] for s in seed_results]
mrrs = [seed_results[s]['mrr'] for s in seed_results]
n5s = [seed_results[s]['ndcg5'] for s in seed_results]
n10s = [seed_results[s]['ndcg10'] for s in seed_results]

# Save comprehensive JSON summary
results_summary = {
    'single_seed': {
        'auc': float(orig_df['auc'].mean()),
        'mrr': float(orig_df['mrr'].mean()),
        'ndcg5': float(orig_df['ndcg5'].mean()),
        'ndcg10': float(orig_df['ndcg10'].mean())
    },
    'multi_seed': {
        'seeds': seed_results,
        'mean': {'auc': float(np.mean(aucs)), 'mrr': float(np.mean(mrrs)), 'ndcg5': float(np.mean(n5s)), 'ndcg10': float(np.mean(n10s))},
        'std': {'auc': float(np.std(aucs)), 'mrr': float(np.std(mrrs)), 'ndcg5': float(np.std(n5s)), 'ndcg10': float(np.std(n10s))}
    },
    'cold_start': {
        b: {
            'count': int((orig_df['history_bin'] == b).sum()),
            'auc': float(orig_df[orig_df['history_bin'] == b]['auc'].mean()),
            'mrr': float(orig_df[orig_df['history_bin'] == b]['mrr'].mean()),
            'ndcg5': float(orig_df[orig_df['history_bin'] == b]['ndcg5'].mean()),
            'ndcg10': float(orig_df[orig_df['history_bin'] == b]['ndcg10'].mean())
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
            'auc': float(orig_df['auc'].mean()),
            'mrr': float(orig_df['mrr'].mean()),
            'ndcg5': float(orig_df['ndcg5'].mean()),
            'ndcg10': float(orig_df['ndcg10'].mean())
        },
        'overall_popularity_gated': {
            'auc': float(pg_df['auc'].mean()),
            'mrr': float(pg_df['mrr'].mean()),
            'ndcg5': float(pg_df['ndcg5'].mean()),
            'ndcg10': float(pg_df['ndcg10'].mean())
        }
    },
    'ablation': {
        'full': {'auc': float(orig_df['auc'].mean()), 'mrr': float(orig_df['mrr'].mean()), 'ndcg5': float(orig_df['ndcg5'].mean()), 'ndcg10': float(orig_df['ndcg10'].mean())},
        'no_self_attn': {'auc': abl_results['auc'], 'mrr': abl_results['mrr'], 'ndcg5': abl_results['ndcg5'], 'ndcg10': abl_results['ndcg10']}
    },
    'history': history
}

with open('models/figures/results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("All figures and results_summary.json updated successfully!")
