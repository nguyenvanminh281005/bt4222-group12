"""Run the study with tiny synthetic inputs; these are not project benchmarks."""
import json
import ast
import inspect
import os
import sys
import tempfile
import types
from unittest.mock import patch
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE = r'''
import os, json, time, copy, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.metrics import ndcg_at_k
torch.set_num_threads(1)
plt.show = lambda: plt.close('all')
display = lambda *args: None
device = torch.device('cpu')
FEATURE_DIR = OUT_DIR
FIG_DIR = Path(OUT_DIR)/'figures'
FIG_DIR.mkdir()
def savefig(name):
    plt.savefig(FIG_DIR/name)
    plt.close('all')
SEED=42
STUDY_SEEDS=[42]
EMBED_DIM=4
CAT_DIM=SUBCAT_DIM=4
HIDDEN=[12,8]
DROPOUT=ID_DROPOUT=0.1
LR=0.001
WEIGHT_DECAY=1e-5
BATCH_SIZE=4
MAX_EPOCHS=PATIENCE=1
COLD_SIM_RATE=0.15
HIST_LEN=50
ATT_HIDDEN=4
EVAL_ROWS=30
NDCG_KS=RECALL_KS=(5,10)
DROP_FEATURES=['diversity_ratio','habit_score']
RUN_SHAP=True
SHAP_ROWS=2
SHAP_BACKGROUND=2
SHAP_NSAMPLES=64
LATENT_IMPRESSIONS=4
BOOTSTRAP_REPEATS=10
SPARSE_MAX=5
feature_id='synthetic-not-project-data'
N_ART,N_CAT,N_SUBCAT=12,4,5
SPLITS=['train','val','test']
rng=np.random.default_rng(3)
content_emb=rng.normal(size=(N_ART,8)).astype('float32')
content_emb/=np.maximum(np.linalg.norm(content_emb,axis=1,keepdims=True),1e-9)
content_emb[0]=0
kg_emb=rng.normal(size=(N_ART,6)).astype('float32');kg_emb[0]=0
has_kg=np.ones(N_ART,dtype='float32');has_kg[0]=0
art_cat=np.arange(N_ART,dtype='int64')%3+1;art_cat[0]=0
art_subcat=np.arange(N_ART,dtype='int64')%4+1;art_subcat[0]=0
HIST=np.zeros((4,HIST_LEN),dtype='int64')
for p,n in enumerate([0,1,5,6]): HIST[p,:n]=np.arange(1,n+1)
np.save(Path(FEATURE_DIR)/'user_history_indices.npy',HIST)
profiles=pd.DataFrame({'profile_idx':range(4),'history_length':[0,1,5,6]})
user_emb=np.zeros((4,8),dtype='float32')
user_aff=np.zeros((4,N_CAT),dtype='float32')
for p in range(1,4):
    h=HIST[p][HIST[p]>0]
    user_emb[p]=content_emb[h].mean(0)
    user_aff[p]=np.bincount(art_cat[h],minlength=N_CAT)/len(h)
meta={'features':{
    'popularity':['hist_ctr','log_exposure','is_cold_article'],
    'user_x_article':['category_match','cos_user_hist','max_cos_hist','cos_last5','cat_affinity','kg_match'],
    'user':{'scalar':['log_history_length','has_history','reading_entropy']},
    'context':['hour_sin','hour_cos'],
    'optional':{'user':['diversity_ratio'],'context':['habit_score']}},
    'excluded_features':['slate_length','article_recency','session_dow']}
CAND_COLS=meta['features']['popularity']+meta['features']['user_x_article']
USER_COLS=meta['features']['user']['scalar']+['diversity_ratio']
CTX_COLS=['hour_sin','hour_cos','habit_score']
cand_mu=dict.fromkeys(CAND_COLS,0.0);cand_sd=dict.fromkeys(CAND_COLS,1.0)
ctx_mu=dict.fromkeys(CTX_COLS,0.0);ctx_sd=dict.fromkeys(CTX_COLS,1.0)
meta['imputation_means']=dict.fromkeys(CAND_COLS,0.0)
user_x_all=np.zeros((4,len(USER_COLS)),dtype='float32')
user_x_all[:,0]=np.log1p([0,1,5,6]);user_x_all[1:,1]=1
A={}
for split in SPLITS:
    n=np.full(12,5,dtype='int64');start=np.arange(12,dtype='int64')*5
    prof=np.arange(12,dtype='int64')%4
    cand_x=rng.normal(size=(60,len(CAND_COLS))).astype('float32')
    for g in np.flatnonzero(prof==0): cand_x[start[g]:start[g]+5,3:]=0
    hours=np.arange(12)*2
    ctx=np.column_stack([np.sin(2*np.pi*hours/24),np.cos(2*np.pi*hours/24),np.zeros(12)]).astype('float32')
    A[split]=dict(n=n,start=start,prof=prof,user_x=user_x_all[prof],ctx_x=ctx,
        news=rng.integers(1,N_ART,size=60),cand_x=cand_x,
        label=np.tile([1.,0.,0.,0.,0.],12).astype('float32'),
        hist_len=np.array([0,1,5,6])[prof],user_id=prof,
        is_cold_article=np.ones(60,dtype='int8'))
'''

CHECKS = r'''
assert len(SPECS)==14
assert len(folders)==14
assert all(p.joinpath('complete.json').exists() for p in folders.values())
assert len(hourly_tables['val'])==len(hourly_tables['test'])==24
assert len(latent_summary)==16
assert np.load(Path(OUT_DIR)/'shap_values.npz')['values'].shape[0]==2
assert pd.read_csv(Path(OUT_DIR)/'shap_sample_and_additivity.csv').residual.abs().max()<1e-3
for name in HOUR_MODELS:
    m=restore(name,42)
    b=singleton_batch('val',np.array([2]),np.array([10]))
    with torch.no_grad():
        before=m(**b)
        b['ctx_x']=b['ctx_x']+10
        after=m(**b)
        if 'no_context' in name: assert torch.allclose(before,after)
        else: assert not torch.allclose(before,after)
        ids=torch.tensor([[1,2,3]])
        z=m.encode_article(ids)
        cat=torch.cat([m.cat_emb(m.art_cat[ids]),m.subcat_emb(m.art_subcat[ids])],-1)
        assert torch.allclose(z,m.text_proj(m.content_emb[ids])+m.cat_proj(cat))
        if m.use_attention:
            u,w=m.attention(z,m.encode_article(torch.zeros((1,50),dtype=torch.long)),torch.zeros((1,50),dtype=torch.bool))
            assert torch.count_nonzero(u)==torch.count_nonzero(w)==0
identity=run_identity(next(iter(SPECS)),42)
LR*=2
assert identity!=run_identity(next(iter(SPECS)),42)
print('PASS: 14 retrained variants, cache reloads, validation selection, hourly tables, SHAP additivity and shared-space diagnostics')
'''

def verify_interactive_identity(notebook):
    """Reproduce Colab's source-less __main__ classes before running training."""
    source = next(c['source'] for c in notebook['cells']
                  if c['cell_type']=='code' and 'def implementation_signature(' in c['source'])
    tree = ast.parse(source)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'implementation_signature')
    scope = {'json':json}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), '<fingerprint>', 'exec'), scope)
    signature = scope['implementation_signature']
    snippet = 'class Ranker:\n    def forward(self, x):\n        return (lambda y: y + 1)(x)\n'
    module = types.ModuleType('__main__')
    with patch.dict(sys.modules, {'__main__':module}):
        exec(compile(snippet, '<colab-cell-1>', 'exec'), module.__dict__)
        try:
            inspect.getsource(module.Ranker)
        except OSError:
            pass
        else:
            raise AssertionError('Fixture did not reproduce the reported Colab error')
        original = signature(module.Ranker)
        exec(compile('\n\n'+snippet, '<colab-cell-99>', 'exec'), module.__dict__)
        assert original == signature(module.Ranker), 'Cell filenames/lines must not affect cache identity'
        exec(compile(snippet.replace('y + 1', 'y + 2'), '<colab-cell-99>', 'exec'), module.__dict__)
        assert original != signature(module.Ranker), 'Method/nested-code edits must invalidate the cache'
    print('PASS: source-less Colab class identity, stable cell relocation and nested-code invalidation')


def main():
    notebook=json.loads((ROOT/'models/6_mlp/6_3_mlp_comprehensive_ablation.ipynb').read_text(encoding='utf-8'))
    verify_interactive_identity(notebook)
    code_cells=[c['source'] for c in notebook['cells'] if c['cell_type']=='code']
    start=next(i for i,s in enumerate(code_cells) if s.startswith('F = meta['))
    with tempfile.TemporaryDirectory() as temp:
        smoke=Path(temp)/'study_smoke.py'
        smoke.write_text(FIXTURE+'\n'+'\n'.join(code_cells[start:])+'\n'+CHECKS,encoding='utf-8')
        module=types.ModuleType('study_smoke')
        module.__file__=str(smoke)
        module.OUT_DIR=temp
        sys.modules[module.__name__]=module
        exec(compile(smoke.read_text(encoding='utf-8'),str(smoke),'exec'),module.__dict__)

if __name__=='__main__':
    main()
