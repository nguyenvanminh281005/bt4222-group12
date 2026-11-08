"""Inference-only NRMS latent regression: both checkpoints and exact dot scores."""
import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import hashlib
    from sklearn.decomposition import PCA
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import rankdata
    from utils.metrics import ndcg_at_k
    torch.set_num_threads(1)
    notebook=json.loads((ROOT/'models/7_nrms/7_2_nrms_saved_latent.ipynb').read_text(encoding='utf-8'))
    sources=[c['source'] for c in notebook['cells'] if c['cell_type']=='code']
    def find(prefix):return next(s for s in sources if s.startswith(prefix))
    assert not any('def train_' in s or 'torch.optim.' in s for s in sources)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        torch.manual_seed(42)
        scope=dict(np=np,pd=pd,torch=torch,nn=nn,F=F,math=__import__('math'),json=json,Path=Path,
            PCA=PCA,LinearSegmentedColormap=LinearSegmentedColormap,plt=plt,hashlib=hashlib,
            rankdata=rankdata,ndcg_at_k=ndcg_at_k,device=torch.device('cpu'),SOURCE_N_HEADS=16,
            SEED=42,SPLIT='test',SAMPLE_IMPRESSIONS=4,ENCODE_BATCH=5,USER_BATCH=2,VERIFY_FORWARD_RECORDS=12,
            BLUE_LIGHT='#9ecae1',BLUE_MEDIUM='#4292c6',BLUE_DARK='#08519c',display=lambda *args:None,
            _TIE_RNG=np.random.default_rng(0),feature_id='synthetic-fixture',PATHS={'features':root})
        for cls in ('AttentionPooling','NewsEncoder','UserEncoder','NRMS','GatedNRMS'):
            exec(find('class '+cls+'('),scope)
        exec(find('def eval_impression('),scope)
        code=find('def load_saved_nrms(')
        definition=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef))
        exec(compile(ast.Module(body=[definition],type_ignores=[]),'<loader>','exec'),scope)
        lookup=torch.randint(1,20,(12,5));lookup[0]=0
        pop=torch.tensor([[1,2,0],[3,4,0]])
        records=[]
        for i,n in enumerate([0,1,4,7]):
            hist=[0]*(50-n)+list(range(1,n+1))
            records.append(dict(history=hist,candidates=[8,9,10],labels=[1,0,0],day_idx=i%2,
                history_len=n,impression_id=100+i,timestamp=pd.Timestamp('2019-11-15')+pd.Timedelta(hours=i)))
        payload=dict(news_title_lookup=lookup,pop_sets=pop,max_hist_len=50,
            popularity_config=dict(per_cat=2,extra=10,min_cat_clicks=20),test_records=records,val_records=records)
        pd.to_pickle(payload,root/'nrms_features.pkl')
        for gated in (False,True):
            cls=scope['GatedNRMS'] if gated else scope['NRMS']
            kwargs=dict(d_model=16,n_heads=16,query_dim=6,pop_sets=pop,dropout=.2)
            if gated:kwargs.update(gate_hidden=4,max_hist_len=50)
            model=cls(torch.randn(20,8),lookup,**kwargs).eval()
            checkpoint=root/('gated.pt' if gated else 'nrms.pt')
            torch.save(model.state_dict(),checkpoint)
            before=checkpoint.read_bytes()
            out=root/('gated' if gated else 'nrms');out.mkdir()
            scope.update(CHECKPOINT=checkpoint,SUMMARY_PATH=root/'missing_summary.json',OUT_DIR=out)
            def savefig(name):
                plt.tight_layout();plt.savefig(scope['OUT_DIR']/name);plt.close('all')
            scope['savefig']=savefig
            # Any optimizer construction would reveal an unintended fitting path.
            with patch.object(torch.optim,'Adam',side_effect=AssertionError('Training is forbidden')):
                for prefix in ('def load_saved_nrms(','records=payload[','news_map=','pair_history='):
                    exec(compile(find(prefix),'<nrms-latent-cell>','exec'),scope)
            assert checkpoint.read_bytes()==before
            lat=scope['latent']
            assert lat['verified_max_error']<1e-4
            assert lat['verified_slates']==4
            assert len(lat['scores'])==12
            assert scope['user_xy'].shape==(4,2)
            assert scope['news_xy'].shape==(12,2)
            assert (out/'nrms_user_news_latent.png').is_file()
            assert json.loads((out/'nrms_latent_manifest.json').read_text())['training_runs']==0
            if not gated:
                assert np.all(lat['user_vectors'][0]==0)
                assert np.all(lat['scores'][lat['pair_group']==0]==0)
                assert np.isnan(scope['cosine'][lat['pair_group']==0]).all()
            else:
                assert np.allclose(lat['user_vectors'],lat['alpha'][:,None]*lat['history_vectors']+
                    (1-lat['alpha'][:,None])*lat['popularity_vectors'])
            bad=dict(payload,news_title_lookup=lookup.flip(0))
            try:scope['load_saved_nrms'](checkpoint,bad)
            except ValueError:pass
            else:raise AssertionError('Wrong lookup must fail')
        try:scope['load_saved_nrms'](root/'missing.pt',payload)
        except FileNotFoundError:pass
        else:raise AssertionError('Missing weights must not trigger training')
        print('PASS: NRMS and gated checkpoint inference; exact dot products; cold histories; common PCA; distributions; zero training; invalid references rejected')

if __name__=='__main__':main()
