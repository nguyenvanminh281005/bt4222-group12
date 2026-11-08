"""Verify saved-weight reuse and exactly one hour-ablation fit on synthetic data."""
import ast
import json
import tempfile
from pathlib import Path

from verify_mlp_study import FIXTURE, ROOT


def main():
    nb=json.loads((ROOT/'models/6_mlp/6_4_mlp_saved_best_hour_latent.ipynb').read_text(encoding='utf-8'))
    sources=[c['source'] for c in nb['cells'] if c['cell_type']=='code']
    def find(prefix):return next(s for s in sources if s.startswith(prefix))
    with tempfile.TemporaryDirectory() as temp:
        scope={'OUT_DIR':temp}
        exec(FIXTURE,scope)
        scope.update(OUT_DIR=Path(temp),EVAL_BATCH=4,SELECT_METRIC='nDCG@5',USE_GROUPS=dict(
            article=True,popularity=True,user=True,user_x_article=True,context=True))
        scope['LATENT_IMPRESSIONS']=4
        scope['POP_COLS']=scope['meta']['features']['popularity']
        scope['INTER_COLS']=scope['meta']['features']['user_x_article']
        config_source=find('prepared_inputs =')
        functions=[n for n in ast.parse(config_source).body if isinstance(n,ast.FunctionDef)]
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<test-config>','exec'),scope)
        cfg=scope['make_config']('no_optional',scope['USE_GROUPS'],['diversity_ratio','habit_score'])
        scope['FULL_CFG']=cfg
        architecture=find('def set_seed(')
        exec(architecture,scope)
        scope['set_seed'](42)
        model=scope['build_model'](cfg)
        weights=Path(temp)/'final_model.pt'
        config=Path(temp)/'run_config.json'
        torch=scope['torch']
        torch.save(model.state_dict(),weights)
        hp=dict(feature_id=scope['feature_id'],seed=42,embed_dim=4,cat_dim=4,subcat_dim=4,
            hidden=[12,8],dropout=.1,id_dropout=.1,lr=.001,weight_decay=1e-5,
            batch_size=4,epochs=1,patience=1,select_metric='nDCG@5')
        metadata=dict(final_model='Improved MLP',hyperparameters=hp,
            dropped_features=cfg['drop'],final_features=scope['active_features'](cfg),
            cold_user_simulation={'rate':.15},cold_user_fallback=False)
        config.write_text(json.dumps(metadata))
        scope.update(REFERENCE_WEIGHTS=weights,REFERENCE_CONFIG=config,REFERENCE_DIR=Path(temp))
        previous_bytes=weights.read_bytes()
        calls=[]
        selected=[find('for p in (REFERENCE_WEIGHTS,'),find('METRIC_NAMES ='),architecture,
            find('COLD_PROFILE ='),find('def batch_from('),find('state=torch.load('),
            find('rng=np.random.default_rng(SEED)'),find('NO_HOUR_CFG='),
            find('def recorded_hours('),find('no_hour_latent=')]
        training_source=find('NO_HOUR_CFG=')
        for source in selected:
            exec(compile(source,'<interactive-hour-cell>','exec'),scope)
            if source.startswith('def batch_from('):
                original=scope['train_model']
                def counted(*args,**kwargs):
                    calls.append((args,kwargs))
                    return original(*args,**kwargs)
                scope['train_model']=counted
        assert len(calls)==1
        assert calls[0][1]['seed']==42
        assert calls[0][0][0]['drop']==['diversity_ratio','habit_score','hour_cos','hour_sin']
        assert weights.read_bytes()==previous_bytes
        assert len(scope['latent_summary'])==4
        assert 'space' not in scope['latent_summary'].columns
        np=scope['np']
        for name in ('with_hour_saved','without_hour'):
            with np.load(Path(temp)/f'pair_hidden_{name}.npz') as export:
                assert export['hidden'].shape==(len(scope['pair_rows']),8)
                assert export['pca'].shape==(len(scope['pair_rows']),2)
                assert np.array_equal(export['labels'],scope['sample_labels'])
                assert np.isfinite(export['scores']).all()
                assert str(export['representation'])=='final_hidden_linear_pre_relu'
                model=scope['reference'] if name=='with_hour_saved' else scope['without_hour']
                captured=[]
                hook=model.mlp[-4].register_forward_hook(lambda module,args,out:captured.append(out.detach().cpu().numpy()))
                with torch.no_grad():
                    scope['forward'](model,scope['batch_from'](scope['A']['test'],[int(scope['sample_g'][0])]))
                hook.remove()
                assert np.allclose(export['hidden'][0],captured[0][0,0]),'Hidden must be the Linear output before ReLU'
        assert len(scope['pair_rows'])==sum(scope['A']['test']['n'][scope['sample_g']])
        assert scope['latent_summary']['histogram_overlap'].between(0,1.000001).all()
        assert len(scope['hourly']['val'])==len(scope['hourly']['test'])==24
        assert scope['recorded_hours']('test').tolist()==[0,0,0,1,1,7,8,8,9,20,23,23]
        exec(training_source,scope)
        assert len(calls)==1,'Rerunning must load the completed ablation, not fit again'
        b=scope['batch_from'](scope['A']['val'],[2])
        b.pop('label')
        b={k:v.to(scope['device']) for k,v in b.items()}
        with torch.no_grad():
            before=scope['without_hour'](**b)
            b['ctx_x']+=10
            after=scope['without_hour'](**b)
            assert torch.allclose(before,after)
        scope['REFERENCE_WEIGHTS']=Path(temp)/'missing.pt'
        try:exec(selected[0],scope)
        except FileNotFoundError:pass
        else:raise AssertionError('Missing reference must stop, never retrain')
        scope['REFERENCE_WEIGHTS']=weights
        metadata['hyperparameters']['feature_id']='wrong-bundle'
        config.write_text(json.dumps(metadata))
        try:exec(selected[0],scope)
        except ValueError:pass
        else:raise AssertionError('Wrong feature bundle must be rejected')
        assert len(calls)==1
        print('PASS: saved weights unchanged; exactly one matched-seed fit; cache reuse; missing/mismatched reference rejected; hourly and latent outputs')


if __name__=='__main__':main()
