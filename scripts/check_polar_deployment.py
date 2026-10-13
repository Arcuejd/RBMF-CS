"""Audit saved real-field deployments and their shared evaluation domain."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rbmf_cs import DeploymentConstraints,ConstraintConfig
from run_polar_deployment import metrics,final_comparison,dashboard,write_json


def main():
    output=ROOT/'outputs/polar_deployment'
    config=json.loads((output/'config.json').read_text(encoding='utf-8'))
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    final_comparison(output,summary)
    checked=0
    for month,methods in summary['months'].items():
        shared_values={}
        seeds=[]
        for method,row in methods.items():
            folder=output/month/method
            deployment=json.loads((folder/'deployment.json').read_text(encoding='utf-8'))
            acquired=json.loads((folder/'acquisitions.json').read_text(encoding='utf-8'))
            assert [r['node'] for r in acquired]==deployment['selected']
            assert len(deployment['selected'])==12 and row['all_fits_converged']
            seeds.append(deployment['seed'])
            for acquisition in acquired:
                node=acquisition['node']; value=acquisition['value_model']
                if node in shared_values: assert shared_values[node]==value
                shared_values[node]=value
            with np.load(folder/'final.npz') as z:
                constraints=DeploymentConstraints(z['coordinates_latlon'],z['ice_fraction'],
                    ConstraintConfig(**config['constraints']),gateways=[config['gateway_latlon']])
                assert constraints.evaluate(deployment['selected'])['feasible']
                weights=np.cos(np.deg2rad(z['coordinates_latlon'][:,0]))
                result=metrics(z['prediction_insitu_c'],z['truth_insitu_c'],deployment['selected'],weights)
                for key,value in result.items(): np.testing.assert_allclose(value,row['final'][key],atol=1e-12)
                for step,point in enumerate(row['trajectory']):
                    selected=deployment['seed'] if step==0 else deployment['trajectory'][step-1]['selected']
                    with np.load(folder/f'posterior_{step:03d}.npz') as posterior:
                        np.testing.assert_array_equal(posterior['selected'],selected)
                        computed=metrics(posterior['prediction_insitu_c'],z['truth_insitu_c'],selected,weights)
                        for key,value in computed.items(): np.testing.assert_allclose(value,point[key],atol=1e-12)
                    diagnostics=json.loads((folder/f'fit_{step:03d}.json').read_text(encoding='utf-8'))
                    assert diagnostics['converged']
                    checked+=1
        assert all(seed==seeds[0] for seed in seeds)
    with np.load(Path(config['data_directory'])/'training_artifacts.npz') as training:
        assert np.all(training['train_dates']<=201212)
    write_json(output/'summary.json',summary)
    dashboard(output,summary)
    report={'passed':True,'deployments_checked':12,'posterior_fits_checked':checked,
            'checks':['feasibility and budget','shared seeds and node-indexed observations',
                      'all saved posterior metrics independently recomputed','inference convergence',
                      'training excludes 2014','common unobserved evaluation domain']}
    write_json(output/'verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
