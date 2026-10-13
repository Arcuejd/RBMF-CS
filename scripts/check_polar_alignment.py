"""Independently check the persisted real-data alignment, without modifying it."""
import json
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('D:/Dataset/RBMF-CS_polar_sample/aligned_50m')


def main():
    report = json.loads((BASE/'alignment_report.json').read_text())
    results = {}
    for name in report['regions']:
        folder = BASE/name
        with np.load(folder/'aligned_fields.npz') as f, np.load(folder/'training_artifacts.npz') as a:
            assert len(f['train_dates']) == 156 and np.all(f['train_dates'] <= 201212)
            assert len(f['validation_dates']) == 12 and np.all(f['validation_dates']//100 == 2013)
            assert f['test_dates'].tolist() == [201401,201404,201407,201410]
            np.testing.assert_array_equal(a['train_dates'], f['train_dates'])
            for split in ['train','validation','test']:
                for fidelity in (['hf','lf'] if split == 'test' else ['hf']):
                    stem = split+'_'+fidelity
                    absolute = f[stem+'_insitu_c']
                    assert np.isfinite(absolute).all()
                    expected = absolute-a['monthly_climatology_c'][f[split+'_dates']%100-1]
                    np.testing.assert_allclose(f[stem+'_anomaly_c'],expected,atol=1e-12)
                    np.testing.assert_allclose(f[stem+'_model'],(expected-a['mean_c'])/a['scale_c'],atol=1e-12)
            climate = np.stack([f['train_hf_insitu_c'][f['train_dates']%100==m].mean(axis=0) for m in range(1,13)])
            np.testing.assert_allclose(climate,a['monthly_climatology_c'],atol=1e-12)
            np.testing.assert_allclose(a['basis'].T@a['basis'],np.eye(a['basis'].shape[1]),atol=1e-10)
            energy = np.sum((f['train_hf_model']@a['basis'])**2)/np.sum(f['train_hf_model']**2)
            assert energy >= .99 and np.isclose(energy,a['pod_energy_fraction'])
            np.testing.assert_allclose(f['lf_source_weights'].sum(axis=1),1,atol=1e-12)
            rows = json.loads((folder/'aligned_observations.json').read_text())
            for row in rows:
                nodes, weights = np.array(row['node_ids']),np.array(row['weights'])
                assert (weights>0).all() and np.isclose(weights.sum(),1)
                assert nodes.min()>=0 and nodes.max()<len(f['node_ids'])
                anomaly = row['temperature_insitu_c']-weights@climate[row['month']%100-1,nodes]
                np.testing.assert_allclose(row['anomaly_c'],anomaly,atol=1e-12)
                np.testing.assert_allclose(row['model_value'],(anomaly-weights@a['mean_c'][nodes])/a['scale_c'],atol=1e-12)
            ice = f['ice_fraction']
            assert np.all((ice[np.isfinite(ice)]>=0)&(ice[np.isfinite(ice)]<=1))
            results[name] = {'train_shape':list(f['train_hf_model'].shape),'pod_energy':float(energy),
                             'observations_checked':len(rows),'all_checks_passed':True}
    for relative,expected in report['output_files'].items():
        h=hashlib.sha256()
        with (BASE/relative).open('rb') as stream:
            while chunk := stream.read(1024*1024):
                h.update(chunk)
        assert h.hexdigest()==expected, relative
    output=ROOT/'outputs/polar_alignment_verification.json'
    output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
