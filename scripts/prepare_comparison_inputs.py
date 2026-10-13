"""Audit the supplied 2013 package and create a NEW combined experiment input."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from run_comparison import ROOT, digest, write_json, preflight


def prepare(validation, output, config_output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'configs/polar_deployment.json').read_text(encoding='utf-8'))
    original = Path(config['data_directory'])
    region = validation/'aligned_50m/atlantic'
    checks = []
    def require(ok, name):
        checks.append({'name': name, 'passed': bool(ok)})
        if not ok:
            write_json(output/'audit.json', {'passed': False, 'checks': checks})
            raise ValueError(name)
    downloads = json.loads((validation/'downloads.json').read_text(encoding='utf-8'))
    for item in downloads['records']:
        p = Path(item['path'])
        require(p.stat().st_size == item['bytes'] and digest(p) == item['sha256'], 'download checksum '+p.name)
    require(digest(original/'training_artifacts.npz') == digest(region/'training_artifacts.npz'), 'training byte-identical')
    with np.load(original/'aligned_fields.npz') as z:
        combined = {k: z[k] for k in z.files}
    with np.load(region/'validation_fields.npz') as z:
        val = {k: z[k] for k in z.files}
    art = np.load(region/'training_artifacts.npz')
    for key in ('validation_dates', 'coordinates_latlon', 'original_grid_flat_ids', 'valid_domain_mask',
                'validation_hf_insitu_c', 'validation_hf_anomaly_c', 'validation_hf_model'):
        require(np.array_equal(combined[key], val[key], equal_nan=True), 'unchanged '+key)
    require(np.array_equal(val['validation_dates'], np.arange(201301, 201313)), 'all twelve validation months in order')
    require(np.isfinite(val['validation_lf_model']).all(), 'finite LF model')
    require(val['validation_lf_model'].shape == (12, 346), 'fixed LF shape')
    ice = val['ice_fraction']
    require(np.all((ice[np.isfinite(ice)] >= 0) & (ice[np.isfinite(ice)] <= 1)), 'ice fraction units')
    require(np.allclose(val['lf_source_weights'].sum(axis=1), 1), 'LF interpolation weights')
    require(np.max(val['ice_distance_km']) <= 40, 'ice mapping distance')
    baseline = art['monthly_climatology_c']+art['mean_c']
    require(np.allclose(val['validation_lf_model']*float(art['scale_c'])+baseline,
                        val['validation_lf_insitu_c'], rtol=0, atol=1e-12), 'LF physical reconstruction')
    require(np.allclose(val['validation_lf_anomaly_c']+art['monthly_climatology_c'],
                        val['validation_lf_insitu_c'], rtol=0, atol=1e-12), 'LF climatology equation')
    for key in ('validation_lf_model', 'validation_lf_insitu_c', 'validation_lf_anomaly_c'):
        combined[key] = val[key]
    for key in ('ice_fraction', 'ice_qa_flags', 'ice_source_flat_ids', 'ice_distance_km'):
        combined['validation_'+key] = val[key]
    np.savez_compressed(output/'aligned_fields.npz', **combined)
    shutil.copyfile(region/'training_artifacts.npz', output/'training_artifacts.npz')
    sources = [original/'aligned_fields.npz', original/'training_artifacts.npz', region/'validation_fields.npz',
               region/'training_artifacts.npz', validation/'audit_report.json', validation/'downloads.json']
    write_json(output/'audit.json', {'passed': True, 'checks': checks,
        'source_sha256': {str(p): digest(p) for p in sources},
        'policy': 'only new validation keys added; existing arrays and original files unchanged',
        'evaluation_status': '2014 exploratory; no unseen paired environment'})
    config['data_directory'] = str(output.resolve())
    write_json(config_output, config)
    write_json(output/'preflight.json', preflight(config))
    print('Audited checks:', len(checks), 'config:', config_output)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config-output', type=Path, required=True)
    a = p.parse_args()
    prepare(a.validation, a.output, a.config_output)
