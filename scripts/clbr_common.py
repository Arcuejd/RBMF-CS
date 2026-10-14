"""P1 provenance and strict development inputs; no performance work on import."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rbmf_cs.p0_isolation import install_development_guard
from rbmf_cs.comparison import fixed_domain
from run_p0_diagnosis import load_development


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()), encoding='utf-8')


def source_files():
    files = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT/folder).rglob('*.py')]
    files += [ROOT/'configs/clbr_p1_v1.json', ROOT/'configs/p0_strong_baselines_v1.json']
    files += list((ROOT/'docs').glob('clbr_p1_*.md'))
    return sorted(files)


def source_hash():
    pairs = [(p.relative_to(ROOT).as_posix(), digest(p)) for p in source_files()]
    return hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest()


def preservation():
    seal = json.loads((ROOT/'outputs/p0_holdout_v1/seal.json').read_text())
    hold = digest(ROOT/'outputs/p0_holdout_v1/manifest.json')
    if seal['release_authorized'] or hold != seal['manifest_sha256']:
        raise ValueError('holdout must remain sealed with unchanged manifest')
    hashes = {'original_model': digest(ROOT/'src/rbmf_cs/model.py'),
              'training': digest(ROOT/'outputs/comparison_validation_v1/inputs/training_artifacts.npz'),
              'selection': digest(ROOT/'outputs/comparison_validation_v1/selection.json')}
    expected = {'original_model': '5e7173f7aca50511e4f1afff8a25150fea419eb9958008258b8b4781589c3168'}
    # Training digest is read from the preserved P0 freeze to avoid an alternative contract.
    baseline = json.loads((ROOT/'configs/p0_strong_baselines_v1.json').read_text())
    expected['training'] = baseline['training_artifacts_sha256']
    expected['selection'] = baseline['parameter_source_sha256']
    if hashes != expected:
        raise ValueError('original model/training/selection changed')
    return {**hashes, 'holdout_manifest': hold, 'release_authorized': False}


def load_inputs():
    source = ROOT/'outputs/comparison_validation_v1/inputs'
    data = load_development(source)
    with np.load(source/'training_artifacts.npz', allow_pickle=False) as f:
        art = {k: f[k] for k in f.files}
    with np.load(source/'aligned_fields.npz', allow_pickle=False) as f:
        if 'train_lf_model' in f.files:
            raise ValueError('protocol assumes no audited paired training LF; review protocol first')
        low_c = f['validation_lf_insitu_c']
    scale = float(art['scale_c'])
    base = art['monthly_climatology_c'] + art['mean_c']
    for key, physical in [('validation_hf_model', data['validation_hf_insitu_c']), ('validation_lf_model', low_c)]:
        if not np.allclose(data[key]*scale+base, physical, rtol=0, atol=1e-12):
            raise ValueError('HF/LF physical transform changed')
    e, _ = fixed_domain(data['original_grid_flat_ids'])
    w = np.cos(np.deg2rad(data['coordinates_latlon'][:, 0]))
    designs = json.loads((ROOT/'outputs/comparison_validation_v1/jobs/rbmf_000/validation_designs.json').read_text())
    design_map = {}
    for row in designs:
        if row['date'] != int(data['validation_dates'][row['i']]) or set(row['ids']) & set(e):
            raise ValueError('design dates or scoring domain inconsistent')
        design_map[(row['date'], row['seed'])] = row
    selection = json.loads((ROOT/'outputs/comparison_validation_v1/selection.json').read_text())
    return data, art, e, w, design_map, selection


def new_output(path):
    path = Path(path).resolve()
    if path.parent != ROOT/'outputs' or not path.name.startswith('clbr_p1_'):
        raise ValueError('use a new outputs/clbr_p1_* directory')
    if path.exists():
        raise FileExistsError(path)
    return path


def snapshot(out):
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in source_files():
            z.write(p, p.relative_to(ROOT).as_posix())
