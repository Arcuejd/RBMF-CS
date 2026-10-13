"""Checks only: no predictions, no automatic release of sealed data."""
import hashlib,json
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def require_qualified_release(directory):
    root=Path(directory);seal=json.loads((root/'seal.json').read_text())
    if not seal.get('release_authorized'):
        raise PermissionError('Independent test remains sealed: evaluation not authorized')
    manifest_path=root/'manifest.json'
    if sha256(manifest_path)!=seal['manifest_sha256']:raise ValueError('manifest hash mismatch')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='qualified_frozen':raise ValueError('paired data not qualified')
    for field in ['final_method_sha256','final_protocol_sha256']:
        if not seal.get(field):raise ValueError('missing freeze identity: '+field)
    if sha256(manifest['sealed_artifact']['path'])!=seal['sealed_sha256']:raise ValueError('sealed data hash mismatch')
    return manifest
