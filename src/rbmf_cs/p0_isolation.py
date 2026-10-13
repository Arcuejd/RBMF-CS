"""Fail-closed development I/O gate; complements date and NPZ schema checks."""
import sys
from pathlib import Path

def install_development_guard(root, output):
    root=Path(root).resolve();output=Path(output).resolve()
    allowed_dirs=[output,root/'outputs/comparison_validation_v1/inputs']
    allowed_files=[root/'outputs/comparison_validation_v1/selection.json',
        root/'outputs/comparison_validation_v1/jobs/rbmf_000/validation_designs.json']
    outputs=root/'outputs';dataset=Path('D:/Dataset').resolve()
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)): return
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
        if p.is_relative_to(dataset): raise PermissionError('P0 development: raw dataset access denied')
        if p.is_relative_to(outputs) and not (p in allowed_files or any(p.is_relative_to(q) for q in allowed_dirs)):
            raise PermissionError('P0 development: non-allowlisted output access denied')
    sys.addaudithook(hook)
    return hook
