import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClosedLoopDemoTests(unittest.TestCase):
    def test_configured_demo_with_independent_saved_artifact_audit(self):
        output_root=(ROOT/'outputs').resolve(); output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as tmp,contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(Path(tmp).resolve().is_relative_to(output_root))
            manifest=load_script('run_closed_loop_demo').run(ROOT/'configs/closed_loop_demo.json',tmp)
            self.assertEqual(manifest['status'],'completed')
            self.assertTrue(manifest['all_fits_converged'])
            self.assertGreater(manifest['posterior_fits'],2)
            audit=load_script('validate_closed_loop_run').validate(tmp)
            self.assertTrue(audit['success'])
            self.assertGreater(audit['delta_u_checks'],0)
            self.assertGreater(audit['feasibility_checks'],10)
            deployment=json.loads((Path(tmp)/'deployment.json').read_text())
            self.assertTrue(any(r['observation_missing'] for r in deployment['trajectory']))
            self.assertTrue((Path(tmp)/'trajectory.csv').is_file())


if __name__=='__main__': unittest.main()
