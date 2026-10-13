"""Regression test for the runnable configured demo (including its Python API)."""
import contextlib, importlib.util, io, json, tempfile, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('synthetic_demo',ROOT/'scripts/run_synthetic_demo.py')
demo=importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoTests(unittest.TestCase):
    def test_configured_demo_converges_and_persists_each_refit(self):
        output_root = (ROOT/'outputs').resolve()
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as tmp, contextlib.redirect_stdout(io.StringIO()):
            # Keep artifacts and automatic cleanup strictly inside the workspace.
            self.assertTrue(Path(tmp).resolve().is_relative_to(output_root))
            result=demo.run(ROOT/'configs/synthetic_demo.json',tmp)
            self.assertEqual(result['status'],'completed')
            self.assertEqual([r['fit_count'] for r in result['trajectory']],[1,2,3,4])
            self.assertEqual([r['observed_count'] for r in result['trajectory']],[11,12,13,14])
            for step in range(4):
                fit=json.loads((Path(tmp)/f'fit_{step:02d}.json').read_text())
                self.assertTrue(fit['diagnostics']['converged'])
                self.assertEqual(fit['diagnostics']['elbo_decreases'],0)
                self.assertEqual(fit['diagnostics']['max_precision_jitter'],0)
                with np.load(Path(tmp)/f'posterior_{step:02d}.npz') as data:
                    self.assertEqual(data['variance_physical'].shape,(120,))
                    self.assertTrue(np.isfinite(data['mean_physical']).all())
                    self.assertNotIn('cov_x',data.files)


if __name__=='__main__': unittest.main()
