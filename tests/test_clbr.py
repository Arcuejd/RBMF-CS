"""Independent mathematical references and deliberately adversarial fold tests."""
import json
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm
from rbmf_cs.clbr import (CLBR, ResidualPrior, learn_prior, physical, check_dates,
                         interval_metrics, weighted_quantiles, LEVELS)


class CLBRTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261013)
        self.b = np.linalg.qr(rng.normal(size=(9, 3)))[0]
        self.p = ResidualPrior(self.b, .2, .8, np.array([.1, -.3, .2]), .7, .04,
                               [201301, 201302, 201303], 2)
        self.low = np.linspace(-1, 1, 9)
        self.ids = np.array([0, 3, 3, 6])
        self.y = np.array([.4, .1, -.2, .6])
        self.noise = .03

    def fitted(self, **kwargs):
        return CLBR(self.p).fit(self.low, self.y, self.ids, fixed_noise_variance=self.noise, **kwargs)

    def test_posterior_augmented_svd(self):
        m = self.fitted()
        # Independently augment latent z and N persistent discrepancies with whitened prior.
        n, k = self.b.shape
        f = np.c_[self.b, np.eye(n)]
        prior_mean = np.r_[self.p.coefficient_mean, np.zeros(n)]
        prior_sd = np.sqrt(np.r_[np.full(k, .7), np.full(n, .04)])
        a = np.vstack([f[self.ids]/np.sqrt(self.noise), np.diag(1/prior_sd)])
        target = np.r_[(self.y-(.2+.8*self.low)[self.ids])/np.sqrt(self.noise), prior_mean/prior_sd]
        u, s, vh = np.linalg.svd(a, full_matrices=False)
        mean = vh.T @ ((u.T@target)/s)
        covariance = (vh.T/s**2) @ vh
        assert_allclose(m.mean_, .2+.8*self.low+f@mean, atol=1e-12)
        assert_allclose(m.variance_, np.diag(f@covariance@f.T), atol=1e-12)
        assert_allclose(m.coefficient_covariance_, covariance[:k, :k], atol=1e-12)

    def test_normalized_joint_conditional_identity(self):
        m = self.fitted()
        mu = .2+.8*self.low+self.b@self.p.coefficient_mean
        covariance = .7*self.b@self.b.T+.04*np.eye(9)
        obs = covariance[np.ix_(self.ids, self.ids)] + self.noise*np.eye(4)
        post = covariance-covariance[:, self.ids]@np.linalg.solve(obs, covariance[self.ids])
        x = mu + np.linspace(-.2, .2, 9)
        logjoint = multivariate_normal.logpdf(x, mu, covariance) + np.sum(norm.logpdf(self.y, x[self.ids], np.sqrt(self.noise)))
        logposterior = multivariate_normal.logpdf(x, m.mean_, post)
        evidence = multivariate_normal.logpdf(self.y, mu[self.ids], obs)
        self.assertAlmostEqual(logjoint-logposterior, evidence, places=10)
        self.assertAlmostEqual(evidence, m.diagnostics_['conditional_log_evidence'], places=10)

    def test_scalar_numerical_integration(self):
        p = ResidualPrior(np.ones((1, 1)), .2, .8, np.array([.1]), .7, .04, [], 2)
        m = CLBR(p).fit([.3], [.9], [0], fixed_noise_variance=.03)
        density = lambda x: norm.pdf(x, .2+.8*.3+.1, np.sqrt(.74))*norm.pdf(.9, x, np.sqrt(.03))
        z = quad(density, -np.inf, np.inf, epsabs=1e-11)[0]
        first = quad(lambda x: x*density(x), -np.inf, np.inf, epsabs=1e-11)[0]/z
        second = quad(lambda x: (x-first)**2*density(x), -np.inf, np.inf, epsabs=1e-11)[0]/z
        assert_allclose([m.mean_[0], m.variance_[0]], [first, second], atol=1e-10)

    def test_no_observations_missing_duplicate_and_invalid(self):
        empty = CLBR(self.p).fit(self.low, [np.nan], [0])
        assert_allclose(empty.mean_, .2+.8*self.low+self.b@self.p.coefficient_mean)
        assert_allclose(empty.variance_, .7*np.sum(self.b**2, axis=1)+.04)
        self.assertEqual(empty.diagnostics_['noise_status'], 'unidentified_no_observations')
        for values, indices in [([np.inf], [0]), ([0.], [9]), ([0.], [1.2])]:
            with self.assertRaises(ValueError):
                CLBR(self.p).fit(self.low, values, indices)
        with self.assertRaises(ValueError):
            CLBR(self.p).fit(np.full(9, np.nan), [], [])
        with self.assertRaises(ValueError):
            CLBR(self.p).fit(self.low, self.y, self.ids, fixed_noise_variance=0)

    def test_units_and_interval_sampling(self):
        model = self.fitted()
        rng = np.random.default_rng(108)
        mu, v = physical(model.mean_, model.variance_, np.arange(9)*.1, .47)
        assert_allclose(v, model.variance_*.47**2)
        assert_allclose(mu, np.arange(9)*.1+.47*model.mean_)
        samples = rng.normal(model.mean_[4], np.sqrt(model.variance_[4]), 80000)
        future = samples + rng.normal(0, np.sqrt(self.noise), len(samples))
        for target, draws in [(False, samples), (True, future)]:
            lower, upper = model.intervals(target)
            coverage = ((draws[:, None]>=lower[4]) & (draws[:, None]<=upper[4])).mean(axis=0)
            assert_allclose(coverage, LEVELS, atol=.007)

    def test_deterministic_equal_on_unobserved_scoring_domain(self):
        model = self.fitted()
        e = np.array([1, 2, 4, 5, 7, 8])
        assert_allclose(model.deterministic_prediction()[e], model.mean_[e], atol=1e-12)

    def test_added_observation_variance_and_response(self):
        before = CLBR(self.p).fit(self.low, self.y[:1], self.ids[:1], fixed_noise_variance=self.noise)
        after = self.fitted()
        self.assertTrue(np.all(after.variance_ <= before.variance_+1e-12))
        shifted = self.y.copy(); shifted[0] += .04
        changed = CLBR(self.p).fit(self.low, shifted, self.ids, fixed_noise_variance=self.noise)
        assert_allclose(changed.mean_-after.mean_, .04*after.gain_[:, 0], atol=1e-12)
        strong = CLBR(self.p, multiplier=1e-8).fit(self.low, [], [])
        assert_allclose(strong.mean_, .2+.8*self.low+self.b@self.p.coefficient_mean)

    def test_training_affine_reference_and_constant_lf(self):
        rng = np.random.default_rng(4)
        low = rng.normal(size=(6, 9))
        x = .3+1.2*low+rng.normal(0, .1, low.shape)
        dates = list(range(201301, 201307)); w = np.arange(1, 10)
        p = learn_prior(x, low, dates, dates, self.b, w)
        wn = np.tile(w/w.sum()/6, 6); flat = low.ravel(); target = x.ravel()
        slope = np.sum(wn*(flat-np.sum(wn*flat))*(target-np.sum(wn*target)))/np.sum(wn*(flat-np.sum(wn*flat))**2)
        intercept = np.sum(wn*target)-slope*np.sum(wn*flat)
        assert_allclose([p.intercept, p.slope], [intercept, slope], atol=1e-12)
        constant = learn_prior(x, np.ones_like(x), dates, dates, self.b, w)
        self.assertEqual(constant.slope, 0)
        self.assertEqual(constant.calibration_rank, 1)
        full = CLBR(constant).fit(np.zeros(9), [], [])
        other_low = CLBR(constant).fit(np.ones(9)*100, [], [])
        assert_allclose(full.mean_, other_low.mean_)

    def test_fold_taint_rejected_and_heldout_cannot_change_learning(self):
        dates = np.arange(201301, 201313)
        allowed = dates[3:]
        for invalid in ([201301, 201304, 201305], [201501, 201502, 201503], [201401, 201402, 201403]):
            with self.assertRaises(ValueError):
                check_dates(invalid, allowed)
        rng = np.random.default_rng(5)
        hf = rng.normal(size=(12, 9)); lf = rng.normal(size=(12, 9))
        before = learn_prior(hf[3:], lf[3:], allowed, allowed, self.b, np.ones(9))
        hf[:3] = 1e9
        after = learn_prior(hf[3:], lf[3:], allowed, allowed, self.b, np.ones(9))
        self.assertEqual(before.metadata(), after.metadata())

    def test_noise_estimation_does_not_require_truth(self):
        m = CLBR(self.p, scale_c=.47).fit(self.low, self.y, self.ids)
        self.assertTrue(.001-1e-12 <= m.diagnostics_['noise_std_c'] <= 1+1e-12)
        self.assertIn(m.diagnostics_['noise_status'], ('lower_bound', 'upper_bound', 'interior'))

    def test_synthetic_exact_affine_control(self):
        rng = np.random.default_rng(231)
        low = rng.normal(size=(6, 9))
        high = .4+1.3*low
        dates = list(range(201301, 201307))
        p = learn_prior(high, low, dates, dates, self.b, np.ones(9))
        assert_allclose([p.intercept, p.slope], [.4, 1.3], atol=1e-12)
        independent = rng.normal(size=9)
        m = CLBR(p).fit(independent, [], [])
        assert_allclose(m.mean_, .4+1.3*independent, atol=1e-12)

    def test_quantiles_and_wis_against_manual_example(self):
        assert_allclose(weighted_quantiles([1, 2, 10], [.2, .6, .2]), [2, 2, 10, 10])
        lo = np.array([[-1, -2, -3, -4.]])
        hi = -lo
        result = interval_metrics([0], [0], lo, hi, [1])
        self.assertEqual(result['coverage'].tolist(), [1, 1, 1, 1])
        self.assertAlmostEqual(result['wis'], np.sum((1-LEVELS)*np.array([1, 2, 3, 4]))/4.5)

    def test_verifier_float32_area_weights(self):
        import math
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root/'scripts'))
        from verify_clbr_p1 import area_average
        weights = np.array([.1, .2, .3, .7], dtype=np.float32)
        for values in (np.array([.3, .5, 1.2, -.4]), np.array([True, False, True, True])):
            expected = math.fsum(float(a)*float(b) for a, b in zip(values, weights))/math.fsum(map(float, weights))
            self.assertAlmostEqual(area_average(values, weights), expected, places=14)
            self.assertAlmostEqual(area_average(values, weights), np.average(values, weights=weights), places=14)

    def test_io_gate_blocks_2014_and_sealed_fields(self):
        root = Path(__file__).resolve().parents[1]
        code = '''from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()/'src'))
from rbmf_cs.p0_isolation import install_development_guard
root=Path.cwd()
install_development_guard(root, root/'outputs/clbr_p1_synthetic_gate')
for p in [root/'outputs/p0_holdout_v1/sealed/temperature.npz', root/'outputs/comparison_formal_v1/arrays.npz', Path('D:/Dataset/forbidden.nc')]:
    try:
        p.open('rb')
    except PermissionError:
        continue
    raise RuntimeError('forbidden data not denied')
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
