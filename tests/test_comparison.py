import inspect
import unittest
from dataclasses import replace
import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize
from rbmf_cs import RBMFCS, VBConfig, TrainingCovariance, PlanningConfig, DeploymentConstraints, ConstraintConfig
from rbmf_cs.comparison import (ridge, lasso, entropy_weights, qr_residual_norm,
    ComparisonPlanner, DEPLOYMENT, reconstruct, perturbation, SelectedObserver, fixed_domain, metrics)
from rbmf_cs.comparison_statistics import holm, paired_intervals


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(391)
        self.basis = np.linalg.qr(self.rng.normal(size=(12, 5)))[0]
        self.low = self.rng.normal(size=12)
        self.cfg = VBConfig(max_iter=1000, tolerance=1e-7)

    def test_ridge_matches_augmented_lstsq_underdetermined(self):
        h = self.basis[:3]
        y = np.array([1., 2., -1.])
        expected = np.linalg.lstsq(np.vstack((h, np.sqrt(.3)*np.eye(5))), np.r_[y, np.zeros(5)], rcond=None)[0]
        np.testing.assert_allclose(ridge(h, y, .3), expected, atol=1e-12)

    def test_lasso_independent_split_variable_optimization(self):
        h = self.basis[:3]
        y = np.array([1., 2., -1.])
        a, diag = lasso(h, y, .05)
        def objective(z):
            error = h@(z[:5]-z[5:])-y
            return .5*(error@error)+.05*z.sum()
        result = minimize(objective, np.ones(10)*.1, bounds=[(0, None)]*10,
                          method='L-BFGS-B', options={'ftol': 1e-14, 'gtol': 1e-9, 'maxiter': 10000})
        self.assertTrue(diag['converged'])
        self.assertAlmostEqual(.5*np.sum((h@a-y)**2)+.05*abs(a).sum(), result.fun, places=7)

    def test_qr_matches_independent_scipy_column_pivots(self):
        _, _, pivots = qr(self.basis.T, pivoting=True)
        selected = []
        for target in pivots[:5]:
            candidates = [j for j in range(12) if j not in selected]
            chosen = candidates[int(np.argmax(qr_residual_norm(self.basis, selected, candidates)))]
            self.assertEqual(chosen, target)
            selected.append(chosen)

    def test_entropy_negative_indicator_and_weight_direction(self):
        x = np.array([[0., 0., 9.], [0., 1., 9.], [2., 1., 9.]])
        # negative indicator probabilities: column0=(1/2,1/2,0); col1=(1,0,0).
        information = np.array([1-np.log(2)/np.log(3), 1., 0.])
        np.testing.assert_allclose(entropy_weights(x), information/information.sum())
        self.assertGreater(entropy_weights(x)[1], entropy_weights(x)[0])
        with self.assertRaises(ValueError):
            entropy_weights(np.ones((3, 4)))

    def test_qr_stops_at_exhausted_rank_without_roundoff_pivots(self):
        np.testing.assert_array_equal(qr_residual_norm(self.basis, list(range(5)), list(range(5, 12))), np.zeros(7))

    def test_no_mf_removes_factor_not_just_low_input(self):
        cfg = replace(self.cfg, multifidelity=False)
        a = RBMFCS(self.basis, cfg).fit(self.low, [1., -1.], indices=[0, 1])
        b = RBMFCS(self.basis, cfg).fit(self.low*100, [1., -1.], indices=[0, 1])
        np.testing.assert_array_equal(a.mean_alpha_, b.mean_alpha_)
        self.assertEqual(a._elbo(), b._elbo())
        zero_low = RBMFCS(self.basis, self.cfg).fit(np.zeros(12), [1., -1.], indices=[0, 1])
        self.assertGreater(np.linalg.norm(a.cov_alpha_-zero_low.cov_alpha_), 1e-3)
        self.assertAlmostEqual(float(a.delta_.mean), cfg.initial_tau_delta)

    def test_gaussian_sbl_stationary_reference(self):
        model = RBMFCS(self.basis, replace(self.cfg, multifidelity=False, robust_likelihood=False,
                       convergence='factors', tolerance=1e-8, max_iter=5000))
        model.fit(self.low, [1., -1., .3], indices=[0, 1, 3])
        np.testing.assert_array_equal(model.local_.mean, np.ones(3))
        # Independent augmented-design solve at converged hyperparameters.
        h = model.h_
        design = np.vstack((np.sqrt(float(model.tau_.mean))*h, np.diag(np.sqrt(model.gamma_.mean))))
        y = np.r_[np.sqrt(float(model.tau_.mean))*model.y_, np.zeros(5)]
        expected = np.linalg.lstsq(design, y, rcond=None)[0]
        np.testing.assert_allclose(model.mean_alpha_, expected, atol=1e-6)

    def test_shared_noise_reproducibility_and_independent_stream(self):
        a = perturbation(12, 0, 201401, 'student', stream=20)
        b = perturbation(12, 0, 201401, 'student', stream=20)
        c = perturbation(12, 0, 201401, 'student', stream=21)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertFalse(np.array_equal(a[0], c[0]))
        sample = perturbation(200000, 0, 1, 'student')[0]
        self.assertLess(abs(np.std(sample)-.02), .002)

    def test_evaluation_exclusion_and_acquisition_contract(self):
        ev, pool = fixed_domain(np.arange(30))
        observer = SelectedObserver(np.arange(30), pool)
        with self.assertRaises(ValueError):
            observer.acquire(int(ev[0]))
        self.assertEqual(observer.acquire(int(pool[0])), float(pool[0]))
        with self.assertRaises(ValueError):
            observer.acquire(int(pool[0]))
        self.assertFalse(set(ev)&set(pool))

    def test_planner_no_truth_interface_and_constraints_all_methods(self):
        signature = inspect.signature(ComparisonPlanner.score_candidates)
        self.assertEqual(list(signature.parameters), ['self', 'model', 'selected', 'feasible_candidates'])
        train = self.rng.normal(size=(15, 12))
        coordinates = np.c_[np.full(12, 75.), np.arange(12)]
        constraints = DeploymentConstraints(coordinates, np.r_[np.zeros(8), np.ones(4)],
            ConstraintConfig(coordinate_system='latlon_degrees', communication_radius_km=200,
                             max_sensors=6, energy_budget_j=1e9))
        selected = [0, 1]
        feasible, reports = constraints.filter_candidates(selected, list(range(12)))
        model = RBMFCS(self.basis, self.cfg).fit(self.low, [1., -1.], indices=selected)
        for method in DEPLOYMENT:
            planner = ComparisonPlanner(TrainingCovariance.fit(train), np.ones(12), np.zeros(12),
                np.zeros(12), PlanningConfig(), method=method, coordinates=coordinates,
                priorities=self.rng.random(12), entropy_weight=entropy_weights(train))
            rows = planner.score_candidates(model, selected, feasible)['candidates']
            self.assertEqual({r['node'] for r in rows}, set(feasible))
            self.assertTrue(all(constraints.evaluate(selected+[r['node']])['feasible'] for r in rows))

    def test_saved_metric_definition_and_missing_baselines(self):
        result = metrics(np.array([2., 3., 99.]), np.array([0., 0., 0.]), [0, 1], [1., 3., 1.], 2.)
        self.assertAlmostEqual(result['rmse_c'], np.sqrt(31/4))
        self.assertAlmostEqual(result['nrmse'], np.sqrt(31/4)/2)
        a, _, _, _ = reconstruct('pod_ridge', self.basis, self.low, [0, 1], [1., np.nan])
        b, _, _, _ = reconstruct('pod_ridge', self.basis, self.low, [0], [1.])
        np.testing.assert_array_equal(a, b)

    def test_holm_and_correlated_pair_identity(self):
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
        e = self.rng.normal(size=(2, 4, 12))
        coords = np.c_[np.arange(12), np.arange(12)*2]
        result = paired_intervals(e, e, np.ones(12), coords, replicates=50)
        self.assertEqual(result['paired_mean_difference_c'], 0)
        self.assertEqual(result['time_space_block_sensitivity95_c'], [0., 0.])
        self.assertIsNone(result['environment_p_value'])
        single = paired_intervals(e[:, :1], e[:, :1], np.ones(12), coords, replicates=50)
        self.assertIsNone(single['time_space_block_sensitivity95_c'])

if __name__ == '__main__':
    unittest.main()
