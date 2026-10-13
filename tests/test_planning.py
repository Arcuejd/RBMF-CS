import unittest
import numpy as np
from numpy.testing import assert_allclose
from rbmf_cs import RBMFCS, TrainingCovariance, PlanningConfig, MCEIR
from rbmf_cs.planning import (structural_entropy_gain, uncertainty_reduction,
    rank_one_covariance, redundancy, normalize_scores, select_next)


class PlanningTests(unittest.TestCase):
    def test_delta_u_matches_independent_augmented_svd_trace_difference(self):
        rng=np.random.default_rng(785)
        psi=rng.normal(size=(8,3))  # deliberately NOT orthonormal
        a=rng.normal(size=(9,3))
        def covariance_from_svd(design):
            _,s,vt=np.linalg.svd(design,full_matrices=False)
            return (vt.T/s**2)@vt
        sigma=covariance_from_svd(a)
        candidates=[1,3,5]; variances=np.array([.2,1.1,2.])
        actual=uncertainty_reduction(psi,sigma,psi[candidates],variances)
        expected=[]
        for j,r in zip(candidates,variances):
            direct=covariance_from_svd(np.vstack((a,psi[j]/np.sqrt(r))))
            expected.append(np.trace(psi@sigma@psi.T)-np.trace(psi@direct@psi.T))
            assert_allclose(rank_one_covariance(sigma,psi[j],r),direct,atol=1e-13)
            self.assertGreaterEqual(np.linalg.eigvalsh(sigma-direct).min(),-1e-13)
        assert_allclose(actual,expected,rtol=1e-12,atol=1e-13)

    def test_structural_entropy_matches_independent_ridge_regression(self):
        rng=np.random.default_rng(987)
        training=TrainingCovariance.fit(rng.normal(size=(17,6)))
        cfg=PlanningConfig(epsilon_k=.13,entropy_reference_variance=.7)
        selected=[0,3]; candidates=[1,2,5]
        actual=structural_entropy_gain(training,selected,candidates,cfg)
        design=np.vstack((training.factor[selected].T,np.sqrt(cfg.epsilon_k)*np.eye(2)))
        expected=[]
        for j in candidates:
            response=np.concatenate((training.factor[j],np.zeros(2)))
            coeff=np.linalg.lstsq(design,response,rcond=None)[0]
            energy=np.sum((response-design@coeff)**2)
            expected.append(.5*np.log1p(energy/.7))
        assert_allclose(actual,expected,atol=1e-13)

    def test_redundancy_absolute_correlation_and_empty_selection(self):
        x=np.array([-2.,-1.,0.,1.,2.])
        cov=TrainingCovariance.fit(np.column_stack((x,-x,np.zeros(5))))
        assert_allclose(redundancy(cov,[0],[1,2],1e-12),[1.,0.],atol=1e-12)
        assert_allclose(redundancy(cov,[],[0,1],1e-12),0.)
        assert_allclose(structural_entropy_gain(cov,[],[2],PlanningConfig()),[0.])

    def test_normalization_zeros_equal_positive_and_actual_denominators(self):
        cfg=PlanningConfig(epsilon_h=.1,epsilon_u=.2)
        h,u,r,s=normalize_scores([0,0],[0,0],[.2,.8],cfg)
        assert_allclose(h,0); assert_allclose(u,0); assert_allclose(r,[.2,.8])
        h,u,r,s=normalize_scores([2,2],[3,3],[0,0],cfg)
        assert_allclose(h,2/2.1); assert_allclose(u,3/3.2)
        self.assertEqual(s['delta_h_denominator'],2.1)
        self.assertEqual(s['delta_u_denominator'],3.2)

    def test_scores_only_normalize_feasible_set_and_use_historical_reliability(self):
        rng=np.random.default_rng(13); psi=rng.normal(size=(5,2))
        m=RBMFCS(psi).fit(np.ones(5),[.1,.2],indices=[0,1])
        cov=TrainingCovariance.fit(rng.normal(size=(12,5)))
        cfg=PlanningConfig(maintenance_weight=2.,failure_weight=3.)
        planner=MCEIR(cov,np.ones(5),np.ones(5)*.5,np.ones(5)*.2,cfg,
                      historical_reliability=[1,1,2,3,4])
        scores=planner.score_candidates(m,[0,1],[3])
        self.assertEqual(len(scores['candidates']),1)
        row=scores['candidates'][0]
        self.assertAlmostEqual(row['planning_variance'],1/(float(m.tau_.mean)*3))
        self.assertAlmostEqual(row['soft_penalty'],1.6)
        self.assertAlmostEqual(row['effective_cost'],2.6+cfg.epsilon_cost)
        self.assertAlmostEqual(row['eir'],row['effective_information']/row['effective_cost'])
        self.assertEqual(scores['normalization']['max_delta_h'],row['delta_h_raw'])
        self.assertEqual(scores['normalization']['max_delta_u'],row['delta_u_raw'])
        # No candidate's future y or lambda appears in the scoring interface.
        self.assertEqual(planner.score_candidates(m,[0,1],[])['candidates'],[])

    def test_maximum_eir_tie_empty_and_threshold_stops(self):
        scores={'candidates':[{'node':7,'eir':.4},{'node':2,'eir':.9},{'node':1,'eir':.9}]}
        self.assertEqual(select_next(scores,0.)[0]['node'],1)
        self.assertEqual(select_next(scores,.9),(None,'insufficient_marginal_value'))
        self.assertEqual(select_next({'candidates':[]},0.),(None,'no_feasible_candidate'))
        self.assertEqual(select_next({'candidates':[{'node':1,'eir':-.2}]},0.)[1],'insufficient_marginal_value')

    def test_zero_gains_produce_no_artificial_information(self):
        cov=TrainingCovariance.fit(np.ones((4,3)))
        m=RBMFCS(np.ones((3,1))).fit(np.zeros(3),[],indices=[])
        m.cov_alpha_[:]=0  # degenerate PSD planning fixture
        planner=MCEIR(cov,np.zeros(3),np.zeros(3),np.zeros(3))
        scores=planner.score_candidates(m,[],[0,1,2])
        self.assertTrue(all(row['eir']==0. for row in scores['candidates']))
        self.assertEqual(select_next(scores,0.)[1],'insufficient_marginal_value')

    def test_invalid_planning_inputs(self):
        with self.assertRaises(ValueError): PlanningConfig(epsilon_u=0.)
        with self.assertRaises(ValueError): PlanningConfig(eir_threshold=-1.)
        with self.assertRaises(ValueError): uncertainty_reduction(np.eye(2),-np.eye(2),np.eye(2),1.)
        cov=TrainingCovariance.fit(np.ones((4,3)))
        with self.assertRaises(ValueError): MCEIR(cov,[1]*3,[0]*3,[0]*3,historical_reliability=[1,0,1])

    def test_redundancy_dominating_information_remains_negative(self):
        cov=TrainingCovariance.fit(np.arange(8.)[:,None]*np.ones((1,3)))
        model=RBMFCS(np.ones((3,1))).fit(np.ones(3),[1.,1.],indices=[0,1])
        planner=MCEIR(cov,np.ones(3),np.zeros(3),np.zeros(3),PlanningConfig(beta=1.,eta=3.))
        scores=planner.score_candidates(model,[0,1],[2])
        self.assertLess(scores['candidates'][0]['effective_information'],0.)
        self.assertEqual(select_next(scores,0.)[1],'insufficient_marginal_value')


if __name__ == '__main__': unittest.main()
