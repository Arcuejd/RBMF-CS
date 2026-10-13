import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose
from rbmf_cs import (RBMFCS, VBConfig, TrainingCovariance, ConstraintConfig,
                     DeploymentConstraints, PlanningConfig, MCEIR, run_closed_loop)
from rbmf_cs.planning import uncertainty_reduction


class ClosedLoopTests(unittest.TestCase):
    def setup_problem(self, max_sensors=5, threshold=0., inference=None):
        rng=np.random.default_rng(110)
        psi=np.linalg.qr(rng.normal(size=(8,3)))[0]
        model=RBMFCS(psi,inference or VBConfig(tolerance=1e-8))
        low=psi@np.array([.6,-.9,1.2])+.03*rng.normal(size=8)
        truth=psi@np.array([1.1,-.8,.5])
        constraints=DeploymentConstraints(np.column_stack((np.arange(8),np.zeros(8))),
            np.zeros(8),ConstraintConfig(max_sensors=max_sensors,communication_radius_km=20.),
            gateways=[[-1.,0.]],anchors=[0])
        planner=MCEIR(TrainingCovariance.fit(rng.normal(size=(30,8))),np.ones(8),np.zeros(8),
                      np.zeros(8),PlanningConfig(eir_threshold=threshold))
        return model,low,truth,constraints,planner

    def test_real_acquire_refit_rescore_order_and_energy_accounting(self):
        model,low,truth,constraints,planner=self.setup_problem()
        calls=[]
        def acquire(j):
            calls.append((j,model.fit_count_,len(model.y_)))
            return truth[j]+.07
        with patch.object(model,'update',wraps=model.update) as update, \
             patch.object(model,'_run',wraps=model._run) as vb:
            result=run_closed_loop(model,low,[0,1],truth[:2],constraints,planner,acquire)
            self.assertEqual(update.call_count,3)
            self.assertEqual(vb.call_count,4)
        self.assertEqual(result['stop_reason'],'sensor_budget_reached')
        self.assertEqual([c[1] for c in calls],[1,2,3])
        self.assertEqual([c[2] for c in calls],[2,3,4])
        self.assertEqual([e['event'] for e in result['events']],
                         ['fit_seed']+['score','acquire','refit']*3)
        snapshots=result['snapshots']
        for t,decision in enumerate(result['decisions']):
            scores=decision['scores']
            self.assertEqual(scores['posterior_fit_count'],t+1)
            rows=scores['candidates']; ids=[r['node'] for r in rows]
            self.assertTrue(all(decision['prospective_feasibility'][j]['feasible'] for j in ids))
            expected=uncertainty_reduction(model.basis,snapshots[t]['cov_alpha'],model.basis[ids],
                                          np.array([r['planning_variance'] for r in rows]))
            assert_allclose([r['delta_u_raw'] for r in rows],expected,atol=1e-13)
            self.assertEqual(result['trajectory'][t]['score']['eir'],max(r['eir'] for r in rows))
            after=snapshots[t+1]
            self.assertTrue(after['diagnostics']['warm_start'])
            self.assertEqual(after['diagnostics']['new_observations'],1)
            self.assertEqual(after['diagnostics']['initial_factors']['gamma'],snapshots[t]['factors']['gamma'])
            self.assertEqual(after['diagnostics']['initial_factors']['lambda']['mean'][-1],1.)
            for name in ('gamma','tau','tau_delta'):
                self.assertFalse(np.allclose(snapshots[t]['factors'][name]['mean'],after['factors'][name]['mean']))
            self.assertNotEqual(snapshots[t]['factors']['rho']['mean'],after['factors']['rho']['mean'])
            self.assertGreater(np.linalg.norm(after['mean_alpha']-snapshots[t]['mean_alpha']),1e-7)
            if t:
                stale=uncertainty_reduction(model.basis,snapshots[t-1]['cov_alpha'],model.basis[ids],
                                           np.array([r['planning_variance'] for r in rows]))
                self.assertGreater(np.linalg.norm(expected-stale),1e-7)
        initial=result['initial_feasibility']['energy']['total_j']
        changes=sum(x['energy_change_j'] for x in result['trajectory'])
        self.assertAlmostEqual(initial+changes,result['terminal_feasibility']['energy']['total_j'])
        self.assertTrue(all(x['feasibility']['connected'] for x in result['trajectory']))

    def test_future_values_affect_only_scores_after_acquisition(self):
        results=[]
        for offset in (0.,2.):
            m,low,truth,c,p=self.setup_problem(max_sensors=4)
            results.append(run_closed_loop(m,low,[0,1],truth[:2],c,p,lambda j:truth[j]+offset))
        self.assertEqual(results[0]['decisions'][0]['scores'],results[1]['decisions'][0]['scores'])
        self.assertEqual(results[0]['trajectory'][0]['selected_node'],results[1]['trajectory'][0]['selected_node'])
        a=results[0]['decisions'][1]['scores']; b=results[1]['decisions'][1]['scores']
        self.assertGreater(np.linalg.norm(np.array([x['delta_u_raw'] for x in a['candidates']])-
                                         np.array([x['delta_u_raw'] for x in b['candidates']])),1e-7)

    def test_budget_empty_candidates_and_insufficient_eir_stops(self):
        for budget,threshold,pool,reason in [(2,0.,None,'sensor_budget_reached'),
              (5,0.,[],'no_feasible_candidate'),(5,1e6,None,'insufficient_marginal_value')]:
            m,low,truth,c,p=self.setup_problem(budget,threshold)
            calls=[]
            r=run_closed_loop(m,low,[0,1],truth[:2],c,p,lambda j:calls.append(j),candidates=pool)
            self.assertEqual(r['stop_reason'],reason); self.assertEqual(calls,[])

    def test_infeasible_candidates_never_scored_or_acquired(self):
        m,low,truth,c,p=self.setup_problem(max_sensors=3)
        c.set_ice([0.,0.,1.,1.,1.,1.,1.,0.])
        r=run_closed_loop(m,low,[0,1],truth[:2],c,p,lambda j:truth[j])
        self.assertEqual(r['trajectory'][0]['selected_node'],7)
        self.assertEqual([row['node'] for row in r['decisions'][0]['scores']['candidates']],[7])
        self.assertEqual(r['decisions'][0]['scores']['normalization']['feasible_count'],1)

    def test_environment_change_stops_existing_infeasible_deployment(self):
        m,low,truth,c,p=self.setup_problem()
        def refresh(round_id,selected,constraints):
            if round_id==1:
                ice=np.zeros(8); ice[0]=1.; constraints.set_ice(ice)
        r=run_closed_loop(m,low,[0,1],truth[:2],c,p,lambda j:truth[j],refresh_environment=refresh)
        self.assertEqual(r['stop_reason'],'existing_deployment_infeasible')
        self.assertEqual(len(r['trajectory']),1)
        self.assertIn('ice_threshold',r['terminal_feasibility']['reasons'])

    def test_missing_measurement_counts_sensor_but_is_removed_from_fit(self):
        m,low,truth,c,p=self.setup_problem(max_sensors=3)
        r=run_closed_loop(m,low,[0,1],truth[:2],c,p,lambda j:np.nan)
        self.assertEqual(len(r['selected']),3); self.assertEqual(len(m.y_),2)
        self.assertEqual(m.fit_count_,2)
        self.assertTrue(r['trajectory'][0]['observation_missing'])
        self.assertGreater(r['trajectory'][0]['cumulative_mission_energy_j'],r['initial_feasibility']['energy']['total_j'])

    def test_nonconvergence_stops_before_acquisition(self):
        m,low,truth,c,p=self.setup_problem(inference=VBConfig(max_iter=1,min_iter=1,tolerance=1e-15))
        calls=[]
        r=run_closed_loop(m,low,[0,1],truth[:2],c,p,lambda j:calls.append(j))
        self.assertEqual(r['stop_reason'],'inference_not_converged'); self.assertEqual(calls,[])

    def test_invalid_seed_rejected_before_fitting(self):
        m,low,truth,c,p=self.setup_problem()
        with self.assertRaises(ValueError): run_closed_loop(m,low,[],[],c,p,lambda j:0.)
        with self.assertRaises(ValueError): run_closed_loop(m,low,[1],[truth[1]],c,p,lambda j:0.)
        c.set_ice(np.ones(8))
        with self.assertRaises(ValueError): run_closed_loop(m,low,[0,1],truth[:2],c,p,lambda j:0.)
        self.assertEqual(m.fit_count_,0)


if __name__ == '__main__': unittest.main()
