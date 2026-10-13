import unittest
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose
from rbmf_cs.constraints import ConstraintConfig, DeploymentConstraints, distances_km


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.config = ConstraintConfig(ice_threshold=.5, max_sensors=2, energy_budget_j=14.,
            communication_radius_km=1., mission_seconds=10., sense_power_w=.1,
            process_power_w=.2, packets_per_sensor=1, tx_electronics_j_per_packet=1.,
            tx_amplifier_j_per_packet_km2=0., gateway_energy_j=5.)
        self.xy = np.array([[1.,0.], [2.,0.], [8.,0.]])

    def make(self, config=None, ice=None, **kwargs):
        return DeploymentConstraints(self.xy, [.5,.2,.1] if ice is None else ice,
            config or self.config, gateways=[[0.,0.]], **kwargs)

    def test_all_hard_boundaries_and_cumulative_relay_energy(self):
        constraints = self.make()
        one, two = constraints.evaluate([0]), constraints.evaluate([0,1])
        self.assertTrue(one['feasible']); self.assertTrue(two['feasible'])
        # Independent packet ledger: A sends own+relayed packet; B sends one.
        # Sensing+processing = 3 J/node; gateway = 5 J. Totals 9 and 14 J.
        self.assertEqual(one['energy']['total_j'],9.)
        self.assertEqual(two['energy']['total_j'],14.)
        self.assertEqual(two['energy']['per_sensor']['0']['tx_j'],2.)
        self.assertEqual(two['energy']['per_sensor']['1']['tx_j'],1.)
        self.assertEqual(two['sensor_count'],2)  # gateway excluded from node count
        self.assertEqual(two['energy']['gateway_total_j'],5.)
        self.assertEqual(two['energy']['routes']['1'],['sensor:1','sensor:0','gateway:0'])
        self.assertGreater(two['energy']['total_j']-one['energy']['total_j'],
                           two['energy']['per_sensor']['1']['total_j'])

    def test_each_hard_violation_excludes_candidate(self):
        cases = [(replace(self.config,ice_threshold=.49),[.5,.2,.1],'ice_threshold'),
                 (replace(self.config,max_sensors=1),[.5,.2,.1],'sensor_budget'),
                 (replace(self.config,energy_budget_j=13.999),[.5,.2,.1],'energy_budget'),
                 (replace(self.config,communication_radius_km=.999),[.5,.2,.1],'connectivity')]
        for cfg,ice,reason in cases:
            with self.subTest(reason=reason):
                f,reports=self.make(cfg,ice).filter_candidates([0],[1])
                self.assertEqual(f,[]); self.assertIn(reason,reports[1]['reasons'])

    def test_unknown_ice_and_existing_node_rechecked(self):
        for unknown in (np.nan,np.inf,-.1,1.1):
            c=self.make(ice=[.5,unknown,.1]); f,r=c.filter_candidates([0],[1])
            self.assertEqual(f,[]); self.assertIn('ice_unknown_or_invalid',r[1]['reasons'])
        c=self.make(); c.set_ice([.8,.2,.1])
        f,r=c.filter_candidates([0],[1])
        self.assertEqual(f,[]); self.assertIn('ice_threshold',r[1]['reasons'])

    def test_disconnected_and_no_implicit_relay(self):
        c=self.make(); f,r=c.filter_candidates([0],[1,2])
        self.assertEqual(f,[1]); self.assertFalse(r[2]['connected'])
        self.assertIsNone(r[2]['energy'])

    def test_singleton_without_gateway_and_anchor_accounting(self):
        c=DeploymentConstraints(self.xy,[.1]*3,replace(self.config,max_sensors=1),anchors=[1])
        self.assertTrue(c.evaluate([1])['feasible'])
        self.assertEqual(c.evaluate([1])['energy']['total_j'],3.)
        self.assertIn('missing_anchor',c.evaluate([0])['reasons'])
        self.assertIn('sensor_budget',c.evaluate([0,1])['reasons'])
        self.assertFalse(c.evaluate([])['connected'])

    def test_prospective_topology_reroutes_existing_nodes(self):
        cfg=replace(self.config,communication_radius_km=4.,sense_power_w=0.,process_power_w=0.,
                    gateway_energy_j=0.,tx_amplifier_j_per_packet_km2=1.,energy_budget_j=16.)
        c=DeploymentConstraints([[4,0],[2,0]],[.1,.1],cfg,gateways=[[0,0]])
        before,after=c.evaluate([0]),c.evaluate([0,1])
        # Direct distance 4: 1+16=17. New relay: old sends 5, relay sends 2*5.
        self.assertEqual(before['energy']['total_j'],17.)
        self.assertEqual(after['energy']['total_j'],15.)
        self.assertFalse(before['feasible']); self.assertTrue(after['feasible'])
        self.assertEqual(after['energy']['per_sensor']['0']['tx_j'],5.)

    def test_latlon_units_and_date_line(self):
        d=distances_km([[0,0],[0,.1],[0,179.9],[0,-179.9]],'latlon_degrees')
        self.assertAlmostEqual(d[0,1],11.119508,places=5)
        self.assertAlmostEqual(d[2,3],22.239016,places=5)
        cfg=replace(self.config,coordinate_system='latlon_degrees',communication_radius_km=1.)
        c=DeploymentConstraints([[0,.1]],[0.],cfg,gateways=[[0,0]])
        self.assertFalse(c.evaluate([0])['connected'])

    def test_multiple_gateways_require_connected_augmented_graph(self):
        c=DeploymentConstraints([[1,0]],[0.],self.config,gateways=[[0,0],[10,0]])
        self.assertFalse(c.evaluate([0])['connected'])

    def test_validation(self):
        for settings in ({'max_sensors':-1},{'ice_threshold':2.},{'mission_seconds':0.},
                         {'coordinate_system':'degrees_as_km'},{'energy_budget_j':np.nan}):
            with self.assertRaises(ValueError): ConstraintConfig(**settings)
        with self.assertRaises(ValueError): self.make().evaluate([0,0])
        with self.assertRaises(ValueError): self.make().evaluate([-1])
        with self.assertRaises(ValueError): distances_km([[91,0]],'latlon_degrees')


if __name__ == '__main__': unittest.main()
