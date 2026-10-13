"""Independently audit saved demo decisions: SVD conditioning + SciPy graph paths.

Does not import rbmf_cs scoring, constraint or deployment implementations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra


def validate(directory):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    result = json.loads((directory/'deployment.json').read_text())
    manifest = json.loads((directory/'run.json').read_text())
    data = np.load(directory/'synthetic_data.npz')
    artifacts = np.load(directory/'training_artifacts.npz')
    psi, xy, ice = artifacts['basis'], data['coordinates_km'], data['ice_fraction']
    cfg, planning = config['constraints'], config['planning']
    if cfg['coordinate_system'] != 'cartesian_km':
        raise ValueError('this independent synthetic auditor requires Cartesian km')
    gateway = np.array(config['deployment']['gateways_km'], float).reshape(-1,2)
    checked_constraints = 0
    energy_error = 0.
    def audit_feasibility(report):
        nonlocal checked_constraints, energy_error
        ids = report['selected']; p = len(ids)
        coords = np.vstack((xy[ids],gateway))
        distance = np.sqrt(np.sum((coords[:,None,:]-coords[None,:,:])**2,axis=2))
        links = (distance <= cfg['communication_radius_km']) & ~np.eye(len(coords),dtype=bool)
        components = connected_components(links,directed=False,return_labels=False)
        communication_ok = components == 1
        expected_energy = None
        if communication_ok:
            weights = np.where(links,cfg['tx_electronics_j_per_packet']+
                               cfg['tx_amplifier_j_per_packet_km2']*distance**2,0.)
            paths = dijkstra(weights,directed=False)
            sinks = list(range(p,len(coords))) if len(gateway) else [0]
            # Sum each source's shortest total route cost, independent of the
            # implementation's parent-tree forwarding bookkeeping.
            tx = np.min(paths[:p,sinks],axis=1).sum()*cfg['packets_per_sensor']
            expected_energy = p*cfg['mission_seconds']*(cfg['sense_power_w']+cfg['process_power_w'])+tx+len(gateway)*cfg['gateway_energy_j']
            actual_energy = report['energy']['total_j']
            energy_error = max(energy_error,abs(expected_energy-actual_energy))
            np.testing.assert_allclose(actual_energy,expected_energy,atol=1e-10,rtol=1e-12)
        else:
            assert report['energy'] is None
        valid_ice = np.isfinite(ice[ids]) & (ice[ids]>=0) & (ice[ids]<=cfg['ice_threshold'])
        expected = bool(np.all(valid_ice) and p<=cfg['max_sensors'] and communication_ok and
                        expected_energy<=cfg['energy_budget_j'] and set(config['deployment']['sensing_anchors']).issubset(ids))
        assert report['feasible']==expected
        assert report['connected']==communication_ok
        checked_constraints += 1
    audit_feasibility(result['initial_feasibility'])
    audit_feasibility(result['terminal_feasibility'])
    delta_u_error = 0.
    delta_u_checks = 0
    for decision in result['decisions']:
        for report in decision['prospective_feasibility'].values():
            audit_feasibility(report)
        feasible = sorted(int(j) for j,r in decision['prospective_feasibility'].items() if r['feasible'])
        scores = decision['scores']; rows = scores['candidates']
        assert [r['node'] for r in rows]==feasible
        snapshot = np.load(directory/manifest['snapshots'][decision['round']]['posterior'])
        sigma = snapshot['cov_alpha']
        prior_design = np.linalg.solve(np.linalg.cholesky(sigma),np.eye(len(sigma)))
        before = np.trace(psi@sigma@psi.T)
        for row in rows:
            augmented = np.vstack((prior_design,psi[row['node']]/np.sqrt(row['planning_variance'])))
            _,s,vt = np.linalg.svd(augmented,full_matrices=False)
            direct_covariance = (vt.T/s**2)@vt
            direct_gain = before-np.trace(psi@direct_covariance@psi.T)
            delta_u_error = max(delta_u_error,abs(direct_gain-row['delta_u_raw']))
            np.testing.assert_allclose(row['delta_u_raw'],direct_gain,atol=1e-11,rtol=1e-9)
            delta_u_checks += 1
        hmax = max((r['delta_h_raw'] for r in rows),default=0.)
        umax = max((r['delta_u_raw'] for r in rows),default=0.)
        assert scores['normalization']['max_delta_h']==hmax
        assert scores['normalization']['max_delta_u']==umax
        for row in rows:
            np.testing.assert_allclose(row['delta_h_normalized'],row['delta_h_raw']/(hmax+planning['epsilon_h']))
            np.testing.assert_allclose(row['delta_u_normalized'],row['delta_u_raw']/(umax+planning['epsilon_u']))
        if 'chosen_node' in decision:
            expected_node = min(rows,key=lambda r:(-r['eir'],r['node']))['node']
            assert decision['chosen_node']==expected_node
    events = result['events']
    for i,event in enumerate(events):
        if event['event']=='acquire':
            assert events[i-1]['event']=='score' and events[i+1]['event']=='refit'
            assert events[i+1]['fit_count']==events[i-1]['fit_count']+1
    for i,snapshot in enumerate(manifest['snapshots']):
        fit = json.loads((directory/snapshot['inference']).read_text())
        assert fit['diagnostics']['fit_count']==i+1 and fit['diagnostics']['converged']
        if i:
            assert fit['diagnostics']['warm_start']
    final_energy = result['terminal_feasibility']['energy']['total_j']
    np.testing.assert_allclose(result['initial_feasibility']['energy']['total_j']+
        sum(r['energy_change_j'] for r in result['trajectory']),final_energy,atol=1e-10)
    audit = {'success':True,'feasibility_checks':checked_constraints,'delta_u_checks':delta_u_checks,
             'max_delta_u_absolute_error':delta_u_error,'max_energy_absolute_error_j':energy_error,
             'final_mission_energy_j':final_energy,'posterior_fits':len(manifest['snapshots']),
             'stop_reason':result['stop_reason']}
    (directory/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    data.close(); artifacts.close()
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,
                        default=Path(__file__).resolve().parents[1]/'outputs/closed_loop_demo')
    args=parser.parse_args()
    print(json.dumps(validate(args.directory),indent=2))
