"""Algorithm 2: feasible scoring -> one acquisition -> complete VB -> rescore."""
from copy import deepcopy
from time import perf_counter
import numpy as np
from .constraints import node_ids
from .planning import select_next


def _snapshot(model):
    return {'mean': model.predict().copy(), 'variance': model.posterior_variance().copy(),
            'mean_alpha': model.mean_alpha_.copy(), 'cov_alpha': model.cov_alpha_.copy(),
            'factors': deepcopy(model.factor_summary()), 'diagnostics': deepcopy(model.diagnostics_)}


def run_closed_loop(model, low_fidelity, seed, seed_observations, constraints, planner,
                    acquire, *, candidates=None, refresh_environment=None):
    """Return deployment, all decisions, posterior snapshots and stopping reason.

    acquire(node_id) is invoked ONLY for the chosen node. A NaN acquisition
    keeps the physical sensor deployed but removes its statistical observation;
    its cost/count still apply and all VB factors are refitted on available data.
    refresh_environment(round, tuple(selected), constraints) may change ice or
    operational parameters before each decision. The state is then held fixed
    throughout that decision. Nonconverged inference stops further selection.
    Empty sensing seeds are deliberately rejected; no automatic relays.
    """
    start = perf_counter()
    if model.n != constraints.n or model.n != planner.n:
        raise ValueError('inference/planning/constraints grids must align')
    selected = node_ids(seed, model.n)
    if not selected:
        raise ValueError('a nonempty connected sensing seed is required')
    seed_y = np.asarray(seed_observations, float)
    if seed_y.shape != (len(selected),):
        raise ValueError('seed observations must match seed IDs in order')
    pool = list(range(model.n)) if candidates is None else node_ids(candidates, model.n)
    initial_feasibility = constraints.evaluate(selected)
    if not initial_feasibility['feasible']:
        raise ValueError(f'infeasible seed: {initial_feasibility["reasons"]}')
    model.fit(low_fidelity, seed_y, indices=selected)
    snapshots = [_snapshot(model)]
    decisions, trajectory = [], []
    events = [{'event': 'fit_seed', 'fit_count': model.fit_count_}]
    stop = None
    previous_energy = initial_feasibility['energy']['total_j']
    terminal_feasibility = initial_feasibility
    if not model.diagnostics_['converged']:
        stop = 'inference_not_converged'
    while stop is None:
        round_id = len(decisions)
        round_start = perf_counter()
        if refresh_environment is not None:
            refresh_environment(round_id, tuple(selected), constraints)
        current = constraints.evaluate(selected)
        terminal_feasibility = current
        if not current['feasible']:
            stop = 'existing_deployment_infeasible'
            break
        if len(selected) >= constraints.config.max_sensors:
            stop = 'sensor_budget_reached'
            break
        feasible, reports = constraints.filter_candidates(selected, pool)
        scores = planner.score_candidates(model, selected, feasible)
        decision = {'round': round_id, 'selected_before': selected.copy(), 'scores': scores,
                    'current_feasibility': current, 'prospective_feasibility': reports,
                    'scoring_runtime_seconds': perf_counter()-round_start}
        decisions.append(decision)
        events.append({'event': 'score', 'round': round_id, 'fit_count': model.fit_count_,
                       'feasible_candidates': feasible})
        best, stop = select_next(scores, planner.config.eir_threshold)
        if stop is not None:
            decision['stop_reason'] = stop
            break
        j = best['node']
        acquisition_start = perf_counter()
        value = float(acquire(j))
        if np.isinf(value):
            raise ValueError('acquired value cannot be infinite')
        events.append({'event': 'acquire', 'round': round_id, 'node': j,
                       'fit_count': model.fit_count_, 'missing': bool(np.isnan(value))})
        acquisition_runtime = perf_counter()-acquisition_start
        decision['chosen_node'] = j
        decision['acquired_observation'] = value if np.isfinite(value) else None
        selected.append(j)
        expanded = constraints.evaluate(selected)
        terminal_feasibility = expanded
        # Recheck the deployment actually committed, before statistical assimilation.
        if not expanded['feasible']:
            decision['expanded_feasibility'] = expanded
            stop = 'expanded_deployment_infeasible'
            break
        model.update([value], indices=[j])
        events.append({'event': 'refit', 'round': round_id, 'fit_count': model.fit_count_})
        snapshots.append(_snapshot(model))
        energy = expanded['energy']['total_j']
        record = {'round': round_id, 'selected_node': j, 'observation': value if np.isfinite(value) else None,
                  'observation_missing': bool(np.isnan(value)), 'selected': selected.copy(), 'score': best,
                  'normalization': scores['normalization'], 'feasibility': expanded,
                  'cumulative_mission_energy_j': energy, 'energy_change_j': energy-previous_energy,
                  'total_posterior_uncertainty': model.total_uncertainty(),
                  'posterior_fit_count': model.fit_count_, 'n_observations': len(model.y_),
                  'inference_converged': model.diagnostics_['converged'],
                  'inference_iterations': model.diagnostics_['iterations'],
                  'inference_runtime_seconds': model.diagnostics_['runtime_seconds'],
                  'acquisition_runtime_seconds': acquisition_runtime,
                  'runtime_seconds': perf_counter()-round_start}
        trajectory.append(record)
        previous_energy = energy
        if not model.diagnostics_['converged']:
            stop = 'inference_not_converged'
    denominator = sum(row['score']['effective_cost'] for row in trajectory)
    numerator = sum(row['score']['effective_information_raw'] for row in trajectory)
    return {'selected': selected, 'seed': node_ids(seed, model.n), 'stop_reason': stop,
            'initial_feasibility': initial_feasibility, 'terminal_feasibility': terminal_feasibility,
            'decisions': decisions, 'trajectory': trajectory, 'snapshots': snapshots, 'events': events,
            'global_eir_raw': numerator/denominator if denominator else None,
            'global_raw_gain_sum': numerator, 'global_effective_cost_sum': denominator,
            'runtime_seconds': perf_counter()-start}
