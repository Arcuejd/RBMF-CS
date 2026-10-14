"""User-run nested CLBR development pilot/full; full requires a later approval."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from clbr_common import (ROOT, digest, write, source_hash, preservation, new_output,
                         snapshot, install_development_guard, load_inputs)
from rbmf_cs.clbr import CLBR, learn_prior, LEVELS, weighted_quantiles, interval_metrics
from rbmf_cs.comparison import ridge, reconstruct, perturbation
from rbmf_cs.model import VBConfig
from run_p0_structure import memory


def run(args):
    config = json.loads((ROOT/'configs/clbr_p1_v1.json').read_text())
    source = source_hash()
    test = json.loads(Path(args.tests).read_text())
    test_digest = digest(Path(args.tests))
    if not test['passed'] or test['source_sha256'] != source:
        raise ValueError('current-source tests must pass before experiments')
    history = preservation()
    limits = config['pilot']
    plan = config['pilot'] if args.phase == 'pilot' else config['proposed_full']
    approval = None
    if args.phase == 'full':
        if not args.approval:
            raise ValueError('full batch awaits user resource confirmation after pilot')
        approval = json.loads(Path(args.approval).read_text())
        if not approval.get('user_confirmed') or approval['source_sha256'] != source or approval['plan'] != plan:
            raise ValueError('approval does not match source/plan')
        pilot = json.loads(Path(approval['pilot_verification']).read_text())
        if not pilot['passed'] or pilot['source_sha256'] != source or pilot['phase'] != 'pilot':
            raise ValueError('independent pilot verification required')
        limits = approval['limits']
    out = new_output(args.output)
    install_development_guard(ROOT, out)
    out.mkdir()
    start = time.perf_counter()
    snapshot(out)
    (out/'inner').mkdir()
    (out/'outer').mkdir()
    data, art, e, w, designs, selected = load_inputs()
    scale = float(art['scale_c'])
    basis = art['basis'][:, :config['rank']]
    dates = data['validation_dates']
    hf, lf = data['validation_hf_model'], data['validation_lf_model']
    blocks = [list(map(int, b)) for b in config['outer_blocks']]
    cfg = VBConfig(max_iter=3000, tolerance=1e-6, **selected['inference'])
    multipliers = config['prior_variance_multipliers']
    grid = config['ridge_grid']
    freeze = {'phase': args.phase, 'plan': plan, 'limits': limits, 'config': config,
              'source_sha256': source, 'preservation': history, 'approval': approval,
              'tests_sha256': test_digest,
              'input_sha256': digest(ROOT/'outputs/comparison_validation_v1/inputs/aligned_fields.npz'),
              'scale_c': scale, 'paired_training_lf_available': False}
    write(out/'freeze.json', freeze)
    np.savez_compressed(out/'domain.npz', evaluation=e, weights=w, original_ids=data['original_grid_flat_ids'],
                        scale_c=scale, baseline=art['monthly_climatology_c']+art['mean_c'])
    failures, inner_rows, point_rows, interval_rows, folds, responses = [], [], [], [], [], []
    work = 0

    def checkpoint():
        if time.perf_counter()-start > limits['wall_limit_seconds']:
            raise TimeoutError('pre-registered wall-time limit reached')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > limits['disk_limit_mib']*1024**2:
            raise TimeoutError('pre-registered disk limit reached')

    def loss(pred, truth):
        return float(np.sqrt(np.average(((pred-truth)[e]*scale)**2, weights=w[e])))

    def prior_for(train_dates):
        ix = [int(np.flatnonzero(dates == date)[0]) for date in train_dates]
        return learn_prior(hf[ix], lf[ix], train_dates, train_dates, basis, w)

    def case(date, seed, scenario, budget):
        d = designs[(date, seed)]
        i = d['i']
        ids = np.asarray(d['ids'][:budget], int)
        if len(ids) != budget or len(set(ids)) != budget or set(ids) & set(e):
            raise ValueError('insufficient budget or scoring node sampled')
        y = hf[i, ids] if scenario == 'clean' else np.asarray(d['y'])[ids]
        noise, _ = perturbation(hf[i].shape, seed, date, scenario, stream=1)
        future = hf[i] + noise/scale
        return i, ids, y, future

    def fit(prior, multiplier, low, y, ids, key, remove=False):
        nonlocal work
        work += 1
        try:
            return CLBR(prior, multiplier, scale, config['noise_std_bounds_c']).fit(low, y, ids, remove_lf=remove)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            failures.append({'key': key, 'error': repr(exc)})
            write(out/'failures.json', failures)
            return None

    try:
      for fold in plan['outer_folds']:
        allowed = [d for j, block in enumerate(blocks) if j != fold for d in block]
        records = []
        manifest = {'fold': fold, 'outer_evaluation': blocks[fold], 'outer_training': allowed, 'inner': []}
        for inner_fold, inner_eval in enumerate(blocks):
            if inner_fold == fold:
                continue
            train_dates = [d for d in allowed if d not in inner_eval]
            prior = prior_for(train_dates)
            manifest['inner'].append({'evaluation': inner_eval, 'training': train_dates, 'prior': prior.metadata()})
            for date in inner_eval:
              for scenario in plan['scenarios']:
               for seed in plan['seeds']:
                for budget in plan['budgets']:
                    checkpoint()
                    i, ids, y, future = case(date, seed, scenario, budget)
                    key = f'{fold}_{date}_{scenario}_{seed}_{budget}'
                    predictions, variances, noises, valid = [], [], [], []
                    for j, multiplier in enumerate(multipliers):
                        model = fit(prior, multiplier, lf[i], y, ids, 'inner_'+key+'_'+str(j))
                        ok = model is not None
                        predictions.append(model.mean_ if ok else np.full(len(w), np.nan))
                        variances.append(model.variance_ if ok else np.full(len(w), np.nan))
                        noises.append(model.noise_variance_ if ok else np.nan)
                        valid.append(ok)
                        inner_rows.append({'fold': fold, 'date': date, 'scenario': scenario, 'seed': seed, 'budget': budget,
                                           'family': 'clbr', 'candidate': j, 'success': ok,
                                           'rmse_c': loss(model.mean_, hf[i]) if ok else None})
                    ridge_predictions = []
                    for j, (rank, penalty) in enumerate(grid):
                        b = art['basis'][:, :rank]
                        pred = b @ ridge(b[ids], y, penalty)
                        ridge_predictions.append(pred)
                        work += 1
                        inner_rows.append({'fold': fold, 'date': date, 'scenario': scenario, 'seed': seed, 'budget': budget,
                                           'family': 'ridge', 'candidate': j, 'success': True, 'rmse_c': loss(pred, hf[i])})
                    record = {'date': date, 'scenario': scenario, 'seed': seed, 'budget': budget,
                              'clbr_mean': np.asarray(predictions), 'clbr_var': np.asarray(variances),
                              'noise_var': np.asarray(noises), 'ridge_mean': np.asarray(ridge_predictions),
                              'truth': hf[i], 'future': future, 'ids': ids, 'observations': y,
                              'training_dates': np.asarray(train_dates)}
                    np.savez_compressed(out/'inner'/f'{key}.npz', **record)
                    records.append(record)
        choices = {}
        for family, count in [('clbr', len(multipliers)), ('ridge', len(grid))]:
            scores = []
            for j in range(count):
                rows = [r for r in inner_rows if r['fold'] == fold and r['family'] == family and r['candidate'] == j]
                score = float(np.mean([r['rmse_c'] for r in rows])) if all(r['success'] for r in rows) else None
                scores.append(score)
            if all(s is None for s in scores):
                raise RuntimeError('all finite candidates failed: '+family)
            choices[family] = min(range(count), key=lambda j: float('inf') if scores[j] is None else scores[j])
            manifest[family+'_candidate_scores'] = scores
        manifest['choices'] = choices
        quantiles = {}
        j, rj = choices['clbr'], choices['ridge']
        for scenario in plan['scenarios']:
          for budget in plan['budgets']:
            group = [r for r in records if r['scenario'] == scenario and r['budget'] == budget]
            for target in ('field', 'future'):
                clbr_scores, ridge_scores = [], []
                for record in group:
                    truth = record['truth'] if target == 'field' else record['future']
                    sd = np.sqrt(record['clbr_var'][j]+(record['noise_var'][j] if target == 'future' else 0))
                    clbr_scores.extend(abs(truth[e]-record['clbr_mean'][j, e])/np.maximum(sd[e], 1e-12))
                    ridge_scores.extend(abs(truth[e]-record['ridge_mean'][rj, e]))
                weight = np.tile(w[e]/w[e].sum()/len(group), len(group))
                quantiles[f'{scenario}_{budget}_{target}'] = {
                    'clbr': weighted_quantiles(clbr_scores, weight), 'ridge': weighted_quantiles(ridge_scores, weight),
                    'calibration_dates': sorted(set(r['date'] for r in group))}
        prior = prior_for(allowed)
        manifest['outer_prior'] = prior.metadata()
        manifest['quantiles'] = quantiles
        folds.append(manifest)
        write(out/'fold_manifest.json', folds)
        for date in blocks[fold]:
          for scenario in plan['scenarios']:
           for seed in plan['seeds']:
            previous = {}
            for budget in plan['budgets']:
                checkpoint()
                i, ids, y, future = case(date, seed, scenario, budget)
                key = f'{fold}_{date}_{scenario}_{seed}_{budget}'
                model = fit(prior, multipliers[j], lf[i], y, ids, 'outer_'+key)
                ablated = fit(prior, multipliers[j], lf[i], y, ids, 'ablation_'+key, True)
                predictions = {'climatology': -art['mean_c']/scale,
                               'calibrated_lf': prior.intercept+prior.slope*lf[i]}
                diagnostics = {name: {'converged': True} for name in predictions}
                rank, penalty = grid[rj]
                b = art['basis'][:, :rank]
                predictions['ridge_nested'] = b @ ridge(b[ids], y, penalty)
                diagnostics['ridge_nested'] = {'converged': True}
                for name, m in [('clbr', model), ('clbr_no_lf', ablated)]:
                    predictions[name] = m.mean_ if m else np.full(len(w), np.nan)
                    diagnostics[name] = m.diagnostics_ if m else {'converged': False}
                predictions['residual_ridge'] = model.deterministic_prediction() if model else np.full(len(w), np.nan)
                diagnostics['residual_ridge'] = diagnostics['clbr']
                if model and not np.allclose(predictions['residual_ridge'][e], model.mean_[e], rtol=0, atol=1e-10):
                    raise ValueError('deterministic/Bayesian point prediction reference failed')
                comp = selected['comparison']
                from dataclasses import replace
                for method in ('pod_ridge', 'pod_lasso', 'gaussian_sbl_vb', 'rbmf'):
                    name = method+'_frozen'
                    try:
                        pred, _, diag, _ = reconstruct(method, art['basis'][:, :comp['method_rank'][method]], lf[i], ids, y,
                            replace(cfg, **comp['method_inference'][method]), comp['regularization'][method])
                        predictions[name], diagnostics[name] = pred, diag
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        failures.append({'key': key+'_'+name, 'error': repr(exc)})
                        predictions[name], diagnostics[name] = np.full(len(w), np.nan), {'converged': False}
                    work += 1
                arrays = {'truth': hf[i], 'future': future, 'ids': ids, 'observations': y, 'low': lf[i],
                          'training_dates': np.asarray(allowed)}
                for name, pred in predictions.items():
                    ok = bool(diagnostics[name]['converged']) and np.isfinite(pred).all()
                    row = {'fold': fold, 'date': date, 'scenario': scenario, 'seed': seed, 'budget': budget,
                           'method': name, 'success': ok, 'rmse_c': loss(pred, hf[i]) if np.isfinite(pred).all() else None,
                           'mae_c': float(np.average(abs(pred[e]-hf[i, e])*scale, weights=w[e])) if np.isfinite(pred).all() else None}
                    if not ok:
                        failures.append({'key': key+'_'+name, 'error': 'nonconverged_or_nonfinite'})
                    arrays['mean_'+name] = pred
                    point_rows.append(row)
                    if name in ('clbr', 'clbr_no_lf') and ok:
                        m = model if name == 'clbr' else ablated
                        r = {**{k: row[k] for k in ('fold', 'date', 'scenario', 'seed', 'budget', 'method')}, **m.diagnostics_}
                        r['added_response_rms_c'] = loss(pred, previous[name]) if name in previous else None
                        empty = CLBR(prior, multipliers[j], scale, config['noise_std_bounds_c']).fit(lf[i], [], [], remove_lf=name=='clbr_no_lf')
                        r['observation_response_rms_c'] = loss(pred, empty.mean_)
                        if seed == plan['seeds'][0] and budget == max(plan['budgets']):
                            yp = y.copy(); yp[0] += .02/scale
                            pert = fit(prior, multipliers[j], lf[i], yp, ids, 'response_'+key+'_'+name, name=='clbr_no_lf')
                            r['plus_0p02C_response_rms_c'] = loss(pert.mean_, pred) if pert else None
                            arrays['perturbed_'+name] = pert.mean_ if pert else np.full(len(w), np.nan)
                        arrays['no_observation_'+name] = empty.mean_
                        responses.append(r)
                    previous[name] = pred
                if model:
                    arrays['clbr_var'] = model.variance_
                    arrays['coefficient_covariance'] = model.coefficient_covariance_
                    arrays['noise_var'] = np.asarray(model.noise_variance_)
                for target in ('field', 'future'):
                    truth = hf[i] if target == 'field' else future
                    quant = quantiles[f'{scenario}_{budget}_{target}']
                    entries = []
                    if model:
                        for name, factor in [('clbr_conditional', None), ('clbr_empirical', quant['clbr'])]:
                            lo, hi = model.intervals(target=='future', factor)
                            entries.append((name, model.mean_, lo, hi))
                    center = predictions['ridge_nested']
                    lo, hi = center[:, None]-quant['ridge'], center[:, None]+quant['ridge']
                    entries.append(('ridge_constant', center, lo, hi))
                    for name, center, lo, hi in entries:
                        stats = interval_metrics(truth[e]*scale, center[e]*scale, lo[e]*scale, hi[e]*scale, w[e])
                        for level_idx, level in enumerate(LEVELS):
                            interval_rows.append({'fold': fold, 'date': date, 'scenario': scenario, 'seed': seed, 'budget': budget,
                                'target': target, 'method': name, 'level': float(level), 'wis_c': stats['wis'],
                                **{k: float(stats[k][level_idx]) for k in ('coverage', 'coverage_unweighted', 'width', 'interval_score')}})
                        arrays[f'{target}_{name}_lower'] = lo
                        arrays[f'{target}_{name}_upper'] = hi
                np.savez_compressed(out/'outer'/f'{key}.npz', **arrays)
        print('outer fold finished:', fold, flush=True)
      write(out/'status.json', {'completed': True, 'failure_count': len(failures)})
    except Exception as exc:
      failures.append({'key': 'batch', 'error': repr(exc)})
      write(out/'status.json', {'completed': False, 'error': repr(exc)})
      raise
    finally:
      for name, rows in [('inner_metrics', inner_rows), ('point_metrics', point_rows), ('interval_metrics', interval_rows)]:
        if rows:
            with (out/(name+'.csv')).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
      write(out/'failures.json', failures)
      write(out/'responses.json', responses)
      resources = {'wall_seconds': time.perf_counter()-start, 'peak_working_set_bytes': memory(),
                   'work_units': work, 'point_rows': len(point_rows), 'interval_rows': len(interval_rows),
                   'output_bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file()), 'single_thread': True}
      if args.phase == 'pilot':
        # Full/pilot workload ratios: outer folds4, seeds3, budgets5/2; conservative factor2.
        resources['full_estimate'] = {'work_multiplier': 30, 'conservative_wall_seconds': resources['wall_seconds']*60,
            'conservative_disk_bytes': resources['output_bytes']*60,
            'conservative_peak_bytes': resources['peak_working_set_bytes']*3 if resources['peak_working_set_bytes'] else None,
            'status': 'estimate_only_user_confirmation_required'}
      write(out/'resources.json', resources)
      print(json.dumps(resources, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['pilot', 'full'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tests', required=True)
    p.add_argument('--approval')
    run(p.parse_args())
