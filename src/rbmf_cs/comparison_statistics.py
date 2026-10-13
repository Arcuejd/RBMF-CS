"""Paired uncertainty summaries with explicit dependence and scope."""
import numpy as np


def holm(pvalues):
    """Holm adjusted p-values; caller must supply defensible environment tests."""
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('finite p-values in [0,1] required')
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def paired_intervals(error_a, error_b, weights, coordinates, seed=8128, replicates=2000):
    """Errors have shape (noise seed, chronological month, fixed evaluation node).

    Seed CI is conditional on fixed environments. A joint time/spatial block
    sensitivity interval is intentionally NOT a generalization confidence claim.
    Time blocks have two observed months (not two calendar months).
    """
    a, b = np.asarray(error_a, float), np.asarray(error_b, float)
    w = np.asarray(weights, float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != len(w):
        raise ValueError('paired errors require matching seed x month x node arrays')
    rmse_a = np.sqrt(np.average(a*a, axis=2, weights=w))
    rmse_b = np.sqrt(np.average(b*b, axis=2, weights=w))
    differences = rmse_a-rmse_b
    seed_diff = differences.mean(axis=1)
    rng = np.random.default_rng(seed)
    result = {'paired_mean_difference_c': float(differences.mean()),
              'noise_seeds': len(a), 'observed_months': a.shape[1],
              'seed_conditional_ci95_c': None, 'time_space_block_sensitivity95_c': None,
              'environment_p_value': None, 'holm_adjusted_p_value': None,
              'environment_inference': 'unavailable: no independent held-out years; seeds are not environments',
              'bootstrap_replicates': replicates, 'bootstrap_seed': seed}
    if len(a) >= 2:
        draws = seed_diff[rng.integers(0, len(a), (replicates, len(a)))].mean(axis=1)
        result['seed_conditional_ci95_c'] = np.quantile(draws, [.025, .975]).tolist()
    if a.shape[1] >= 2:
        blocks = np.floor(np.asarray(coordinates)/np.array([2., 5.])).astype(int)
        unique = np.unique(blocks, axis=0)
        members = [np.flatnonzero(np.all(blocks == key, axis=1)) for key in unique]
        boot = []
        for _ in range(replicates):
            starts = rng.integers(0, a.shape[1], (a.shape[1]+1)//2)
            times = np.ravel(np.c_[starts, (starts+1) % a.shape[1]])[:a.shape[1]]
            nodes = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
            # Seeds are averaged, not counted as independent environments.
            ra = np.sqrt(np.average(a[:, times][:, :, nodes]**2, weights=w[nodes], axis=2))
            rb = np.sqrt(np.average(b[:, times][:, :, nodes]**2, weights=w[nodes], axis=2))
            boot.append(float((ra-rb).mean()))
        result['time_space_block_sensitivity95_c'] = np.quantile(boot, [.025, .975]).tolist()
        result['spatial_blocks'] = len(members)
        result['warning'] = 'Few correlated months; block interval is exploratory sensitivity, not cross-year inference.'
    return result
