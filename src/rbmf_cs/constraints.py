"""Hard deployment feasibility in physical units, Section 5.5.

Energy is the TOTAL forecast over one common mission horizon, not a sum of
successive total forecasts. Each source sends its configured packets along a
minimum-transmit-energy path to a gateway (or the smallest sensing ID as sink).
Relays pay transmit energy for every forwarded packet; receive energy is not
modeled. All routes and existing-node loads are recomputed for prospective sets.
"""
from dataclasses import dataclass
import heapq
import numpy as np


@dataclass(frozen=True)
class ConstraintConfig:
    ice_threshold: float = 0.6
    max_sensors: int = 15
    energy_budget_j: float = 10000.
    communication_radius_km: float = 25.
    coordinate_system: str = 'cartesian_km'
    earth_radius_km: float = 6371.0088
    mission_seconds: float = 3600.
    sense_power_w: float = 0.01
    process_power_w: float = 0.005
    packets_per_sensor: int = 12
    tx_electronics_j_per_packet: float = 0.2
    tx_amplifier_j_per_packet_km2: float = 0.001
    gateway_energy_j: float = 10.

    def __post_init__(self):
        if not np.isfinite(self.ice_threshold) or not 0 <= self.ice_threshold <= 1:
            raise ValueError('ice_threshold must be a fraction in [0,1]')
        for name in ('max_sensors', 'packets_per_sensor'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('energy_budget_j', 'sense_power_w', 'process_power_w',
                     'tx_amplifier_j_per_packet_km2', 'gateway_energy_j'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('communication_radius_km', 'earth_radius_km', 'mission_seconds',
                     'tx_electronics_j_per_packet'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.coordinate_system not in ('cartesian_km', 'latlon_degrees'):
            raise ValueError('declare cartesian_km or latlon_degrees')


def distances_km(coordinates, system='cartesian_km', earth_radius_km=6371.0088):
    """Pairwise physical distances; lat/lon input order is [latitude, longitude]."""
    xy = np.asarray(coordinates, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('coordinates must be finite (n,2)')
    if system == 'cartesian_km':
        return np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    if system != 'latlon_degrees':
        raise ValueError('unknown coordinate system')
    if np.any(abs(xy[:, 0]) > 90) or np.any(abs(xy[:, 1]) > 180):
        raise ValueError('latitude/longitude must be within +/-90, +/-180 degrees')
    if not np.isfinite(earth_radius_km) or earth_radius_km <= 0:
        raise ValueError('invalid earth radius')
    lat, lon = np.deg2rad(xy).T
    a = np.sin((lat[:, None]-lat[None, :])/2)**2
    a += np.cos(lat[:, None])*np.cos(lat[None, :])*np.sin((lon[:, None]-lon[None, :])/2)**2
    return 2*earth_radius_km*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def node_ids(values, size):
    ids = np.asarray(values)
    if ids.size == 0:
        ids = np.array([], dtype=int)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= size):
        raise ValueError('invalid node IDs')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate node IDs')
    return ids.astype(int).tolist()


def connected(adjacency):
    if len(adjacency) == 0:
        return False
    seen, pending = {0}, [0]
    while pending:
        u = pending.pop()
        for v in np.flatnonzero(adjacency[u]):
            if int(v) not in seen:
                seen.add(int(v)); pending.append(int(v))
    return len(seen) == len(adjacency)


class DeploymentConstraints:
    """Gateway vertices do not count as sensors; their fixed energy DOES count.

    Anchors are ordinary sensing IDs that must be included in every deployment.
    Environmental ice can be refreshed through set_ice before each decision.
    """
    def __init__(self, coordinates, ice, config=None, *, gateways=None, anchors=()):
        self.config = config or ConstraintConfig()
        self.coordinates = np.array(coordinates, float, copy=True)
        # Validate without constructing an N by N candidate-grid distance matrix.
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError('coordinates must be (N,2)')
        self.n = len(self.coordinates)
        self.gateways = np.array([] if gateways is None else gateways, float).reshape(-1, 2)
        all_xy = np.vstack((self.coordinates, self.gateways))
        if not np.isfinite(all_xy).all():
            raise ValueError('coordinates must be finite')
        if self.config.coordinate_system == 'latlon_degrees':
            if np.any(abs(all_xy[:, 0]) > 90) or np.any(abs(all_xy[:, 1]) > 180):
                raise ValueError('invalid latitude/longitude')
        self.anchors = node_ids(anchors, self.n)
        self.set_ice(ice)

    def set_ice(self, ice):
        ice = np.array(ice, float, copy=True)
        if ice.shape != (self.n,):
            raise ValueError('ice must align with candidate grid')
        self.ice = ice  # Unknown/out-of-range values exclude affected nodes.

    def evaluate(self, selected):
        ids = sorted(node_ids(selected, self.n))
        c = self.config
        p, g = len(ids), len(self.gateways)
        coordinates = np.vstack((self.coordinates[ids], self.gateways))
        distance = distances_km(coordinates, c.coordinate_system, c.earth_radius_km)
        adjacency = distance <= c.communication_radius_km
        np.fill_diagonal(adjacency, False)
        is_connected = connected(adjacency)
        reasons = []
        values = self.ice[ids]
        ice_known = bool(np.all(np.isfinite(values) & (values >= 0) & (values <= 1)))
        if not ice_known:
            reasons.append('ice_unknown_or_invalid')
        elif np.any(values > c.ice_threshold):
            reasons.append('ice_threshold')
        if p > c.max_sensors:
            reasons.append('sensor_budget')
        if not set(self.anchors).issubset(ids):
            reasons.append('missing_anchor')
        if not is_connected:
            reasons.append('connectivity')
        energy = self._mission_energy(ids, distance, adjacency) if is_connected else None
        if energy is None:
            reasons.append('energy_unavailable')
        elif energy['total_j'] > c.energy_budget_j:
            reasons.append('energy_budget')
        labels = [f'sensor:{j}' for j in ids] + [f'gateway:{j}' for j in range(g)]
        edges = [[labels[u], labels[v], float(distance[u, v])]
                 for u in range(p+g) for v in range(u+1, p+g) if adjacency[u, v]]
        return {'feasible': not reasons, 'reasons': reasons, 'selected': ids,
                'sensor_count': p, 'gateway_count': g, 'connected': is_connected,
                'ice': {str(j): float(self.ice[j]) if np.isfinite(self.ice[j]) else None for j in ids},
                'edges_km': edges, 'energy': energy,
                'gateway_counts_as_sensor': False, 'gateway_energy_included': True}

    def _mission_energy(self, ids, distance, adjacency):
        c, p, g = self.config, len(ids), len(self.gateways)
        n = p+g
        roots = list(range(p, n)) if g else [0]
        weights = c.tx_electronics_j_per_packet + c.tx_amplifier_j_per_packet_km2*distance**2
        costs = np.full(n, np.inf)
        parent = np.full(n, -1, int)
        queue = []
        for root in roots:
            costs[root] = 0.; heapq.heappush(queue, (0., root))
        while queue:
            cost, u = heapq.heappop(queue)
            if cost != costs[u]:
                continue
            for v in np.flatnonzero(adjacency[u]):
                proposed = cost+weights[u, v]
                if proposed < costs[v]:
                    costs[v], parent[v] = proposed, u
                    heapq.heappush(queue, (proposed, int(v)))
        loads = np.zeros(p, int)
        routes = {}
        for source in range(p):
            u, route = source, [f'sensor:{ids[source]}']
            while parent[u] != -1:
                loads[u] += c.packets_per_sensor
                u = int(parent[u])
                route.append(f'sensor:{ids[u]}' if u < p else f'gateway:{u-p}')
            routes[str(ids[source])] = route
        nodes = {}
        for u, j in enumerate(ids):
            tx = float(loads[u]*weights[u, parent[u]]) if parent[u] != -1 else 0.
            sense, process = c.mission_seconds*c.sense_power_w, c.mission_seconds*c.process_power_w
            nodes[str(j)] = {'sense_j': sense, 'process_j': process, 'tx_j': tx,
                             'forwarded_and_own_packets': int(loads[u]), 'total_j': sense+process+tx}
        gateway = g*c.gateway_energy_j
        return {'total_j': float(sum(x['total_j'] for x in nodes.values())+gateway),
                'sensor_total_j': float(sum(x['total_j'] for x in nodes.values())),
                'gateway_total_j': gateway, 'per_sensor': nodes, 'routes': routes,
                'mission_seconds': c.mission_seconds, 'routing': 'minimum_transmit_energy',
                'sink_without_gateway': min(ids) if not g and ids else None}

    def filter_candidates(self, selected, candidates=None):
        selected = node_ids(selected, self.n)
        pool = range(self.n) if candidates is None else node_ids(candidates, self.n)
        reports = {int(j): self.evaluate(selected+[int(j)]) for j in sorted(pool) if j not in selected}
        return [j for j, report in reports.items() if report['feasible']], reports
