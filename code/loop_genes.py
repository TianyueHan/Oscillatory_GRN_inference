import numpy as np
from itertools import combinations

from co_oscillating_genes import find_coscillating_groups


# ── Pairwise Dispersion (Eqs. 24-25) ──────────────────────────────────────────

def two_means_1d(values):
    """Exact 2-means in 1-D. Returns a boolean mask of the upper cluster."""
    n = len(values)
    order = np.argsort(values)
    v = values[order]
    cs, cs2 = np.cumsum(v), np.cumsum(v ** 2)

    s = np.arange(1, n)
    left_cost  = cs2[:-1] - cs[:-1] ** 2 / s
    right_cost = (cs2[-1] - cs2[:-1]) - (cs[-1] - cs[:-1]) ** 2 / (n - s)
    split = int(np.argmin(left_cost + right_cost)) + 1

    upper = np.zeros(n, dtype=bool)
    upper[order[split:]] = True
    return upper


def directional_dispersion(xj, xjp, n_bins, n_edge_bins, min_bin_size):
    """V_j(j') of Eq. (24): bin by gene j, split gene j' in each bin into an
    upper and a lower cluster, and average the two cluster variances."""
    edges = np.linspace(xj.min(), xj.max(), n_bins + 1)
    bin_idx = np.digitize(xj, edges[1:-1])

    variances = []
    for b in range(n_edge_bins, n_bins - n_edge_bins):
        vals = xjp[bin_idx == b]
        if len(vals) < min_bin_size:
            continue
        upper = two_means_1d(vals)
        variances.append((vals[upper].var() + vals[~upper].var()) / 2)

    return float(np.mean(variances)) if variances else np.nan


def pair_dispersion(X, j, jp, **kw):
    """V_{jj'} of Eq. (25)."""
    range_j  = X[:, j].max()  - X[:, j].min()
    range_jp = X[:, jp].max() - X[:, jp].min()
    v_j_jp = directional_dispersion(X[:, j],  X[:, jp], **kw)
    v_jp_j = directional_dispersion(X[:, jp], X[:, j],  **kw)
    return v_j_jp / range_jp + v_jp_j / range_j


# ── Loop Gene Identification (Section III-B) ──────────────────────────────────

def find_dispersion_gap(values, kappa, eps=1e-12):
    """Returns V_(u*) if max_u V_(u+1)/V_(u) exceeds kappa, otherwise None."""
    v = np.sort(values[np.isfinite(values)])
    if len(v) < 2:
        return None
    ratios = v[1:] / np.maximum(v[:-1], eps)
    u = int(np.argmax(ratios))
    return float(v[u]) if ratios[u] > kappa else None


def maximum_clique(adj, V=None):
    """Maximum clique by Bron-Kerbosch with pivoting. Ties are broken by the
    smallest mean pair dispersion when V is given."""
    n = adj.shape[0]
    nbrs = [set(int(v) for v in np.flatnonzero(adj[i])) for i in range(n)]
    cliques = []

    def expand(R, P, Xs):
        if not P and not Xs:
            cliques.append(R)
            return
        pivot = max(P | Xs, key=lambda v: len(P & nbrs[v]))
        for v in list(P - nbrs[pivot]):
            expand(R | {v}, P & nbrs[v], Xs & nbrs[v])
            P = P - {v}
            Xs = Xs | {v}

    expand(set(), set(range(n)), set())

    max_size = max(len(c) for c in cliques)
    best = [sorted(c) for c in cliques if len(c) == max_size]
    if V is None:
        return min(best)
    return min(best, key=lambda c: np.mean([V[a, b] for a, b in combinations(c, 2)]))


def identify_loop_genes(X, group, kappa, **dispersion_kw):
    group = list(group)
    m = len(group)

    V = np.full((m, m), np.nan)
    for a, b in combinations(range(m), 2):
        V[a, b] = V[b, a] = pair_dispersion(X, group[a], group[b], **dispersion_kw)

    threshold = find_dispersion_gap(V[np.triu_indices(m, 1)], kappa)
    if threshold is None:
        return group

    with np.errstate(invalid='ignore'):
        adj = V <= threshold
    clique = maximum_clique(adj, V)
    return [group[i] for i in clique]


# ── Full Preprocessing: Groups -> Loop Genes ─────────────────────────────────

def identify_oscillatory_loops(X, rho, theta, kappa, **dispersion_kw):
    """Returns one list of loop gene indices (columns of X) per co-oscillating group.
    dispersion_kw must contain n_bins, n_edge_bins and min_bin_size.
    Each list is the gene set passed to BEP, e.g.
        run_bep_consensus(X[:, loop], len(loop), k_max)
    Loops with only two genes do not need BEP."""
    groups = find_coscillating_groups(X, rho, theta)
    return [identify_loop_genes(X, g, kappa, **dispersion_kw) for g in groups]
