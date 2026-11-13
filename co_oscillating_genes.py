import math
import numpy as np


# ── Co-oscillating Gene Identification (Section III-A) ────────────────────────

def extremum_spread_matrix(X, k):
    """S[j, jp] = min([B_j(k)]_jp, [T_j(k)]_jp) / [N]_jp, the building block of Eq. (23)."""
    n_cells, n_genes = X.shape
    total_range = X.max(axis=0) - X.min(axis=0)
    order = np.argsort(X, axis=0)

    S = np.full((n_genes, n_genes), np.inf)
    for j in range(n_genes):
        bottom = X[order[:k, j]]
        top    = X[order[-k:, j]]
        range_b = bottom.max(axis=0) - bottom.min(axis=0)
        range_t = top.max(axis=0) - top.min(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.minimum(range_b, range_t) / total_range
        ratio[total_range == 0] = np.inf
        S[j] = ratio

    np.fill_diagonal(S, np.inf)
    return S


def coscillation_adjacency(X, rho, theta):
    n_cells = X.shape[0]
    k = int(math.ceil(rho * n_cells))
    S = extremum_spread_matrix(X, k)
    # Eq. (23): the minimum of the four ratios is min(S[j, jp], S[jp, j])
    return np.minimum(S, S.T) < theta


def connected_components(adj):
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    components = []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        stack, comp = [s], []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.flatnonzero(adj[u]):
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        components.append(sorted(comp))
    return components


def find_coscillating_groups(X, rho, theta):
    adj = coscillation_adjacency(X, rho, theta)
    return [c for c in connected_components(adj) if len(c) >= 2]
