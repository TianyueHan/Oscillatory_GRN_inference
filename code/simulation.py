import numpy as np


# ── Network Generation (Section IV-A) ─────────────────────────────────────────

def draw_loop_signs(G, rng):
    """Edge signs of a G-gene ring; signs[p] is the sign of the edge from gene
    p-1 to gene p. Each edge is activating or repressing with equal probability,
    conditioned on an odd number of repressing edges."""
    rep = rng.integers(0, 2, size=G).astype(bool)
    rep[-1] = (rep[:-1].sum() % 2 == 0)
    return ['-' if r else '+' for r in rep]


def true_cyclic_order(signs):
    """Ground-truth cyclic order of the 2G extrema of a loop, in the same
    (gene, 'max'/'min') format as bep_algorithm. Gene indices are positions
    within the loop."""
    G = len(signs)
    order, pos, kind = [], 0, 'max'
    for _ in range(2 * G):
        order.append((pos, kind))
        nxt = (pos + 1) % G
        if signs[nxt] == '-':
            kind = 'min' if kind == 'max' else 'max'
        pos = nxt
    return tuple(order)


def generate_network(rng, gene_param_sampler, loop_count_options, loop_size_options,
                     branch_length_options, background_count_options):
    """gene_param_sampler(rng) must return a dict with keys c, gamma, beta, n, K, tau.
    The number of loops, the size of each loop, the branch length below each loop gene
    and the number of background genes are each drawn uniformly from the given options."""
    genes = []

    def add_gene(role, regulator, sign):
        genes.append(dict(gene_param_sampler(rng), role=role, regulator=regulator, sign=sign))
        return len(genes) - 1

    loops = []
    for _ in range(int(rng.choice(loop_count_options))):
        G = int(rng.choice(loop_size_options))
        signs = draw_loop_signs(G, rng)

        loop_genes = [add_gene('loop', None, None) for _ in range(G)]
        for p, g in enumerate(loop_genes):
            genes[g]['regulator'] = loop_genes[p - 1]         # gene j is regulated by gene j-1
            genes[g]['sign'] = signs[p]

        branch_genes = []
        for g in loop_genes:
            parent = g
            for _ in range(int(rng.choice(branch_length_options))):
                parent = add_gene('branch', parent, '-' if rng.integers(0, 2) else '+')
                branch_genes.append(parent)

        loops.append({
            'genes':        loop_genes,
            'signs':        signs,
            'branch_genes': branch_genes,
            'edges':        sorted(((p - 1) % G, p, signs[p]) for p in range(G)),
            'cyclic_order': true_cyclic_order(signs),
        })

    background_genes = [add_gene('background', None, None)
                        for _ in range(int(rng.choice(background_count_options)))]

    return {'genes': genes, 'loops': loops, 'background_genes': background_genes}


# ── Dynamics with Intrinsic Noise (Eqs. 26-29) ───────────────────────────────

def simulate_network(genes, x0, dt, n_steps, sigma_int, q, rng):
    """Euler integration of Eq. (28) with temporally correlated noise (Eq. 29).
    The history before t = 0 is taken to be x0."""
    n_genes = len(genes)
    regulated = np.array([g['regulator'] is not None for g in genes])

    c     = np.array([g['c']     for g in genes], dtype=float)
    gamma = np.array([g['gamma'] for g in genes], dtype=float)
    beta  = np.array([g['beta'] if r else 0.0 for g, r in zip(genes, regulated)], dtype=float)
    n_hill = np.array([g['n']    for g in genes], dtype=float)
    K     = np.array([g['K']     for g in genes], dtype=float)
    reg   = np.array([g['regulator'] if r else i
                      for i, (g, r) in enumerate(zip(genes, regulated))])
    delay = np.array([int(round(g['tau'] / dt)) if r else 0
                      for g, r in zip(genes, regulated)])
    is_act = np.array([g['sign'] == '+' for g in genes])

    # eps[n] = (w[n] + ... + w[n+q-1]) / sqrt(q)
    w = rng.standard_normal((n_steps + q - 1, n_genes))
    cs = np.vstack([np.zeros((1, n_genes)), np.cumsum(w, axis=0)])
    eps = (cs[q:q + n_steps] - cs[:n_steps]) / np.sqrt(q)

    traj = np.empty((n_steps + 1, n_genes))
    traj[0] = x0
    for t in range(n_steps):
        x  = traj[t]
        xd = traj[np.maximum(t - delay, 0), reg]
        act = xd ** n_hill / (K ** n_hill + xd ** n_hill)
        h = np.where(is_act, act, 1.0 - act)
        F = c - gamma * x + beta * h
        traj[t + 1] = np.maximum(x + F * (dt + sigma_int * np.sqrt(dt) * eps[t]), 0.0)

    return traj


# ── Acceptance Check ─────────────────────────────────────────────────────────

def is_stationary_oscillation(traj, loop_gene_cols, burn_steps, min_relative_amplitude, stationarity_tol):
    """Every loop gene must have a relative amplitude (max - min) / mean of at least
    min_relative_amplitude in the window after burn_steps, and the peak-to-peak
    amplitudes in the first and second half of the window must differ by less than
    stationarity_tol (relative)."""
    win = traj[burn_steps:, loop_gene_cols]
    half = win.shape[0] // 2

    mean  = win.mean(axis=0)
    amp   = np.ptp(win, axis=0)
    amp_1 = np.ptp(win[:half], axis=0)
    amp_2 = np.ptp(win[half:], axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_amp  = amp / mean
        rel_diff = np.abs(amp_1 - amp_2) / np.maximum(amp_1, amp_2)

    if not np.all(rel_amp >= min_relative_amplitude):
        return False
    return bool(np.all(rel_diff < stationarity_tol))


# ── Snapshot Sampling with Observation Noise ─────────────────────────────────

def sample_snapshots(traj, burn_steps, dt, n_samples, sigma_obs, rng):
    steps = rng.choice(np.arange(burn_steps, traj.shape[0]), size=n_samples, replace=False)
    snapshots = traj[steps]

    X = snapshots * (1.0 + sigma_obs * rng.standard_normal(snapshots.shape))
    X = np.maximum(X, 0.0)

    perm = rng.permutation(n_samples)
    return X[perm], steps[perm] * dt


# ── One Simulation Trial ─────────────────────────────────────────────────────

def simulate_trial(rng, gene_param_sampler, init_sampler,
                   loop_count_options, loop_size_options,
                   branch_length_options, background_count_options,
                   dt, n_steps, burn_steps, n_samples, sigma_int, sigma_obs, q,
                   min_relative_amplitude, stationarity_tol, max_attempts):
    """Draws networks until one is accepted, then returns the shuffled snapshot
    matrix X (cells x genes) together with the ground truth.
    init_sampler(rng, n_genes) must return the initial expression vector.
    sample_times and truth are for evaluation only and must not be given to the pipeline.
    Loop gene indices in truth['loops'][l]['genes'] are columns of X; 'edges' and
    'cyclic_order' use positions within the loop."""
    for _ in range(max_attempts):
        network = generate_network(rng, gene_param_sampler, loop_count_options, loop_size_options,
                                   branch_length_options, background_count_options)
        genes = network['genes']

        x0 = init_sampler(rng, len(genes))
        traj = simulate_network(genes, x0, dt, n_steps, sigma_int, q, rng)

        loop_cols = [g for loop in network['loops'] for g in loop['genes']]
        if is_stationary_oscillation(traj, loop_cols, burn_steps, min_relative_amplitude, stationarity_tol):
            break
    else:
        raise RuntimeError("no network was accepted within max_attempts")

    X, sample_times = sample_snapshots(traj, burn_steps, dt, n_samples, sigma_obs, rng)
    return {'X': X, 'sample_times': sample_times, 'truth': network}
