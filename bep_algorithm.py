import numpy as np
from collections import Counter


# ── Group Sort Order: Mean Value ──────────────────────────────────────────────

def compute_group_sort_order(group_reps, group_masks, noisy_snapshots, sort_gene_idx):
    n_groups = len(group_reps)
    group_means = np.zeros(n_groups)

    for gp_idx in range(n_groups):
        mask = group_masks[gp_idx]
        if mask.any():
            group_means[gp_idx] = float(np.mean(noisy_snapshots[mask, sort_gene_idx]))
        else:
            group_means[gp_idx] = float(group_reps[gp_idx, sort_gene_idx])

    return np.argsort(group_means)


# ── Bilateral Placement ───────────────────────────────────────────────────────

def bilateral_placement(shuffled_data, sort_gene_idx, group_info,
                        precomputed_order=None):
    n_samples, n_genes = shuffled_data.shape
    if n_samples < 2:
        return [], [], list(range(n_samples)), list(range(n_samples)), np.arange(n_samples)

    sorted_order = precomputed_order if precomputed_order is not None \
                   else np.argsort(shuffled_data[:, sort_gene_idx])

    sd   = shuffled_data[sorted_order]
    last = n_samples - 1

    gene_max_row = [None] * n_genes
    gene_min_row = [None] * n_genes
    for r in range(n_samples):
        orig_idx = int(sorted_order[r])
        gene_idx, gtype = group_info[orig_idx]
        if gtype == 'max':
            gene_max_row[gene_idx] = r
        else:
            gene_min_row[gene_idx] = r

    for g in range(n_genes):
        if gene_max_row[g] is None:
            gene_max_row[g] = int(np.argmax(sd[:, g]))
        if gene_min_row[g] is None:
            gene_min_row[g] = int(np.argmin(sd[:, g]))

    def monotonic_ok(val_before, val_after, mono):
        diff = val_after - val_before
        for g in range(n_genes):
            if mono[g] > 0 and diff[g] < 0:
                return False
            if mono[g] < 0 and diff[g] > 0:
                return False
        return True

    def flip_after_placement(mono, row):
        new_mono = mono.copy()
        for g in range(n_genes):
            if row == gene_max_row[g]:
                new_mono[g] = -1.0
            elif row == gene_min_row[g]:
                new_mono[g] =  1.0
        return new_mono

    diff_01 = sd[1] - sd[0]
    init_r  = np.sign(diff_01).astype(float)
    init_r[init_r == 0] = 1.0
    init_l  = -init_r.copy()
    init_l[init_l == 0] = -1.0

    base_mono_right = flip_after_placement(init_r, 0)
    mono_right      = flip_after_placement(base_mono_right, 1)
    base_mono_left  = flip_after_placement(init_l, 0)
    mono_left       = base_mono_left.copy()

    right_arm = [1]
    left_arm  = []
    discarded = []

    first_extremum_side = {}
    for g in range(n_genes):
        if gene_max_row[g] == 1 or gene_min_row[g] == 1:
            first_extremum_side[g] = 'right'

    def place_right(row):
        right_arm.append(row)
        orig = int(sorted_order[row])
        g, _ = group_info[orig]
        if g not in first_extremum_side:
            first_extremum_side[g] = 'right'
        return flip_after_placement(mono_right, row)

    def place_left(row):
        left_arm.append(row)
        orig = int(sorted_order[row])
        g, _ = group_info[orig]
        if g not in first_extremum_side:
            first_extremum_side[g] = 'left'
        return flip_after_placement(mono_left, row)

    for row in range(2, last):
        tip_r = sd[right_arm[-1]] if right_arm else sd[0]
        tip_l = sd[left_arm[-1]]  if left_arm  else sd[0]

        ok_r = monotonic_ok(tip_r, sd[row], mono_right)
        ok_l = monotonic_ok(tip_l, sd[row], mono_left)

        orig_idx = int(sorted_order[row])
        gene_idx, _ = group_info[orig_idx]

        if ok_r and not ok_l:
            mono_right = place_right(row)
        elif ok_l and not ok_r:
            mono_left = place_left(row)
        elif ok_r and ok_l:
            if gene_idx in first_extremum_side:
                if first_extremum_side[gene_idx] == 'right':
                    mono_left = place_left(row)
                else:
                    mono_right = place_right(row)
            else:
                mono_right = place_right(row)
        else:
            discarded.append(row)

    right_arm.append(last)
    left_arm.append(last)

    to_orig = lambda rows: [int(sorted_order[r]) for r in rows]

    center_orig    = int(sorted_order[0])
    right_orig     = to_orig(right_arm)
    left_orig      = to_orig(left_arm)
    discarded_orig = to_orig(discarded)

    full_cycle_orig = (to_orig(list(reversed(left_arm)))
                       + [center_orig]
                       + right_orig)

    return left_orig, right_orig, discarded_orig, full_cycle_orig, sorted_order


# ── Incremental Top/Bottom-k Placement ───────────────────────────────────────

def extract_top_bottom_k_groups(snapshots, k):
    n_samples, n_genes = snapshots.shape
    group_reps  = []
    group_info  = []
    group_masks = []

    for g in range(n_genes):
        vals       = snapshots[:, g]
        sorted_idx = np.argsort(vals)

        min_mask = np.zeros(n_samples, dtype=bool)
        max_mask = np.zeros(n_samples, dtype=bool)
        min_mask[sorted_idx[:k]]  = True
        max_mask[sorted_idx[-k:]] = True

        group_reps.append(np.mean(snapshots[min_mask], axis=0))
        group_reps.append(np.mean(snapshots[max_mask], axis=0))
        group_info.append((g, 'min'))
        group_info.append((g, 'max'))
        group_masks.append(min_mask)
        group_masks.append(max_mask)

    return np.array(group_reps), group_info, group_masks


def run_incremental_placement(noisy_snapshots, sort_gene_idx, k_max):
    results = []

    for k in range(1, k_max + 1):
        group_reps, group_info, group_masks = extract_top_bottom_k_groups(
            noisy_snapshots, k)

        group_sort_order = compute_group_sort_order(
            group_reps, group_masks, noisy_snapshots, sort_gene_idx)

        left_orig, right_orig, discarded_orig, full_cycle_orig, sorted_order = \
            bilateral_placement(group_reps, sort_gene_idx, group_info,
                                precomputed_order=group_sort_order)

        topology = tuple(group_info[i] for i in full_cycle_orig)

        results.append({
            'k':               k,
            'topology':        topology,
            'full_cycle_orig': full_cycle_orig,
            'left_orig':       left_orig,
            'right_orig':      right_orig,
            'discarded_orig':  discarded_orig,
            'group_reps':      group_reps,
            'group_info':      group_info,
            'group_masks':     group_masks,
            'sorted_order':    sorted_order,
        })

    return results


# ── Consensus Voting over (k, r) ─────────────────────────────────────────────

def canonicalize_topology(topo):
    seq = list(topo); best = None
    for reflect in (False, True):
        s = seq[::-1] if reflect else seq
        for shift in range(len(s)):
            rotated = tuple(s[shift:] + s[:shift])
            if best is None or rotated < best: best = rotated
    return best


def run_bep_consensus(snapshots, n_genes, k_max):
    votes = Counter()
    n_total = n_genes * k_max
    for k in range(1, k_max + 1):
        group_reps, group_info, group_masks = extract_top_bottom_k_groups(snapshots, k)
        for r in range(n_genes):
            sort_order = compute_group_sort_order(group_reps, group_masks, snapshots, r)
            _, _, discarded, full_cycle, _ = bilateral_placement(
                group_reps, r, group_info, precomputed_order=sort_order)
            if len(discarded) == 0 and len(full_cycle) == 2 * n_genes + 1:
                topo = tuple(group_info[i] for i in full_cycle[:-1])
                votes[canonicalize_topology(topo)] += 1
    n_valid = sum(votes.values())
    if n_valid == 0:
        return None, 0.0, 0.0
    best_canon, count = votes.most_common(1)[0]
    return best_canon, count / n_valid, n_valid / n_total


# ── Regulatory Sign Inference and Mirror Result ──────────────────────────────

def infer_network_from_topology(topo):
    n = len(topo)
    edges = set()
    for p in range(n):
        (a, ta), (b, tb) = topo[p], topo[(p + 1) % n]
        edges.add((a, b, '+' if ta == tb else '-'))
    return sorted(edges)


def recover_network_structure(snapshots, n_genes, k_max):
    topo, margin, frac_valid = run_bep_consensus(snapshots, n_genes, k_max)
    if topo is None:
        return None
    mirror = topo[::-1]
    return {
        'topology':        topo,
        'mirror_topology': mirror,
        'edges':           infer_network_from_topology(topo),
        'mirror_edges':    infer_network_from_topology(mirror),
        'consensus_margin': margin,
        'frac_valid_runs':  frac_valid,
    }
