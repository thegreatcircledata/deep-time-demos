#!/usr/bin/env python3
"""ep04 demonstration simulations (directive S8: illustrations, not research claims).

Every number this script writes is a claims.json entry with "type": "illustration" and is labelled on screen as a
demonstration. Fixed seeds; re-running reproduces demo_results.json byte for byte (the hash is printed).

Demo A, random dots make lines: N_SITES points uniform on a SIDE_M x SIDE_M square map. A "k-site line" is a straight
line drawn between two dots (its end points) that passes within TOL_M metres of at least k-2 other dots lying between
those ends. All C(N,2) end-point pairs are tried.
Demo B, the baseline: the same count on N_MAPS independent random maps gives the range that pure chance produces.
Demo C, how many things were tried: N_STUDIES simulated "studies", each running N_TESTS significance tests on pure
noise (z-test, two-sided, alpha 0.05); the share of studies that find at least one "significant" result.

Run with the project venv: .venv/bin/python episodes/ep04-coincidence/data/demo_alignments.py
"""
import hashlib, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261003                      # master seed (date the demo was specified); every stream derives from it
N_SITES = 300
SIDE_M = 100_000.0                   # 100 km x 100 km map
TOL_M = 50.0                         # within 50 m of the line: a strip 100 m wide
N_MAPS = 1000
N_STUDIES = 10_000
N_TESTS = 20
N_SAMPLE = 50
ALPHA = 0.05
CHUNK = 4096


def lines_on_map(pts):
    """For every end-point pair (i<j): the number of other dots strictly between the ends and within TOL_M of the line."""
    n = len(pts)
    ii, jj = np.triu_indices(n, k=1)
    out = np.empty(len(ii), dtype=np.int32)
    for s in range(0, len(ii), CHUNK):
        a = pts[ii[s:s + CHUNK]]                          # (m,2)
        b = pts[jj[s:s + CHUNK]]
        ab = b - a                                        # (m,2)
        L2 = np.einsum("ij,ij->i", ab, ab)                # (m,)
        ap = pts[None, :, :] - a[:, None, :]              # (m,n,2)
        t = (ap[:, :, 0] * ab[:, None, 0] + ap[:, :, 1] * ab[:, None, 1]) / L2[:, None]
        cross = np.abs(ab[:, None, 0] * ap[:, :, 1] - ab[:, None, 1] * ap[:, :, 0])
        dist = cross / np.sqrt(L2)[:, None]
        inside = (t > 0.0) & (t < 1.0) & (dist <= TOL_M)
        rows = np.arange(len(a))
        inside[rows, ii[s:s + CHUNK]] = False             # the end points themselves never count
        inside[rows, jj[s:s + CHUNK]] = False
        out[s:s + CHUNK] = inside.sum(axis=1)
    return ii, jj, out


def summarise(counts):
    return {f"lines_through_at_least_{k}_sites": int((counts >= k - 2).sum()) for k in (3, 4, 5, 6, 7)}


def pct(a, q):
    return float(np.percentile(a, q, method="lower"))


def main():
    ss = np.random.SeedSequence(SEED)
    map_seq, study_seq = ss.spawn(2)
    map_seeds = map_seq.spawn(N_MAPS)

    # ---- Demo A: the first map is the one drawn on screen ----------------------------------------------------------
    rng0 = np.random.default_rng(map_seeds[0])
    pts0 = rng0.uniform(0.0, SIDE_M, size=(N_SITES, 2))
    ii, jj, c0 = lines_on_map(pts0)
    lines = [{"a": int(a), "b": int(b), "between": int(k)} for a, b, k in zip(ii, jj, c0) if k >= 2]
    demo_map = {"points_m": [[round(float(x), 1), round(float(y), 1)] for x, y in pts0], "lines_4plus": lines}

    # ---- Demo B: baseline over N_MAPS random maps -------------------------------------------------------------------
    per_map = []
    for m, sd in enumerate(map_seeds):
        rng = np.random.default_rng(sd)
        pts = rng.uniform(0.0, SIDE_M, size=(N_SITES, 2))
        per_map.append(summarise(lines_on_map(pts)[2]))
        if (m + 1) % 100 == 0:
            print(f"  maps {m + 1}/{N_MAPS}", file=sys.stderr, flush=True)
    keys = list(per_map[0])
    base = {}
    for k in keys:
        v = np.array([d[k] for d in per_map])
        base[k] = {"min": int(v.min()), "p05": pct(v, 5), "median": pct(v, 50), "mean": round(float(v.mean()), 2),
                   "p95": pct(v, 95), "max": int(v.max()), "maps_with_at_least_one": int((v >= 1).sum())}

    # analytic check (expected count of 4+ and 5+ lines, uniform points; independent-binomial approximation)
    rngx = np.random.default_rng(map_seq.spawn(1)[0])
    A = rngx.uniform(0, SIDE_M, (200_000, 2)); B = rngx.uniform(0, SIDE_M, (200_000, 2))
    p = 2 * TOL_M * np.hypot(*(B - A).T) / SIDE_M ** 2
    npairs = N_SITES * (N_SITES - 1) // 2
    from math import comb
    def p_atleast(r):
        return 1 - sum(comb(N_SITES - 2, q) * p ** q * (1 - p) ** (N_SITES - 2 - q) for q in range(r))
    analytic = {"expected_lines_4plus": round(float(npairs * p_atleast(2).mean()), 1),
                "expected_lines_5plus": round(float(npairs * p_atleast(3).mean()), 1),
                "expected_lines_6plus": round(float(npairs * p_atleast(4).mean()), 2)}

    # ---- Demo C: how many things were tried -------------------------------------------------------------------------
    rngs = np.random.default_rng(study_seq)
    x = rngs.standard_normal((N_STUDIES, N_TESTS, N_SAMPLE))      # pure noise, true mean 0, sd 1
    z = x.mean(axis=2) * math.sqrt(N_SAMPLE)
    pvals = np.vectorize(lambda v: math.erfc(abs(v) / math.sqrt(2)))(z)   # two-sided normal p-value
    sig = pvals < ALPHA
    demo_c = {"n_studies": N_STUDIES, "tests_per_study": N_TESTS, "alpha": ALPHA,
              "share_of_single_tests_significant": round(float(sig.mean()), 4),
              "share_of_studies_with_at_least_one_significant": round(float(sig.any(axis=1).mean()), 4),
              "textbook_value_1_minus_0.95_pow_20": round(1 - (1 - ALPHA) ** N_TESTS, 4)}

    res = {
        "type": "illustration",
        "note": "Demonstration simulations for ep04 (directive S8). Not research claims; label on screen as a demonstration.",
        "seed": SEED, "numpy": np.__version__,
        "params": {"n_sites": N_SITES, "map_side_m": SIDE_M, "tolerance_m": TOL_M, "strip_width_m": 2 * TOL_M,
                   "n_maps": N_MAPS, "end_point_pairs_per_map": npairs,
                   "line_definition": "straight line between two dots (its end points) passing within tolerance_m of at "
                                      "least k-2 other dots that lie between those ends"},
        "demo_a_first_map": summarise(c0),
        "demo_b_baseline_over_maps": base,
        "analytic_check": analytic,
        "demo_c_how_many_tries": demo_c,
    }
    out = os.path.join(HERE, "demo_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True); f.write("\n")
    with open(os.path.join(HERE, "demo_map_first.json"), "w") as f:
        json.dump(demo_map, f, separators=(",", ":")); f.write("\n")
    for fn in ("demo_results.json", "demo_map_first.json"):
        h = hashlib.sha256(open(os.path.join(HERE, fn), "rb").read()).hexdigest()
        print(f"{fn} sha256 {h}")
    print(json.dumps({k: res[k] for k in ("demo_a_first_map", "analytic_check", "demo_c_how_many_tries")}, indent=1))
    print(json.dumps(base, indent=1))


if __name__ == "__main__":
    main()
