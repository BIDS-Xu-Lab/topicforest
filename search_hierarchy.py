"""Search TopicForest hierarchy depth and cluster counts.

Fits one hierarchical clustering on the same coordinates ``run.py`` uses, then
recommends ``--L``, ``--k_top_layer``, and ``--k_lowest_layer``. Intermediate
layer sizes follow the exponential schedule in
``HierarchicalTopicAnnotator.build_topic_tree``.

This script does not call a language model and does not need an API key.
"""

import argparse
import json
import math
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastcluster import linkage_vector
from scipy.cluster.hierarchy import fcluster
from scipy.ndimage import median_filter
from sklearn.metrics import silhouette_score


# Positive terms are min-max normalized within a role, then combined.
# Penalties stay on their natural 0–1 scale so unusable cuts can lose.
TOP_GAP_WEIGHT = 0.55
TOP_SILHOUETTE_WEIGHT = 0.45
TOP_TINY_PENALTY = 0.40
TOP_IMBALANCE_PENALTY = 0.15

LEAF_GAP_WEIGHT = 0.35
LEAF_SILHOUETTE_WEIGHT = 0.30
LEAF_SIZE_WEIGHT = 0.35
LEAF_TINY_PENALTY = 0.50
LEAF_IMBALANCE_PENALTY = 0.10

BRANCH_DISTANCE_PENALTY = 0.20
BRANCH_OUTSIDE_PENALTY = 0.30


def layer_k_map(k_top, k_lowest, n_layers):
    """Cluster count at each level, matching ``build_topic_tree``.

    Level 0 is the lowest (most specific) layer. Level ``n_layers - 1`` is
    the top (most general) layer. Counts use the same ``int`` truncation as
    the annotator.
    """
    if n_layers < 2:
        raise ValueError("n_layers must be at least 2")
    if k_lowest <= k_top:
        raise ValueError("k_lowest must be greater than k_top")
    growth = (k_lowest / k_top) ** (1 / (n_layers - 1))
    return {
        n_layers - 1 - step: int(k_top * (growth ** step))
        for step in range(n_layers)
    }


def delimiter_for(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return ","
    return "\t"


def load_coordinates(path, dimensions):
    try:
        frame = pd.read_csv(path, sep=delimiter_for(path), usecols=dimensions)
    except ValueError as error:
        raise SystemExit(
            f"Could not read columns {dimensions} from {path}: {error}"
        ) from error
    coordinates = frame[dimensions].to_numpy(dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != len(dimensions):
        raise SystemExit(f"Expected a 2D coordinate matrix, got shape {coordinates.shape}")
    if not np.isfinite(coordinates).all():
        raise SystemExit(
            "Coordinate columns contain non-finite values. Remove those rows and rerun."
        )
    if len(coordinates) < 10:
        raise SystemExit(
            f"Need at least 10 points to search a hierarchy, found {len(coordinates)}."
        )
    return coordinates


def default_top_bounds(n_points):
    if n_points >= 5000:
        return 8, min(40, n_points // 20)
    if n_points >= 1000:
        return 5, min(25, n_points // 15)
    return 2, max(3, min(12, n_points // 10))


def default_leaf_bounds(n_points, min_docs, max_docs):
    k_min = max(2, int(math.ceil(n_points / max_docs)))
    k_max = min(n_points - 1, int(n_points // min_docs))
    if k_max <= k_min:
        k_max = min(n_points - 1, k_min + 1)
    return k_min, k_max


def merge_jumps(heights, ks):
    """Height increase of the merge that would reduce ``k`` clusters to ``k - 1``."""
    n_points = len(heights) + 1
    next_merge = n_points - ks
    at_k = next_merge - 1
    if np.any(at_k < 0) or np.any(next_merge >= len(heights)):
        raise ValueError("k values must lie in 2 .. n-1")
    jumps = heights[next_merge] - heights[at_k]
    return np.maximum(jumps, 0.0)


def odd_window(length, fraction, minimum):
    window = max(minimum, int(round(length * fraction)))
    if window % 2 == 0:
        window += 1
    cap = length if length % 2 == 1 else max(1, length - 1)
    return max(1, min(window, cap))


def gap_prominence(jumps, window):
    baseline = median_filter(jumps, size=window, mode="nearest")
    baseline = np.maximum(baseline, 1e-12)
    return jumps / baseline


def shortlist_counts(k_min, k_max, prominence, must_include, limit):
    """Keep a coverage grid, the target neighborhood, and the strongest gaps."""
    ks = np.arange(k_min, k_max + 1)
    if len(ks) <= limit:
        return ks

    required = {int(k) for k in must_include if k_min <= k <= k_max}
    if k_min < k_max:
        grid = np.unique(np.round(np.geomspace(k_min, k_max, num=min(10, limit))).astype(int))
        required.update(int(k) for k in grid)

    ranked = [int(ks[index]) for index in np.argsort(prominence)[::-1]]
    chosen = set(required)
    for k in ranked:
        if len(chosen) >= limit:
            break
        chosen.add(k)
    if len(chosen) > limit:
        # The coverage set itself overflowed; keep the strongest gaps inside it.
        kept = []
        for k in ranked:
            if k in required:
                kept.append(k)
            if len(kept) >= limit:
                break
        chosen = set(kept)
    return np.array(sorted(chosen), dtype=int)


def minmax(values):
    array = np.asarray(values, dtype=float)
    lo = np.min(array)
    hi = np.max(array)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.full(array.shape, 0.5)
    return (array - lo) / (hi - lo)


def size_summary(labels, min_docs):
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    mean = float(sizes.mean())
    cv = float(sizes.std() / mean) if mean > 0 else 0.0
    return {
        "n_clusters": int(sizes.size),
        "min_size": int(sizes.min()),
        "median_size": float(np.median(sizes)),
        "max_size": int(sizes.max()),
        "tiny_fraction": float(np.mean(sizes < min_docs)),
        "cv": cv,
    }


def safe_silhouette(coordinates, labels, sample_size, rng, metric):
    if len(labels) > sample_size:
        picked = rng.choice(len(labels), size=sample_size, replace=False)
        coordinates = coordinates[picked]
        labels = labels[picked]
    unique, counts = np.unique(labels, return_counts=True)
    keep = unique[counts >= 2]
    if keep.size < 2:
        return None
    mask = np.isin(labels, keep)
    if int(mask.sum()) < 10:
        return None
    try:
        score = silhouette_score(coordinates[mask], labels[mask], metric=metric)
    except ValueError:
        return None
    return float(score)


def size_fit(median_size, target_docs):
    if median_size <= 0 or target_docs <= 0:
        return 0.0
    return math.exp(-abs(math.log(median_size / target_docs)))


def score_candidates(candidates, role, use_silhouette, target_docs):
    gaps = minmax([item["gap_prominence"] for item in candidates])
    silhouettes = np.array(
        [
            np.nan if item["silhouette"] is None else item["silhouette"]
            for item in candidates
        ],
        dtype=float,
    )
    silhouette_usable = use_silhouette and np.isfinite(silhouettes).any()
    if silhouette_usable:
        filled = silhouettes.copy()
        filled[~np.isfinite(filled)] = np.nanmin(silhouettes)
        silhouettes = minmax(filled)
    else:
        silhouettes = np.zeros(len(candidates))

    if role == "top":
        gap_weight = TOP_GAP_WEIGHT
        silhouette_weight = TOP_SILHOUETTE_WEIGHT if silhouette_usable else 0.0
        if not silhouette_usable:
            gap_weight = 1.0
        for index, item in enumerate(candidates):
            imbalance = min(item["cv"], 2.0) / 2.0
            item["size_fit"] = None
            item["score"] = (
                gap_weight * float(gaps[index])
                + silhouette_weight * float(silhouettes[index])
                - TOP_TINY_PENALTY * item["tiny_fraction"]
                - TOP_IMBALANCE_PENALTY * imbalance
            )
        return

    gap_weight = LEAF_GAP_WEIGHT
    silhouette_weight = LEAF_SILHOUETTE_WEIGHT if silhouette_usable else 0.0
    fit_weight = LEAF_SIZE_WEIGHT
    if not silhouette_usable:
        total = LEAF_GAP_WEIGHT + LEAF_SIZE_WEIGHT
        gap_weight = LEAF_GAP_WEIGHT / total
        fit_weight = LEAF_SIZE_WEIGHT / total
    for index, item in enumerate(candidates):
        fit = size_fit(item["median_size"], target_docs)
        imbalance = min(item["cv"], 2.0) / 2.0
        item["size_fit"] = fit
        item["score"] = (
            gap_weight * float(gaps[index])
            + silhouette_weight * float(silhouettes[index])
            + fit_weight * fit
            - LEAF_TINY_PENALTY * item["tiny_fraction"]
            - LEAF_IMBALANCE_PENALTY * imbalance
        )


def select_depth(k_top, k_lowest, target_branch, branch_min, branch_max, max_layers):
    """Pick ``L`` whose branching factor is nearest the target without collapsed layers."""
    if k_lowest <= k_top:
        return None
    ratio = k_lowest / k_top
    viable = []
    for n_layers in range(2, max_layers + 1):
        branching = ratio ** (1 / (n_layers - 1))
        layers = layer_k_map(k_top, k_lowest, n_layers)
        fine_to_coarse = [layers[level] for level in range(n_layers)]
        collapsed = any(
            fine_to_coarse[index] <= fine_to_coarse[index + 1]
            for index in range(n_layers - 1)
        )
        if collapsed:
            continue
        log_distance = abs(math.log(branching / target_branch))
        viable.append(
            {
                "L": n_layers,
                "branching_factor": branching,
                "layers": layers,
                "in_band": branch_min <= branching <= branch_max,
                "log_distance": log_distance,
            }
        )
    if not viable:
        return None
    in_band = [item for item in viable if item["in_band"]]
    pool = in_band or viable
    pool.sort(key=lambda item: (item["log_distance"], item["L"]))
    return pool[0]


def evaluate_counts(
    coordinates,
    linkage,
    ks,
    prominence_by_k,
    min_docs,
    sample_size,
    rng,
    metric,
    use_silhouette,
    role,
):
    label_cache = {}
    evaluated = []
    total = len(ks)
    for index, k in enumerate(ks, start=1):
        k = int(k)
        labels = fcluster(linkage, t=k, criterion="maxclust")
        label_cache[k] = labels
        summary = size_summary(labels, min_docs)
        silhouette = None
        if use_silhouette:
            silhouette = safe_silhouette(
                coordinates, labels, sample_size, rng, metric
            )
        evaluated.append(
            {
                "k": k,
                "gap_prominence": float(prominence_by_k[k]),
                "silhouette": silhouette,
                **summary,
            }
        )
        if index == 1 or index == total or index % 5 == 0:
            print(f"* {role} cut {index}/{total} (k={k})", flush=True)
    return evaluated, label_cache


def prominence_lookup(k_min, k_max, heights):
    ks = np.arange(k_min, k_max + 1)
    jumps = merge_jumps(heights, ks)
    window = odd_window(len(ks), fraction=0.08, minimum=5)
    prominence = gap_prominence(jumps, window)
    return ks, {int(k): float(value) for k, value in zip(ks, prominence)}, prominence


def leaf_must_include(n_points, k_min, k_max, target_docs):
    target_k = int(round(n_points / target_docs))
    target_k = min(max(target_k, k_min), k_max)
    scales = (0.7, 0.85, 1.0, 1.15, 1.35)
    neighbors = [int(round(target_k * scale)) for scale in scales]
    return [k for k in neighbors if k_min <= k <= k_max]


def layer_rows(depth, label_cache, linkage, min_docs_for_level):
    rows = []
    n_layers = depth["L"]
    for level in range(n_layers - 1, -1, -1):
        k = int(depth["layers"][level])
        if k not in label_cache:
            label_cache[k] = fcluster(linkage, t=k, criterion="maxclust")
        if level == n_layers - 1:
            role = "top"
            min_docs = min_docs_for_level["top"]
        elif level == 0:
            role = "lowest"
            min_docs = min_docs_for_level["lowest"]
        else:
            role = "intermediate"
            min_docs = min_docs_for_level["lowest"]
        summary = size_summary(label_cache[k], min_docs)
        rows.append(
            {
                "level": level,
                "role": role,
                "k": summary["n_clusters"],
                "requested_k": k,
                "min_size": summary["min_size"],
                "median_size": round(summary["median_size"], 1),
                "max_size": summary["max_size"],
            }
        )
    return rows


def pair_score(top, lowest, depth):
    score = top["score"] + lowest["score"] - BRANCH_DISTANCE_PENALTY * depth["log_distance"]
    if not depth["in_band"]:
        score -= BRANCH_OUTSIDE_PENALTY
    return score


def build_reasons(top, lowest, depth, target_docs, branch_min, branch_max):
    sil_top = (
        f", silhouette {top['silhouette']:.3f}"
        if top["silhouette"] is not None
        else ""
    )
    sil_low = (
        f", silhouette {lowest['silhouette']:.3f}"
        if lowest["silhouette"] is not None
        else ""
    )
    schedule = " → ".join(
        f"L{level}={k}"
        for level, k in sorted(depth["layers"].items(), reverse=True)
    )
    band = (
        f"inside the preferred branching band {branch_min:g}–{branch_max:g}"
        if depth["in_band"]
        else f"outside the preferred branching band {branch_min:g}–{branch_max:g}; widen the k ranges or change --target_branch"
    )
    resolved_lowest = depth["layers"][0]
    lowest_note = (
        f"The exponential schedule resolves the lowest layer to {resolved_lowest} clusters."
        if resolved_lowest != lowest["k"]
        else f"The lowest layer keeps {resolved_lowest} clusters."
    )
    return [
        (
            f"Top cut k={top['k']} is a strong dendrogram gap "
            f"(prominence {top['gap_prominence']:.2f}{sil_top}); "
            f"median size {top['median_size']:.0f} documents, "
            f"{top['tiny_fraction']:.1%} of clusters below the top-layer size floor."
        ),
        (
            f"Lowest cut k={lowest['k']} has median size {lowest['median_size']:.0f} "
            f"(target {target_docs:g} documents per leaf, size fit {lowest['size_fit']:.2f}"
            f"{sil_low}); "
            f"{lowest['tiny_fraction']:.1%} of clusters are below the leaf size floor."
        ),
        (
            f"L={depth['L']} gives {schedule} "
            f"(branching {depth['branching_factor']:.2f}, {band}). {lowest_note}"
        ),
    ]


def run_command(path_tsv, dimensions, linkage, metric, n_layers, k_top, k_lowest):
    parts = [
        "python",
        "run.py",
        "--path_tsv",
        path_tsv,
        "--L",
        str(n_layers),
        "--k_top_layer",
        str(k_top),
        "--k_lowest_layer",
        str(k_lowest),
        "--linkage",
        linkage,
        "--metric",
        metric,
    ]
    if dimensions != ["x", "y"]:
        parts.extend(["--dimensions", ",".join(dimensions)])
    return " ".join(shlex.quote(part) for part in parts)


def public_candidate(item):
    payload = {
        "k": item["k"],
        "score": round(float(item["score"]), 4),
        "gap_prominence": round(float(item["gap_prominence"]), 4),
        "silhouette": None
        if item["silhouette"] is None
        else round(float(item["silhouette"]), 4),
        "n_clusters": item["n_clusters"],
        "min_size": item["min_size"],
        "median_size": round(float(item["median_size"]), 1),
        "max_size": item["max_size"],
        "tiny_fraction": round(float(item["tiny_fraction"]), 4),
        "cv": round(float(item["cv"]), 4),
    }
    if item.get("size_fit") is not None:
        payload["size_fit"] = round(float(item["size_fit"]), 4)
    return payload


def public_solution(solution):
    depth = solution["depth"]
    return {
        "L": depth["L"],
        "k_top_layer": solution["top"]["k"],
        "k_lowest_layer": solution["lowest"]["k"],
        "branching_factor": round(depth["branching_factor"], 3),
        "branching_in_preferred_band": depth["in_band"],
        "score": round(solution["score"], 4),
        "layers": solution["layer_rows"],
        "reasons": solution["reasons"],
    }


def search(coordinates, args, linkage_method, metric):
    n_points = len(coordinates)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    print(
        f"* linkage on {n_points} points "
        f"({linkage_method}, {metric})",
        flush=True,
    )
    linkage = linkage_vector(coordinates, method=linkage_method, metric=metric)
    heights = linkage[:, 2]
    print(f"* linkage finished in {time.perf_counter() - started:.2f}s", flush=True)

    k_top_min = args.k_top_min
    k_top_max = args.k_top_max
    if k_top_min is None or k_top_max is None:
        auto_min, auto_max = default_top_bounds(n_points)
        k_top_min = auto_min if k_top_min is None else k_top_min
        k_top_max = auto_max if k_top_max is None else k_top_max
    k_leaf_min, k_leaf_max = default_leaf_bounds(
        n_points, args.min_docs_per_leaf, args.max_docs_per_leaf
    )
    if args.k_lowest_min is not None:
        k_leaf_min = args.k_lowest_min
    if args.k_lowest_max is not None:
        k_leaf_max = args.k_lowest_max

    k_top_max = min(k_top_max, n_points - 2)
    k_leaf_max = min(k_leaf_max, n_points - 1)
    k_top_min = max(2, k_top_min)
    k_leaf_min = max(k_top_min + 1, k_leaf_min)
    if k_top_min > k_top_max or k_leaf_min > k_leaf_max:
        raise SystemExit(
            "Cluster-count ranges are empty. Relax --k_top_min/max or the leaf document bounds. "
            f"Got top {k_top_min}..{k_top_max}, lowest {k_leaf_min}..{k_leaf_max}."
        )

    _, top_prominence, top_prom_array = prominence_lookup(k_top_min, k_top_max, heights)
    _, leaf_prominence, leaf_prom_array = prominence_lookup(k_leaf_min, k_leaf_max, heights)
    top_span = k_top_max - k_top_min + 1
    top_ks = shortlist_counts(
        k_top_min,
        k_top_max,
        top_prom_array,
        must_include=[],
        limit=min(top_span, max(40, args.leaf_candidates)),
    )
    leaf_ks = shortlist_counts(
        k_leaf_min,
        k_leaf_max,
        leaf_prom_array,
        must_include=leaf_must_include(
            n_points, k_leaf_min, k_leaf_max, args.target_docs_per_leaf
        ),
        limit=args.leaf_candidates,
    )
    print(
        f"* scoring {len(top_ks)} top-layer cuts ({k_top_min}..{k_top_max}) "
        f"and {len(leaf_ks)} lowest-layer cuts ({k_leaf_min}..{k_leaf_max})",
        flush=True,
    )

    use_silhouette = not args.skip_silhouette
    top_candidates, top_labels = evaluate_counts(
        coordinates,
        linkage,
        top_ks,
        top_prominence,
        args.min_docs_per_top,
        args.silhouette_sample,
        rng,
        metric,
        use_silhouette,
        role="top",
    )
    leaf_candidates, leaf_labels = evaluate_counts(
        coordinates,
        linkage,
        leaf_ks,
        leaf_prominence,
        args.min_docs_per_leaf,
        args.silhouette_sample,
        rng,
        metric,
        use_silhouette,
        role="lowest",
    )
    label_cache = {**top_labels, **leaf_labels}
    score_candidates(top_candidates, "top", use_silhouette, args.target_docs_per_leaf)
    score_candidates(leaf_candidates, "lowest", use_silhouette, args.target_docs_per_leaf)

    solutions = []
    for top in top_candidates:
        for lowest in leaf_candidates:
            if lowest["k"] <= top["k"]:
                continue
            depth = select_depth(
                top["k"],
                lowest["k"],
                args.target_branch,
                args.branch_min,
                args.branch_max,
                args.max_layers,
            )
            if depth is None:
                continue
            solutions.append(
                {
                    "top": top,
                    "lowest": lowest,
                    "depth": depth,
                    "score": pair_score(top, lowest, depth),
                }
            )
    if not solutions:
        raise SystemExit(
            "No hierarchy kept strictly decreasing cluster counts. "
            "Increase --max_layers or widen the gap between top and lowest k."
        )

    solutions.sort(key=lambda item: item["score"], reverse=True)
    unique = []
    seen = set()
    for solution in solutions:
        identity = (
            solution["depth"]["L"],
            solution["top"]["k"],
            solution["lowest"]["k"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        solution["reasons"] = build_reasons(
            solution["top"],
            solution["lowest"],
            solution["depth"],
            args.target_docs_per_leaf,
            args.branch_min,
            args.branch_max,
        )
        solution["layer_rows"] = layer_rows(
            solution["depth"],
            label_cache,
            linkage,
            {
                "top": args.min_docs_per_top,
                "lowest": args.min_docs_per_leaf,
            },
        )
        unique.append(solution)
        if len(unique) >= args.alternatives + 1:
            break

    best = unique[0]
    command = run_command(
        args.path_tsv,
        args.dimensions,
        linkage_method,
        metric,
        best["depth"]["L"],
        best["top"]["k"],
        best["lowest"]["k"],
    )
    report = {
        "input": {
            "path_tsv": args.path_tsv,
            "n_points": n_points,
            "dimensions": args.dimensions,
            "linkage": linkage_method,
            "metric": metric,
            "k_top_range": [k_top_min, k_top_max],
            "k_lowest_range": [k_leaf_min, k_leaf_max],
            "target_docs_per_leaf": args.target_docs_per_leaf,
            "min_docs_per_leaf": args.min_docs_per_leaf,
            "max_docs_per_leaf": args.max_docs_per_leaf,
            "min_docs_per_top": args.min_docs_per_top,
            "target_branch": args.target_branch,
            "branch_band": [args.branch_min, args.branch_max],
            "max_layers": args.max_layers,
            "silhouette_sample": 0 if args.skip_silhouette else args.silhouette_sample,
            "seed": args.seed,
        },
        "method": (
            "One linkage on the map coordinates TopicForest clusters. "
            "Cuts are ranked by dendrogram-gap prominence, subsampled silhouette, "
            "and leaf-size fit. L is the depth whose exponential cluster schedule "
            "matches run.py and whose branching factor is nearest the target."
        ),
        "caveat": (
            "Scores measure separation in the clustered coordinates "
            "(usually the 2D map), which is the space TopicForest uses. "
            "They are not a quality score in the original embedding space."
        ),
        "recommendation": public_solution(best),
        "alternatives": [public_solution(item) for item in unique[1:]],
        "top_candidates": [
            public_candidate(item)
            for item in sorted(top_candidates, key=lambda item: item["score"], reverse=True)[:15]
        ],
        "lowest_candidates": [
            public_candidate(item)
            for item in sorted(leaf_candidates, key=lambda item: item["score"], reverse=True)[:15]
        ],
        "run_command": command,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    return report


def print_report(report):
    recommendation = report["recommendation"]
    print()
    print("Recommendation")
    print(
        f"  L={recommendation['L']}  "
        f"k_top_layer={recommendation['k_top_layer']}  "
        f"k_lowest_layer={recommendation['k_lowest_layer']}  "
        f"branching={recommendation['branching_factor']:.2f}  "
        f"score={recommendation['score']:.3f}"
    )
    print("  layers (coarse → fine):")
    for layer in recommendation["layers"]:
        print(
            f"    L{layer['level']} {layer['role']:<12} "
            f"k={layer['k']:<5} median={layer['median_size']:.0f}  "
            f"min={layer['min_size']}  max={layer['max_size']}"
        )
    print("  why:")
    for reason in recommendation["reasons"]:
        print(f"    - {reason}")
    if report["alternatives"]:
        print("Alternatives")
        for index, alternative in enumerate(report["alternatives"], start=1):
            schedule = " → ".join(
                f"L{layer['level']}={layer['k']}" for layer in alternative["layers"]
            )
            print(
                f"  {index}. L={alternative['L']} "
                f"k_top={alternative['k_top_layer']} "
                f"k_lowest={alternative['k_lowest_layer']} "
                f"branching={alternative['branching_factor']:.2f} "
                f"score={alternative['score']:.3f}  ({schedule})"
            )
    print("Run")
    print(f"  {report['run_command']}")
    print(f"Note: {report['caveat']}")


def write_report(report, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {path}")


def self_check():
    layers = layer_k_map(22, 300, 3)
    expected = {2: 22, 1: 81, 0: 300}
    if layers != expected:
        raise SystemExit(f"layer schedule mismatch: {layers} != {expected}")
    collapsed = select_depth(10, 12, target_branch=4, branch_min=3, branch_max=6, max_layers=4)
    # 10 → 12 over 3+ layers collapses under int truncation; L=2 branching is 1.2, outside the band.
    if collapsed is None or collapsed["L"] != 2 or collapsed["in_band"]:
        raise SystemExit(f"unexpected depth selection: {collapsed}")
    print("self-check ok")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Search the number of TopicForest layers and the cluster counts "
            "for the top and lowest layers."
        ),
        epilog=(
            "Example: python search_hierarchy.py "
            "--path_tsv bio_scirep/24k_abstracts.tsv"
        ),
    )
    parser.add_argument("--path_tsv", help="TSV or CSV with coordinate columns.")
    parser.add_argument(
        "--dimensions",
        default="x,y",
        help="Comma-separated coordinate columns. Default: x,y",
    )
    parser.add_argument("--linkage", default="ward", help="Linkage method. Default: ward")
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="Distance metric. Ward requires euclidean. Default: euclidean",
    )
    parser.add_argument("--k_top_min", type=int, default=None, help="Smallest top-layer k to score.")
    parser.add_argument("--k_top_max", type=int, default=None, help="Largest top-layer k to score.")
    parser.add_argument(
        "--k_lowest_min",
        type=int,
        default=None,
        help="Override the coarsest lowest-layer k. Default: n / max_docs_per_leaf.",
    )
    parser.add_argument(
        "--k_lowest_max",
        type=int,
        default=None,
        help="Override the finest lowest-layer k. Default: n / min_docs_per_leaf.",
    )
    parser.add_argument(
        "--min_docs_per_leaf",
        type=int,
        default=20,
        help="Leaf clusters smaller than this are penalized, and it sets the finest k. Default: 20",
    )
    parser.add_argument(
        "--max_docs_per_leaf",
        type=int,
        default=200,
        help="Sets the coarsest lowest-layer k to about n / this value. Default: 200",
    )
    parser.add_argument(
        "--target_docs_per_leaf",
        type=int,
        default=80,
        help="Preferred median documents per lowest-layer cluster. Default: 80",
    )
    parser.add_argument(
        "--min_docs_per_top",
        type=int,
        default=30,
        help="Top-layer clusters smaller than this are penalized. Default: 30",
    )
    parser.add_argument(
        "--target_branch",
        type=float,
        default=4.0,
        help="Preferred average child-per-parent ratio used to choose L. Default: 4",
    )
    parser.add_argument("--branch_min", type=float, default=3.0, help="Lowest preferred branching. Default: 3")
    parser.add_argument("--branch_max", type=float, default=6.0, help="Highest preferred branching. Default: 6")
    parser.add_argument("--max_layers", type=int, default=5, help="Largest L to consider. Default: 5")
    parser.add_argument(
        "--leaf_candidates",
        type=int,
        default=24,
        help="How many lowest-layer cuts to score fully. Default: 24",
    )
    parser.add_argument(
        "--silhouette_sample",
        type=int,
        default=2500,
        help="Points used for each silhouette score. Default: 2500",
    )
    parser.add_argument(
        "--skip_silhouette",
        action="store_true",
        help="Rank cuts by dendrogram gap and size only.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Subsample seed. Default: 0")
    parser.add_argument(
        "--alternatives",
        type=int,
        default=4,
        help="How many runner-up hierarchies to report. Default: 4",
    )
    parser.add_argument(
        "--output",
        default="",
        help="JSON report path. Default: <input stem>.hierarchy_search.json beside the input.",
    )
    parser.add_argument(
        "--self_check",
        action="store_true",
        help="Check the layer schedule against the known 22 / 81 / 300 example and exit.",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        return args
    if not args.path_tsv:
        parser.error("--path_tsv is required unless --self_check is set")
    if args.linkage == "ward" and args.metric != "euclidean":
        parser.error("Ward linkage requires --metric euclidean.")
    if args.min_docs_per_leaf < 2:
        parser.error("--min_docs_per_leaf must be at least 2")
    if args.max_docs_per_leaf < args.min_docs_per_leaf:
        parser.error("--max_docs_per_leaf must be >= --min_docs_per_leaf")
    if not (args.min_docs_per_leaf <= args.target_docs_per_leaf <= args.max_docs_per_leaf):
        parser.error("--target_docs_per_leaf must lie between the leaf min and max document bounds")
    if args.branch_min <= 1 or args.branch_max < args.branch_min:
        parser.error("--branch_min must be > 1 and --branch_max must be >= --branch_min")
    if args.target_branch <= 1:
        parser.error("--target_branch must be > 1")
    if args.max_layers < 2:
        parser.error("--max_layers must be at least 2")
    if args.leaf_candidates < 3:
        parser.error("--leaf_candidates must be at least 3")
    if args.alternatives < 0:
        parser.error("--alternatives must be >= 0")
    args.dimensions = [part.strip() for part in args.dimensions.split(",") if part.strip()]
    if len(args.dimensions) < 1:
        parser.error("--dimensions must list at least one column")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_check:
        self_check()
        return 0
    coordinates = load_coordinates(args.path_tsv, args.dimensions)
    report = search(coordinates, args, args.linkage, args.metric)
    output = args.output
    if not output:
        input_path = Path(args.path_tsv)
        output = str(input_path.with_name(f"{input_path.stem}.hierarchy_search.json"))
    print_report(report)
    write_report(report, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
