---
name: search-topicforest-hierarchy
description: >-
  Chooses TopicForest hierarchy depth and cluster counts (L, k_top_layer,
  k_lowest_layer) by running search_hierarchy.py. Use when the user asks how
  many layers, topics, or clusters to use, what k_top_layer or k_lowest_layer
  should be, or wants to tune the hierarchy before run.py.
---

# Search TopicForest hierarchy

Decide `--L`, `--k_top_layer`, and `--k_lowest_layer` by running `search_hierarchy.py` from the repository root. Do not invent these values and do not reimplement the search.

The script clusters the same coordinate columns as `run.py` (default `x,y`), using the same Ward linkage and the same exponential layer schedule as `HierarchicalTopicAnnotator.build_topic_tree`. Level 0 is the lowest, most specific layer. The highest level is the top, most general layer. It does not call a language model and does not need `API_KEY`.

## Run

```bash
python search_hierarchy.py --path_tsv PATH_TO_TSV
```

JSON is written to `<input stem>.hierarchy_search.json` next to the input. Pass `--output` to put it somewhere else. Read that JSON. Trust `recommendation`, `alternatives`, and `run_command`.

Useful overrides, only when the user states a preference:

| Preference | Flag |
|---|---|
| Broader leaf topics | raise `--target_docs_per_leaf` (default 80) |
| Finer leaf topics | lower `--target_docs_per_leaf`, and lower `--min_docs_per_leaf` if the floor blocks it |
| Fewer or more top groups | `--k_top_min`, `--k_top_max` (defaults scale with corpus size; often 8–40) |
| Shallower or deeper tree | `--target_branch` (default 4), `--branch_min`, `--branch_max`, `--max_layers` |
| Same linkage as a planned run | `--linkage`, `--metric`, `--dimensions` |
| Faster, geometry-light ranking | `--skip_silhouette` |

Ward linkage requires `--metric euclidean`.

## What to tell the user

Lead with the recommendation: `L`, `k_top_layer`, `k_lowest_layer`, the resolved layer sizes from coarse to fine, and median documents per cluster. Then give two or three alternatives and the `run_command`.

Use the `reasons` strings. Do not claim the result is optimal in the original embedding space. Say that scores are on the coordinates TopicForest clusters, usually the 2D map.

Do not start `run.py` unless the user asks to label the corpus. Labeling calls the LLM. When they do, follow the `run-topicforest` skill.

## How a cut is chosen

- **Top layer:** dendrogram-gap prominence and subsampled silhouette, minus a penalty for tiny or very unbalanced clusters. The search window is the navigable range (`k_top_min`–`k_top_max`).
- **Lowest layer:** the same geometry, plus a preference that the median cluster hold about `--target_docs_per_leaf` documents. Candidates are gap peaks, a log-spaced grid, and counts near `n / target_docs_per_leaf`.
- **Layers:** among depths 2..`max_layers`, pick `L` whose branching factor `(k_lowest / k_top) ** (1 / (L - 1))` is inside `--branch_min`–`--branch_max` and nearest `--target_branch`, and whose `int`-truncated layer sizes stay strictly decreasing. That truncation is the one in `build_topic_tree`.

If `branching_in_preferred_band` is false, say so and rerun with a wider k range or a different `--target_branch` before treating the depth as settled.

## Checks

`python search_hierarchy.py --self_check` confirms the layer schedule for `k_top=22`, `k_lowest=300`, `L=3` is `22 → 81 → 300`.
