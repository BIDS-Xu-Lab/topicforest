---
name: run-topicforest
description: >-
  Runs TopicForest hierarchical clustering and LLM recursive topic labeling
  with run.py. Use when the user asks to run TopicForest, label a corpus,
  build the topic tree, or write cluster assignments and topic labels.
---

# Run TopicForest

Run `run.py` from the repository root. It clusters the coordinate columns, builds the topic tree, then labels every cluster with the LLM, starting at the lowest layer and merging upward. Do not reimplement clustering or the prompts.

## Before labeling

1. Hierarchy values come from the user or from `search_hierarchy.py`. If `--L`, `--k_top_layer`, and `--k_lowest_layer` are unset and the user wants them chosen, use the `search-topicforest-hierarchy` skill first. The published demonstration settings are `--L 3 --k_top_layer 22 --k_lowest_layer 300`.
2. Confirm the table has the identifier column (default `pid`), the coordinate columns (default `x,y`), and a `title` column. Leaf labels are generated from `title`; that column name is fixed.
3. `API_KEY` must be set in the environment or `.env`. `BASE_URL` is optional and is an OpenAI-compatible API root that ends with `/`. Leave it unset for the OpenAI API. Never print the key or the contents of `.env`.
4. Start labeling only after the user asks for it. A few hundred clusters takes minutes and spends API calls. Tested models are `gpt-4o-mini` (the default) and `gpt-4.1-nano`.

Ward linkage requires `--metric euclidean`. For `L` greater than 2, intermediate cluster counts follow the exponential schedule in `build_topic_tree`. Level 0 is the most specific layer.

## Run

```bash
python run.py --path_tsv PATH_TO_TSV \
  --L 3 \
  --k_top_layer 22 \
  --k_lowest_layer 300 \
  --model_name gpt-4.1-nano \
  --deduplicate_topic_labels
```

Pass `--deduplicate_topic_labels` when the user wants sibling labels kept distinct. It is experimental and useful on this corpus. Match `--linkage`, `--metric`, and `--dimensions` to the hierarchy search when one was run. Use `--point_identifier` if the id column is not `pid`, and `--output_dir` to write results somewhere other than the input directory.

Outputs, next to the input unless `--output_dir` is set:

- `<stem>.cluster_assignments.tsv` — identifier plus `L0_clusters` … `L{L-1}_clusters`
- `<stem>.topics.json` — `levels`, each with topic `name`, `description`, `counts`, and `global_topic_key`

A point's label at level `i` is the topic whose `global_topic_key` is `L{i}_{cluster}`. Cluster ids start at 1.

## What to tell the user

Report the command, both output paths, and the cluster count printed for each layer. If labeling fails on a missing key, say that `API_KEY` is unset and stop. Do not paste topic JSON into the chat; point at the file.
