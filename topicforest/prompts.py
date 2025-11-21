generation_schema = {
  "name": "topic_generation_schema",
  "strict": True,
  "schema": {
    "type": "object",
    "required": [
      "topic_label",
      "description"
    ],
    "properties": {
      "description": {
        "type": "string"
      },
      "topic_label": {
        "type": "string"
      }
    },
    "additionalProperties": False
  }
}

topic_generation_system_prompt = "You are an expert in analyzing scientific literature."

topic_generation_prompt = """Your task is to identify a concise and representative topic label for a given list of research titles. The topic label should be 2–3 words long and summarize the overarching theme of the titles. Additionally, provide a brief description (1 sentence) explaining the label and how it captures the shared theme.

Task:
1. Analyze the titles to identify their shared themes or topics.
2. Generate a single representative topic label (2–3 words) summarizing the main theme of these titles.
3. Provide a brief description (1 sentence) explaining the topic label and how it reflects the core ideas of the titles.

Output Format:
```json
{{
  "topic_label": "[Your topic label]",
  "description": "[Brief explanation of the label and its relevance]"
}}
```

Here is the list of titles (one for each line):
{titles}
"""

topic_merging_system_prompt = "You are an expert in analyzing and organizing scientific topics."

topic_merging_prompt = """Your task is to review a list of topic labels and their descriptions, identify any overlapping or closely related topics, and merge them into a single representative topic label. The merged topic label should be 2–3 words long and summarize the shared theme of the combined topics. Additionally, provide a concise description (1–2 sentences) for the merged topic label that reflects its unified scope.

Task:
1. Review the topic labels and descriptions.
2. Identify topics that overlap or are closely related.
3. Merge the related topics into a single representative topic label (2–3 words).
4. Provide a concise description (1–2 sentences) for the merged topic label that explains its unified theme.

Output Format:
```json
{{
  "topic_label": "[Your merged topic label]",
  "description": "[Brief explanation of the merged topic label]"
}}
```

Here is a list of topic labels with their descriptions (one for each line):
{labels_and_descriptions}
"""

topic_deduplication_schema = {
  "name": "topic_deduplication_schema",
  "strict": True,
  "schema": {
    "type": "object",
    "required": [
      "new_topic_label"
    ],
    "properties": {
      "new_topic_label": {
        "type": "string"
      }
    },
    "additionalProperties": False
  }
}

deduplication_system_prompt = """You are a meticulous biomedical curator. Your job is to generate a DISTINCT, SPECIFIC topic label that avoid collisions with sibling topics.

Rules:
- Prioritize specificity: mechanism/action, molecular target, pathway, cell/tissue context, disease/phenotype, species/population, modality (omics/assay), directionality (↑/↓, activation/inhibition), and clinical stage (preclinical/clinical) when available.
- Keep the label tight: 2-4 words, Title Case, no punctuation except hyphens if needed.
- Avoid generic heads like “Biology”, “Pathways”, “Mechanisms”, “Signaling” unless paired with a concrete qualifier (e.g., “TLR4 Signaling in Kupffer Cells”).
- Must be semantically non-overlapping with the provided “other” labels. If necessary, narrow scope (e.g., add tissue, cell type, disease stage, species, modality, or mechanism) to differentiate.
- Do NOT repeat any “other” label verbatim or as a strict synonym.
- Be faithful to the focal description; do not invent unsupported scope.
- Output strictly valid JSON that matches the schema in the user prompt."""

deduplication_prompt = """You will refine a focal topic label to avoid duplication with its siblings.

Inputs
------
Focal:
- label: {focal_topic_label}
- description: {focal_topic_description}

# formatted_other_topics is a bullet list, each item like:
# - label: X
#   description: Y
Other topics to avoid colliding with:
{formatted_other_topics}

Task
----
1) Diagnose overlap: identify what makes the focal topic potentially confusable with the “other” labels (e.g., same pathway, disease, or too-generic head).
2) Propose a DISTINCT, MORE SPECIFIC label (2–4 words, Title Case) that remains true to the focal description while maximizing separability from “other” labels. Use the strongest available differentiators.
3) Validate uniqueness: ensure the new label is not identical to, or a trivial synonym of, any “other” labels.

Output (strict JSON)
--------------------
{{
  "new_topic_label": "<2–4 words, Title Case, distinct>"
}}"""