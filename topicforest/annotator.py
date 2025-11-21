import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from fastcluster import linkage_vector
from openai import OpenAI
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from .config import HierarchicalTopicAnnotatorConfig
from .topic_tree import TopicTree
from .prompts import (
    generation_schema,
    topic_generation_system_prompt,
    topic_generation_prompt,
    topic_merging_system_prompt,
    topic_merging_prompt,
    topic_deduplication_schema,
    deduplication_system_prompt,
    deduplication_prompt
)

from collections import defaultdict
from dataclasses import asdict

from enum import Enum
import json
import time

class HierarchicalTopicAnnotator:
    def __init__(self, 
                 config: HierarchicalTopicAnnotatorConfig,
                 path_tsv:str,
                 model_name:str = "gpt-4o-mini"):

        self.config = config
        self.K_TOP_LAYER = self.config.topic_cluster_configs.k_top_layer
        self.K_LOWEST_LAYER = self.config.topic_cluster_configs.k_lowest_layer
        self.L = self.config.topic_cluster_configs.L
        self.model_name = model_name # the model name to use for the llm

        if config.openAI_api_key != "":
            self.client = OpenAI(
                api_key=config.openAI_api_key,
            )

        # caches for topic tree
        self.topic_tree = TopicTree()
        self.labels_cache = {} # map from level_key to labels (i.e., cluster assignments)
        self.centers_cache = {} # map from level_key to centers (i.e., cluster centers)

        # caches for llm-based topic annotation
        self.llm_topic_cache = {} # map from global_topic_key to topic names (str)
        self.llm_topic_description_cache = {} # map from global_topic_key to topic description (str)
        self.llm_topic_title_cache = {} # map from global_topic_key to list of titles (list of str)

        print('* loading points from %s' % path_tsv)
        self.points_df = pd.read_csv(path_tsv, sep=self._get_delimiter(path_tsv))
        print('* loaded points dataframe from %s' % path_tsv)

    #######################################################
    # Function for clustering data points and constructing the topic tree
    #######################################################
    def cluster(self):
        # Create a DataFrame with the topic tree

        X = self.points_df[self.config.dimensions].values

        # Perform hierarchical clustering
        print(f"* begin clustering with linkage method: {self.config.topic_cluster_configs.linkage}, metric: {self.config.topic_cluster_configs.metric}")
        t0 = time.perf_counter()
        # Use linkage_vector to avoid out-of-memory error and achieve better efficiency
        self.Z = linkage_vector(X, method=self.config.topic_cluster_configs.linkage, metric=self.config.topic_cluster_configs.metric)
        t1 = time.perf_counter()
        print(f"* complete the calculation of the linkage matrix in {t1 - t0} seconds")

    def build_topic_tree(self):
        # A heuristic for dertimining the number of clusters in each layer
        # Compute the exponential growth rate r 
        r = (self.K_LOWEST_LAYER / self.K_TOP_LAYER) ** (1 / (self.L - 1))
        # Generate cluster sizes for each layer
        self.layer_k_values = {self.L-1-i: int(self.K_TOP_LAYER * (r ** i)) for i in range(self.L)}

        # Generate cluster assignments & cluster centers for each layer
        for layer, k in self.layer_k_values.items():
            labels = fcluster(self.Z, k, criterion='maxclust')
            # Assign the cluster assignments to the points dataframe
            self.points_df[f'L{layer}_clusters'] = labels
            self.labels_cache[layer] = labels # cache the cluster assignments (i.e., labels)

            print('* layer', layer, 'number of clusters: ', np.unique(labels).shape[0])

            mean = lambda x: np.average(x)
            # Calculate the average x and y coordinates for each cluster
            dimensions = self.config.dimensions
            centers = (
                self.points_df[dimensions + [f"L{layer}_clusters"]]
                .groupby(f"L{layer}_clusters")
                .agg({dim: mean for dim in dimensions})
                .reset_index()
            )
            self.centers_cache[layer] = centers # cache the cluster centers
        
        # Check from the lowest layer to the top layer
        for layer, k in sorted(self.layer_k_values.items()):

            # Skip the last layer
            if layer == self.L-1:
                print(f"\t* Skip L{layer}_clusters")
                continue

            print(f"\t* Checking L{layer}_clusters")
            for i in range(1, k+1): # topic index starts from 1
                parent_cluster_columns = [f"L{l}_clusters" for l in range(layer+1, self.L)]
                number_of_unique_path = self.points_df[self.points_df[f"L{layer}_clusters"]==i][parent_cluster_columns].value_counts().shape[0]
                # if number of unique path is not 1, print the cluster number
                if number_of_unique_path != 1:
                    print(i)

        # Construct the topic tree by looping every child level 
        for child_level in range(0, self.L-1): # L-1 is the last layer, which will not be the child of any other layer
            parent_level = child_level + 1
            child_column = f"L{child_level}_clusters"
            parent_column = f"L{parent_level}_clusters"

            new_edges = []

            for _, row in self.points_df[[parent_column, child_column]].drop_duplicates().iterrows():
                child_node_key = f"L{child_level}_{row[child_column]}"
                parent_node_key = f"L{parent_level}_{row[parent_column]}"
                new_edges.append((child_node_key, parent_node_key))

            self.topic_tree.add_edges(new_edges)
    
    #######################################################
    # Functions for LLM-based recursive topic labeling
    #######################################################
    def get_llm_based_topic_annotation(self):
        
        levels = []

        # starting from the lowest level (most fine-grained level, i.e., 0)
        for level in sorted(self.labels_cache.keys()):
            
            # level-wise dictionary: creating the mapping dictionary: label (cluster index) to list of point indices (data points)
            cluster_point_indices = defaultdict(list)
            for i, label in enumerate(self.labels_cache[level]):
                cluster_point_indices[label].append(i)
            
            num_of_clusters = len(cluster_point_indices.keys())
            print(f"* At level: {level}, there are {num_of_clusters} cluters.")
            
            topics = []
            pbar = tqdm(total=num_of_clusters)
            for label, indices in cluster_point_indices.items():
                
                # get the pids
                cluster_pmids = self.points_df.iloc[indices][self.config.point_identifier].tolist()
                # get the center for this cluster (label)
                if (center := self.centers_cache[level].loc[self.centers_cache[level][f"L{level}_clusters"] == label].iloc[0]) is None:
                    continue
                if cluster_pmids is None:
                    continue
            
                global_key = self._get_global_topic_key(level, label)

                # if the topic label is not in the cache, generate the topic label
                # otherwise, use the cached topic label
                if global_key not in self.llm_topic_cache: 

                    # select children
                    children = self.topic_tree.parent_to_children[global_key]

                    # if there is no children for this label (i.e., leaf cluster)
                    if len(children) == 0:
                        list_of_titles = self.points_df.iloc[indices]["title"].tolist()
                        # generate the topic label by the TopicGeneration prompt, requiring a list of titles
                        output_dict = self._generate_topic_label(titles="\n".join(list_of_titles))
                        self.llm_topic_title_cache[global_key] = list_of_titles
                        self.llm_topic_cache[global_key] = output_dict['topic_label']
                        self.llm_topic_description_cache[global_key] = output_dict['description']
                    # deriving cluster
                    elif len(children) == 1:
                        # derive the same topic label as its only child
                        child_global_key = children[0]
                        self.llm_topic_cache[global_key] = self.llm_topic_cache[child_global_key]
                        self.llm_topic_description_cache[global_key] = self.llm_topic_description_cache[child_global_key]
                    # merging cluster
                    elif len(children) >1:
                        # collect all children's topic labels and desciptions
                        labels_and_descriptions = "\n".join([f"{self.llm_topic_cache[child_key]}: {self.llm_topic_description_cache[child_key]}" for child_key in children])
                        # generate the topic label by the TopicMerging prompt, requiring a list of topic labels 
                        output_dict = self._merge_topic_labels(labels_and_descriptions)
                        self.llm_topic_cache[global_key] = output_dict['topic_label']
                        self.llm_topic_description_cache[global_key] = output_dict['description']

                _topic = {
                    "name": self.llm_topic_cache[global_key], 
                    "counts": len(cluster_pmids),
                    "pids": cluster_pmids if self.config.flag_output_include_pids else [],
                }

                # add dimensions
                for dim in self.config.dimensions:
                    _topic[dim] = float(center[dim])
                
                # append topic
                topics.append(_topic)
                pbar.update(1)

            ret = {
                "level": level, 
                "meta": asdict(self.config.topic_cluster_configs), 
                "topics": topics
            }

            levels.append(ret)
            pbar.close()
        
        return levels

    def level_wise_label_deduplication(self, level):
        # Group the global topickeys by the topic label.
        label_global_key_dict = defaultdict(list) # Key is topic label, and value is list of global topic keys
        for _, label in enumerate(np.unique(self.labels_cache[level])):
            global_key = self._get_global_topic_key(level, label)
            topic_label = self.llm_topic_cache[global_key]
            label_global_key_dict[topic_label].append(global_key)

        # If a topic label is shared by more than one global key, we use the deduplication prompt to generate a more specific topic label.
        for _, global_keys in label_global_key_dict.items():
            if len(global_keys) > 1:
                for _, focal_global_key in enumerate(global_keys):
                    focal_topic_label = self.llm_topic_cache[focal_global_key]
                    focal_topic_description = self.llm_topic_description_cache[focal_global_key]

                    # select the all other global keys except the focal one
                    other_global_keys = [key for key in global_keys if key != focal_global_key]
                    other_topic_labels = [self.llm_topic_cache[key] for key in other_global_keys]
                    other_topic_descriptions = [self.llm_topic_description_cache[key] for key in other_global_keys]

                    prompt = deduplication_prompt.format(
                        focal_topic_label=focal_topic_label,
                        focal_topic_description=focal_topic_description,
                        formatted_other_topics=self._format_other_topics(other_topic_labels, other_topic_descriptions)
                    )
                    
                    output_dict = self._deduplicate_topic_labels(prompt)
                    new_topic_label = output_dict["new_topic_label"]
                    
                    # uncomment this to print the deduplication results
                    # print(f"Global Key: {focal_global_key} | Old Topic Label: {focal_topic_label} | New Topic Label: {new_topic_label}")
                    
                    # update the topic label
                    self.llm_topic_cache[focal_global_key] = new_topic_label

    @retry(wait=wait_random_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def _generate_topic_label(self, titles):

        system_prompt = topic_generation_system_prompt
        prompt = topic_generation_prompt.format(titles=titles)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": generation_schema
            },
            temperature=0.1,
            max_completion_tokens=8192,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        return json.loads(response.choices[0].message.content)

    @retry(wait=wait_random_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def _merge_topic_labels(self, labels_and_descriptions):

        system_prompt = topic_merging_system_prompt
        prompt = topic_merging_prompt.format(labels_and_descriptions=labels_and_descriptions)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": generation_schema
            },
            temperature=0.1,
            max_completion_tokens=8192,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        return json.loads(response.choices[0].message.content)
    
    @retry(wait=wait_random_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def _deduplicate_topic_labels(self, prompt):
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": deduplication_system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": topic_deduplication_schema
            },
            temperature=0.1,
            max_completion_tokens=8192,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        return json.loads(response.choices[0].message.content)
    
    #######################################################
    # Other helper functions
    #######################################################
    def _get_global_topic_key(self, level_key, label_idx):
        '''
        Get the global topic key based on the level key and label index.
        '''
        return f"L{level_key}_{label_idx}"
    
    def _get_delimiter(self, filename):
        """
        Get the delimiter of the file.
        """
        if filename.lower().endswith(".tsv"):
            return "\t"
        elif filename.lower().endswith(".csv"):
            return ","
        else:
            print(
                f"* ERROR: couldn't determine the delimiter for the embeddings file. Using default delimiter: \t"
            )
            return "\t"
    
    def _format_other_topics(self, other_labels, other_descriptions):
        """
        Format the other topics into a string, required by the deduplication prompt. 
        The format is:
        - label: X
          description: Y
        """
        lines = []
        for lbl, desc in zip(other_labels, other_descriptions):
            lines.append(f"- label: {lbl}\n  description: {desc}")
        return "\n".join(lines)
    
    def export_topic_annotations(
        self, 
        levels,
        path_output
    ):
        
        print(f"* dumping topic json to {path_output}")

        with open(path_output, "w") as f:
            json.dump(
                {"levels": levels},
                f,
                indent=2,
                default=lambda obj: obj.value if isinstance(obj, Enum) else obj,
            )

        print(f"* dumped all topics into {path_output}")      

    def get_levels_from_llm_annotations(self):
        
        levels = []

        # starting from the lowest level (most fine-grained level, i.e., 0)
        for level in sorted(self.labels_cache.keys()):
            
            # level-wise dictionary: creating the mapping dictionary: label (cluster index) to list of point indices (data points)
            cluster_point_indices = defaultdict(list)
            for i, label in enumerate(self.labels_cache[level]):
                cluster_point_indices[label].append(i)
            
            topics = []
            for label, indices in cluster_point_indices.items():
                
                # get the pids
                cluster_pmids = self.points_df.iloc[indices][self.config.point_identifier].tolist()
                # get the center for this cluster (label)
                if (center := self.centers_cache[level].loc[self.centers_cache[level][f"L{level}_clusters"] == label].iloc[0]) is None:
                    continue
                if cluster_pmids is None:
                    continue
            
                global_key = self._get_global_topic_key(level, label)

                _topic = {
                    "global_topic_key": global_key,
                    "name": self.llm_topic_cache[global_key],
                    "description": self.llm_topic_description_cache[global_key],
                    "counts": len(cluster_pmids),
                    "pids": cluster_pmids if self.config.flag_output_include_pids else [],
                }

                # add dimensions
                for dim in self.config.dimensions:
                    _topic[dim] = float(center[dim])
                
                # append topic
                topics.append(_topic)

            ret = {
                "level": level, 
                "topics": topics
            }

            levels.append(ret)
        
        return levels