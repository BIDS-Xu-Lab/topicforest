from dataclasses import dataclass, field
from typing import List

@dataclass
class BaseClusterConfig:
    '''
    Base configuration for the clustering
    '''
    L: int # number of layers
    k_top_layer: int # number of clusters in the top layer
    k_lowest_layer: int # number of clusters in the lowest layer

@dataclass
class HierarchicalClusterConfig(BaseClusterConfig):
    '''
    Configuration for the hierarchical clustering in HierarchicalTopicAnnotator
    '''
    linkage: str = 'ward' # e.g. 'ward', 'average', 'complete'
    metric: str = 'euclidean' # e.g. 'euclidean', 'cosine'

@dataclass
class BaseTopicAnnotatorConfig:
    '''
    Base configuration for the topic annotator
    '''
    point_identifier: str = 'pid' # e.g. 'pmid', 'pid', 'doi'
    dimensions: List[str] = field(default_factory=list) # e.g. ['x', 'y', 'z'] or ['x', 'y']
    flag_output_include_pids: bool = True # e.g. True, False
    openAI_api_key: str = ''

@dataclass
class HierarchicalTopicAnnotatorConfig(BaseTopicAnnotatorConfig):
    '''
    Configuration for the HierarchicalTopicAnnotator
    '''
    topic_cluster_configs: HierarchicalClusterConfig = field(default_factory=dict)