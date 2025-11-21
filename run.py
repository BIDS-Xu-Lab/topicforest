import os
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

from topicforest.config import (
    HierarchicalClusterConfig,
    HierarchicalTopicAnnotatorConfig,
)
from topicforest import HierarchicalTopicAnnotator

# Load environment variables from .env file
load_dotenv()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run TopicForest on a dataset")
    
    # Input file
    parser.add_argument("--path_tsv", type=str, required=True, 
                        help="Path to the TSV file containing the data")
    
    # Data Config
    parser.add_argument("--point_identifier", type=str, default="pid",
                        help="Column name for the identifier of each data point")
    parser.add_argument("--dimensions", type=str, default="x,y",
                        help="Comma-separated list of dimension column names")    
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory to save output files (default: same as input)")

    # Hierarchical Cluster Config
    parser.add_argument("--L", type=int, default=3,
                        help="Number of layers in the hierarchy")
    parser.add_argument("--k_top_layer", type=int, default=30,
                        help="Number of clusters in the top layer")
    parser.add_argument("--k_lowest_layer", type=int, default=300,
                        help="Number of clusters in the lowest layer")
    parser.add_argument("--linkage", type=str, default="ward",
                        help="Linkage method for hierarchical clustering")
    parser.add_argument("--metric", type=str, default="euclidean",
                        help="Distance metric for hierarchical clustering")

    # LLM-based recursive labeling Config
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="Model name for the LLM-based recursive labeling. We only test on gpt-4o-mini and gpt-4.1-nano.")
    parser.add_argument("--deduplicate_topic_labels", action="store_true",
                        help="Deduplicate topic labels in each level of the topic tree. This is an experimental feature, and we found it is useful for deduplicating topic labels in each level of the topic tree.")

    return parser.parse_args()


def get_output_paths(path_tsv, output_dir=None):
    """Generate output file paths based on the input TSV path"""
    input_path = Path(path_tsv)
    base_name = input_path.stem
    
    # Use output_dir if provided, otherwise use the same directory as the input file
    output_dir = Path(output_dir) if output_dir else input_path.parent
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_cluster_assignment_path = output_dir / f"{base_name}.cluster_assignments.tsv"
    output_topic_path = output_dir / f"{base_name}.topics.json"
    
    return str(output_cluster_assignment_path), str(output_topic_path)

def main():
    args = parse_args()
    
    # Get the OpenAI API key from the environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if openai_api_key:
        print("Using OpenAI API key from .env file")
    else:
        print("No OpenAI API key found in .env file. Please set the OPENAI_API_KEY in the .env file.")
        exit(1)
    
    # Create cluster config
    cluster_config = HierarchicalClusterConfig(
        L=args.L,
        k_top_layer=args.k_top_layer,
        k_lowest_layer=args.k_lowest_layer,
        linkage=args.linkage,
        metric=args.metric
    )
    
    # Create annotator config
    annotator_config = HierarchicalTopicAnnotatorConfig(
        point_identifier=args.point_identifier,
        dimensions=args.dimensions.split(','),
        topic_cluster_configs=cluster_config,
        openAI_api_key=openai_api_key
    )
    
    # Get output paths
    output_cluster_assignment_path, output_topic_path = get_output_paths(args.path_tsv, args.output_dir)
    
    # Initialize annotator
    print(f"Initializing TopicForest with {args.path_tsv}")
    hierarchical_annotator = HierarchicalTopicAnnotator(
        annotator_config,
        path_tsv=args.path_tsv,
        model_name=args.model_name
    )
    
    # Perform clustering and build topic tree
    print("Performing hierarchical clustering...")
    start_time = time.time()
    hierarchical_annotator.cluster()
    hierarchical_annotator.build_topic_tree()
    print(f"Clustering completed in {time.time() - start_time:.2f} seconds")
    
    # Perform LLM-based recursive labeling
    print("Performing LLM-based recursive labeling...")
    start_time = time.time()
    llm_levels = hierarchical_annotator.get_llm_based_topic_annotation()
    print(f"LLM-based recursive labeling completed in {time.time() - start_time:.2f} seconds")

    if args.deduplicate_topic_labels:
        print("Deduplicating topic labels in each level of the topic tree...")
        start_time = time.time()
        for level in range(0, hierarchical_annotator.L):
            hierarchical_annotator.level_wise_label_deduplication(level)
        print(f"Topic label deduplication completed in {time.time() - start_time:.2f} seconds")
        llm_levels = hierarchical_annotator.get_levels_from_llm_annotations()

    # Export the cluster assignments and topic labels
    print("Exporting the results...")
    hierarchical_annotator.points_df[[annotator_config.point_identifier] + [f'L{i}_clusters' for i in range(hierarchical_annotator.L)]].to_csv(output_cluster_assignment_path, sep='\t', index=False)
    hierarchical_annotator.export_topic_annotations(llm_levels, output_topic_path)
    print(f"Cluster assignments exported to: {output_cluster_assignment_path}")
    print(f"Topic labels and descriptions exported to: {output_topic_path}")

if __name__ == "__main__":
    main()