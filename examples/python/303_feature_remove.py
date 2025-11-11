#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pyvsag
import numpy as np
import json
import sys

def print_results(title, ids, dists):
    """Helper function to print search results."""
    print(f"\n--- {title} ---")
    if len(ids) == 0:
        print("No results found.")
        return
        
    for i, (id_val, dist_val) in enumerate(zip(ids, dists)):
        print(f"Rank {i}: ID={id_val}, Dist={dist_val:.4f}")

def main():
    pyvsag.init()

    # 1. Prepare Base Dataset
    num_vectors = 2000
    dim = 128
    
    # Use int64 for IDs to match C++ and binding
    ids = np.arange(num_vectors, dtype=np.int64)
    
    # Use float32 for vectors
    np.random.seed(47) # Match C++ rng(47)
    data = np.random.random((num_vectors, dim)).astype(np.float32)

    # 2. Create HGraph Index with remove support enabled
    # This is the most critical step for this example.
    hgraph_build_parameters = json.dumps({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": dim,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 16,
            "ef_construction": 100,
            "support_remove": True  # Enable remove functionality
        }
    })

    print("Creating HGraph index with remove support...")
    index = pyvsag.Index("hgraph", hgraph_build_parameters)

    # 3. Build HGraph Index
    try:
        index.build(vectors=data, ids=ids)
        print(f"Index built. Num elements: {index.num_elements}")
    except pyvsag.VsagError as e:
        print(f"Failed to build index: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Prepare Query
    # Generate the same query vector as in the C++ example
    query_vector = np.random.random(dim).astype(np.float32)

    # 5. HGraph Origin KnnSearch
    hgraph_search_parameters = json.dumps({
        "hgraph": {
            "ef_search": 100
        }
    })
    topk = 10
    
    try:
        ids_before, dists_before = index.knn_search(
            vector=query_vector, 
            k=topk, 
            parameters=hgraph_search_parameters
        )
        print_results("Original Search Results", ids_before, dists_before)
    except pyvsag.VsagError as e:
        print(f"Failed to search index: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 6. HGraph Remove Some result ids
    # Remove the top 5 results from the original search
    ids_to_remove = ids_before[:5]
    print(f"\nRemoving top 5 IDs: {list(ids_to_remove)}...")
    
    removed_count = 0
    try:
        for _id in ids_to_remove:
            index.remove(_id)
            removed_count += 1
        print(f"Successfully removed {removed_count} items.")
        print(f"Index num_elements after remove: {index.num_elements}")
    except pyvsag.VsagError as e:
        print(f"Error during remove: {e}", file=sys.stderr)
        # Continue to search even if some removes failed, to see the state
    
    if removed_count == 0:
        print("Warning: No IDs were removed, search results will be identical.")

    # 7. HGraph KnnSearch After Remove
    # Perform the exact same search again
    ids_after, dists_after = index.knn_search(
        vector=query_vector, 
        k=topk, 
        parameters=hgraph_search_parameters
    )
    print_results("Search Results After Remove", ids_after, dists_after)

    # 8. Verification
    removed_set = set(ids_to_remove)
    after_set = set(ids_after)
    intersection = removed_set.intersection(after_set)
    
    if not intersection:
        print("\nVerification SUCCESS: Removed IDs are not present in new results.")
    else:
        print(f"\nVerification FAILED: Found removed IDs in new results: {intersection}")

if __name__ == '__main__':
    main()