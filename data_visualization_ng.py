import faiss
import pickle
import numpy as np
import networkx as nx
from pyvis.network import Network
from tqdm import tqdm
import random
import logging
from community import community_louvain  # For community detection

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting the process...")

# Load the embeddings and paths
logger.info("Loading embeddings and image paths...")
with open("pkl_files/embeddings_with_ids.pkl", "rb") as f:
    embeddings_with_ids = pickle.load(f)

# Extract embeddings, IDs, and paths
logger.info("Extracting embeddings, IDs, and image paths...")
embeddings = np.array([embedding for _, embedding, _ in embeddings_with_ids])
unique_ids = [unique_id for unique_id, _, img_path in embeddings_with_ids]
image_paths = [img_path for _, _, img_path in embeddings_with_ids]

# Normalize embeddings
logger.info("Normalizing embeddings...")
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Initialize Faiss index
logger.info("Initializing Faiss index...")
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatIP(res, embeddings.shape[1])
index.add(embeddings)

# Perform the search
logger.info("Performing similarity search...")
k = 5
distances, indices = index.search(embeddings, k + 1)

# Create the graph
logger.info("Creating the graph...")
G = nx.Graph()

# Add nodes with image paths to the graph
logger.info("Adding nodes to the graph...")
for unique_id, img_path in zip(unique_ids, image_paths):
    # Ensure that the image paths are correctly formatted for your environment
    img_tag = f"<img src='{img_path}' width='150' height='150'>"
    G.add_node(unique_id, title=img_tag)

# Add edges based on similarity
logger.info("Adding edges to the graph...")
for i in tqdm(range(len(unique_ids)), desc="Processing embeddings", unit="embedding"):
    unique_id = unique_ids[i]
    for j in range(1, k + 1):
        neighbor_idx = indices[i, j]
        neighbor_id = unique_ids[neighbor_idx]
        similarity = float(distances[i, j])
        if similarity > 0.8:
            G.add_edge(unique_id, neighbor_id, weight=similarity)

# Community detection using the Louvain method
logger.info("Detecting communities using the Louvain method...")
communities = community_louvain.best_partition(G)

# Assign colors to communities
logger.info("Assigning colors to communities...")
community_colors = {}
for node, community in communities.items():
    if community not in community_colors:
        community_colors[community] = (
            f"#{random.randint(0, 0xFFFFFF):06x}"  # Random color in hex
        )
    G.nodes[node]["color"] = community_colors[community]

# Sampling nodes from each community
logger.info("Sampling nodes from each community...")
sampled_nodes = set()
sample_size_per_community = 100  # Adjust as needed

for community in set(communities.values()):
    community_nodes = [node for node, comm in communities.items() if comm == community]
    sampled_nodes.update(
        random.sample(
            community_nodes, min(len(community_nodes), sample_size_per_community)
        )
    )

# Get the sampled subgraph
G_sub = G.subgraph(sampled_nodes)

# Initialize Pyvis Network with fixed size
logger.info("Initializing Pyvis network...")
fixed_width = "1200px"  # Set the fixed width
fixed_height = "800px"  # Set the fixed height
net = Network(
    notebook=False,
    width=fixed_width,
    height=fixed_height,
    bgcolor="#222222",
    font_color="white",
)

# Use forceAtlas2Based for better layout
logger.info("Setting up the forceAtlas2Based layout...")
net.force_atlas_2based(
    gravity=-50,
    central_gravity=0.005,
    spring_length=100,
    spring_strength=0.1,
    damping=0.4,
)

# Customize nodes and edges
logger.info("Customizing nodes and edges...")
net.set_options(
    """
var options = {
  "nodes": {
    "borderWidth": 2,
    "size": 10,
    "color": {"inherit": "color"},
    "font": {"color": "#ffffff"}
  },
  "edges": {
    "color": {"inherit": true},
    "smooth": {"type": "continuous"},
    "arrows": {"to": {"enabled": false}}
  },
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.005,
      "springLength": 100,
      "springConstant": 0.1,
      "damping": 0.4
    },
    "minVelocity": 0.75
  }
}
"""
)

# Add nodes and edges to the Pyvis network
logger.info("Adding nodes and edges to Pyvis network...")
net.from_nx(G_sub)

# Save and display the graph
logger.info("Rendering and saving the graph as 'sampled_image_graph.html'...")
net.save_graph("sampled_image_graph.html")

logger.info("Process completed.")
