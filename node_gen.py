import pandas as pd
import numpy as np
import networkx as nx
from ast import literal_eval
from pathlib import Path
from typing import List, Dict, Tuple, Set
import json
from pyvis.network import Network
import colorsys
import os
from node2vec import Node2Vec
import pickle
from config import TEST_MODE, USE_GPU, GPU_ID, NUM_WORKERS
import torch

class SessionGraphBuilder:
    def __init__(self, save_dir: str = "graph_data", test_mode: bool = False):
        """
        Initialize the session graph builder
        
        Args:
            save_dir: Directory to save graph data
            test_mode: If True, only process 1000 rows
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.graph = nx.DiGraph()  # Directed graph for sequential relationships
        self.processed_items: Set[str] = set()  # Track unique items processed
        self.item_limit = 1000 if test_mode else None  # Limit items only in test mode
        self.test_mode = test_mode
        
    def should_process_session(self, items: List[str]) -> bool:
        """
        Check if we should process this session based on item limits
        
        Args:
            items: List of item IDs in the session
        
        Returns:
            Boolean indicating whether to process this session
        """
        # If no limit or haven't reached limit, process all sessions
        if self.item_limit is None or len(self.processed_items) < self.item_limit:
            return True
        # In test mode with limit reached, only process if all items are known
        return all(item in self.processed_items for item in items)
    
    def process_session(self, session_items: List[str]) -> None:
        """
        Process a single session and update the graph
        
        Args:
            session_items: Complete list of items in session order
        """
        if not session_items or len(session_items) < 2:
            return
            
        if not self.should_process_session(session_items):
            return
            
        # Add all items as nodes if they don't exist
        self.graph.add_nodes_from(session_items)
        self.processed_items.update(session_items)
        
        # Add edges between consecutive items, skipping self-loops
        for i in range(len(session_items) - 1):
            source = session_items[i]
            target = session_items[i + 1]
            
            # Skip self-loops (connections to self)
            if source == target:
                continue
            
            # Check if reverse edge exists
            has_reverse_edge = self.graph.has_edge(target, source)
            
            # Add or update edge weight
            if self.graph.has_edge(source, target):
                # If this is a bidirectional connection, add +1 to both weights
                if has_reverse_edge:
                    self.graph[source][target]['weight'] += 2  # +1 for sequence, +1 for bidirectional
                    self.graph[target][source]['weight'] += 1  # +1 for bidirectional
                else:
                    self.graph[source][target]['weight'] += 1
            else:
                # If reverse edge exists, add weight +1, otherwise normal weight
                weight = 2 if has_reverse_edge else 1  # Start with 2 if bidirectional
                self.graph.add_edge(source, target, weight=weight)
                if has_reverse_edge:
                    self.graph[target][source]['weight'] += 1  # Add +1 to reverse edge
    
    def build_from_sessions(self, sessions_file: str, chunk_size: int = 10000) -> None:
        """
        Build graph from sessions data file
        
        Args:
            sessions_file: Path to sessions CSV file
            chunk_size: Number of rows to process at a time
        """
        print("Building graph from sessions...")
        total_chunks = 0
        total_sessions = 0
        valid_sessions = 0
        
        def clean_product_id(pid: str) -> str:
            """Clean a product ID string"""
            pid = str(pid).strip().strip("'").strip('"')
            return pid if len(pid) >= 10 and len(pid) <= 15 else None
        
        def parse_prev_items(prev_items_str: str) -> List[str]:
            """Parse the prev_items string into a list of product IDs"""
            try:
                # Remove brackets and split by space or comma
                items_str = prev_items_str.strip('[]').replace("'", "").replace('"', "")
                items = [item.strip() for item in items_str.split() if item.strip()]
                return [item for item in items if len(item) >= 10 and len(item) <= 15]
            except Exception as e:
                print(f"Error parsing prev_items: {e}")
                return []
        
        # Process file in chunks
        for chunk in pd.read_csv(sessions_file, chunksize=chunk_size):
            if self.test_mode and total_sessions >= 1000:
                print("Test mode: Reached 1000 rows limit")
                break
                
            if self.item_limit and len(self.processed_items) >= self.item_limit and not any(
                item in self.processed_items for item in chunk['next_item']
            ):
                # Skip chunk if we've reached limit and no items are in our processed set
                continue
                
            # Process each session
            for _, row in chunk.iterrows():
                if self.test_mode and total_sessions >= 1000:
                    break
                    
                total_sessions += 1
                try:
                    # Parse prev_items string into list
                    prev_items = parse_prev_items(row['prev_items'])
                    
                    # Clean next_item
                    next_item = clean_product_id(row['next_item'])
                    
                    if next_item and prev_items:
                        # Combine prev_items and next_item into a single sequence
                        session_items = prev_items + [next_item]
                        if len(session_items) >= 2:  # Need at least 2 items to create edges
                            self.process_session(session_items)
                            valid_sessions += 1
                            
                            # Debug output for first few valid sessions
                            if valid_sessions <= 5:
                                print(f"\nDebug - Session {valid_sessions}:")
                                print(f"Prev items: {prev_items}")
                                print(f"Next item: {next_item}")
                                print(f"Session items: {session_items}")
                            
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            total_chunks += 1
            print(f"Processed chunk {total_chunks}, unique items: {len(self.processed_items)}")
            print(f"Valid sessions: {valid_sessions}/{total_sessions}")
            
            # Stop if we've reached the item limit in test mode
            if self.test_mode and len(self.processed_items) >= self.item_limit:
                print(f"Test mode: Reached item limit of {self.item_limit}")
                break
        
        print("\nGraph statistics:")
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print(f"Total sessions processed: {total_sessions}")
        print(f"Valid sessions: {valid_sessions}")
        
        # Print some sample node IDs to verify format
        sample_nodes = list(self.graph.nodes())[:5]
        print("\nSample node IDs:")
        for node in sample_nodes:
            print(node)
    
    def get_node_features(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate node features from the graph
        
        Returns:
            Dictionary of node features
        """
        features = {}
        
        for node in self.graph.nodes():
            # Calculate various centrality measures
            in_degree = self.graph.in_degree(node, weight='weight')
            out_degree = self.graph.out_degree(node, weight='weight')
            
            features[node] = {
                'in_degree': float(in_degree),
                'out_degree': float(out_degree),
                'total_degree': float(in_degree + out_degree)
            }
        
        return features
    
    def save_graph_data(self, filename_prefix: str = "session") -> None:
        """
        Save graph data to files
        
        Args:
            filename_prefix: Prefix for saved files
        """
        # Save graph in GraphML format
        nx.write_graphml(self.graph, self.save_dir / f"{filename_prefix}_graph.graphml")
        
        # Save node features
        features = self.get_node_features()
        with open(self.save_dir / f"{filename_prefix}_features.json", 'w') as f:
            json.dump(features, f)
        
        # Save list of processed items
        with open(self.save_dir / f"{filename_prefix}_items.json", 'w') as f:
            json.dump(list(self.processed_items), f)
        
        print(f"\nSaved graph data to {self.save_dir}")
    
    def get_top_transitions(self, n: int = 10) -> List[Tuple[str, str, float]]:
        """
        Get top N transitions by weight
        
        Args:
            n: Number of top transitions to return
            
        Returns:
            List of (source, target, weight) tuples
        """
        edges = [(u, v, d['weight']) for u, v, d in self.graph.edges(data=True)]
        return sorted(edges, key=lambda x: x[2], reverse=True)[:n]

    def visualize_graph(self, filename: str = "session_graph.html", height: str = "750px", width: str = "100%"):
        """
        Create an interactive visualization of the graph
        
        Args:
            filename: Name of the output HTML file
            height: Height of the visualization
            width: Width of the visualization
        """
        # Create a Pyvis network
        net = Network(height=height, width=width, notebook=False, directed=True)
        
        # Calculate node sizes based on degree centrality
        degree_dict = dict(self.graph.degree(weight='weight'))
        max_degree = max(degree_dict.values()) if degree_dict and degree_dict.values() else 1
        
        # Calculate node colors based on in/out degree ratio
        in_degree_dict = dict(self.graph.in_degree(weight='weight'))
        out_degree_dict = dict(self.graph.out_degree(weight='weight'))
        
        def get_node_color(node):
            in_deg = in_degree_dict.get(node, 0)
            out_deg = out_degree_dict.get(node, 0)
            total = in_deg + out_deg
            if total == 0:
                return "#808080"  # Gray for isolated nodes
            # Use hue to represent in/out ratio (red for more incoming, blue for more outgoing)
            hue = 0.66 if out_deg > in_deg else 0.0
            saturation = min(abs(out_deg - in_deg) / total, 1.0)
            rgb = colorsys.hsv_to_rgb(hue, saturation, 0.9)
            return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
        
        # Add nodes with size and color based on properties
        for node in self.graph.nodes():
            size = 20 + (30 * degree_dict[node] / max_degree) if max_degree > 0 else 20
            color = get_node_color(node)
            title = f"ID: {node}<br>In-degree: {in_degree_dict.get(node, 0)}<br>Out-degree: {out_degree_dict.get(node, 0)}"
            net.add_node(node, size=size, color=color, title=title)
        
        # Add edges with width based on weight
        edge_weights = nx.get_edge_attributes(self.graph, 'weight')
        max_weight = max(edge_weights.values()) if edge_weights else 1
        
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            weight = data.get('weight', 1)
            width = 1 + (4 * weight / max_weight)
            title = f"Weight: {weight}"
            net.add_edge(source, target, width=width, title=title)
        
        # Configure physics
        net.set_options("""
        var options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
            },
            "edges": {
                "smooth": {"type": "continuous"},
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
            }
        }
        """)
        
        # Save the visualization
        output_path = self.save_dir / filename
        net.save_graph(str(output_path))
        print(f"\nVisualization saved to {output_path}")
        
        # Print debug information
        print("\nGraph Debug Information:")
        print(f"Total nodes: {len(self.graph.nodes())}")
        print(f"Total edges: {len(self.graph.edges())}")
        print(f"Nodes with edges: {len([n for n in self.graph.nodes() if self.graph.degree(n) > 0])}")
        print(f"Isolated nodes: {len([n for n in self.graph.nodes() if self.graph.degree(n) == 0])}")
        if edge_weights:
            print(f"Edge weight range: {min(edge_weights.values())} to {max(edge_weights.values())}")
        
        # Generate legend file
        legend_html = """
        <html>
        <head>
            <style>
                .legend-container {
                    font-family: Arial, sans-serif;
                    padding: 20px;
                    background-color: #f5f5f5;
                    border-radius: 5px;
                    margin: 20px;
                }
                .legend-item {
                    margin: 10px 0;
                }
                .color-box {
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    margin-right: 10px;
                    vertical-align: middle;
                }
            </style>
        </head>
        <body>
            <div class="legend-container">
                <h2>Graph Legend</h2>
                <div class="legend-item">
                    <div class="color-box" style="background-color: #ff0000;"></div>
                    More incoming connections (consumer nodes)
                </div>
                <div class="legend-item">
                    <div class="color-box" style="background-color: #0000ff;"></div>
                    More outgoing connections (source nodes)
                </div>
                <div class="legend-item">
                    <div class="color-box" style="background-color: #808080;"></div>
                    Balanced or isolated nodes
                </div>
                <p>
                    <strong>Node Size:</strong> Larger nodes have more connections<br>
                    <strong>Edge Width:</strong> Thicker edges represent more frequent transitions<br>
                    <strong>Hover:</strong> Hover over nodes and edges for detailed information
                </p>
            </div>
        </body>
        </html>
        """
        
        legend_path = self.save_dir / "graph_legend.html"
        with open(legend_path, 'w') as f:
            f.write(legend_html)
        print(f"Legend saved to {legend_path}")

    def train_node2vec(self, dimensions: int = 128, walk_length: int = 10, 
                      num_walks: int = 100, window: int = 5, 
                      min_count: int = 1, batch_words: int = 4) -> None:
        """
        Train Node2Vec model on the graph and save embeddings
        
        Args:
            dimensions: Embedding dimensions
            walk_length: Length of each random walk
            num_walks: Number of random walks per node
            window: Maximum distance between current and predicted word
            min_count: Minimum count of node occurrences
            batch_words: Number of words per batch
        """
        print("\nTraining Node2Vec model...")
        
        # Check if CUDA is available when GPU is requested
        device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print(f"Using GPU device {GPU_ID}")
            torch.cuda.set_device(GPU_ID)
        else:
            print("Using CPU for training")
        
        # Initialize Node2Vec model with GPU support
        node2vec = Node2Vec(
            graph=self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=NUM_WORKERS,  # Number of parallel workers
            device=device,  # Use GPU if available
            # Additional parameters for better GPU utilization
            p=1,  # Return parameter
            q=1,  # In-out parameter
            batch_walks=None,  # Walks per batch (None = automatically determine)
            seed=42,  # Random seed for reproducibility
            verbose=True  # Print progress
        )
        
        print("\nGenerating walks...")
        # Train the model
        model = node2vec.fit(
            window=window,
            min_count=min_count,
            batch_words=batch_words,
            seed=42,
            workers=NUM_WORKERS,  # Use multiple workers for training
            compute_loss=True  # Track training loss
        )
        
        # Move model to CPU for saving
        if device == 'cuda':
            model = model.to('cpu')
        
        # Save the complete model
        model_path = self.save_dir / "node2vec_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save word vectors separately for easier loading
        vectors_path = self.save_dir / "node2vec_vectors.pkl"
        vectors_dict = dict(zip(model.wv.index_to_key, model.wv.vectors))
        with open(vectors_path, 'wb') as f:
            pickle.dump(vectors_dict, f)
            
        print(f"\nModel saved to {model_path}")
        print(f"Vectors saved to {vectors_path}")
        
        # Print training statistics
        print("\nTraining Statistics:")
        print(f"Total nodes processed: {len(model.wv.index_to_key)}")
        print(f"Embedding dimensions: {dimensions}")
        if hasattr(model, 'running_training_loss'):
            print(f"Final training loss: {model.running_training_loss}")
        
        # Print example embeddings for first few nodes
        print("\nExample embeddings:")
        for node in list(self.graph.nodes())[:3]:
            print(f"Node {node} embedding shape: {model.wv[node].shape}")
            
    @staticmethod
    def load_node2vec_vectors(save_dir: str = "graph_data") -> Dict[str, np.ndarray]:
        """
        Load saved Node2Vec vectors
        
        Args:
            save_dir: Directory where vectors are saved
            
        Returns:
            Dictionary mapping node IDs to their vectors
        """
        vectors_path = Path(save_dir) / "node2vec_vectors.pkl"
        with open(vectors_path, 'rb') as f:
            vectors = pickle.load(f)
        print(f"\nLoaded {len(vectors)} node embeddings")
        return vectors

def main():
    # Initialize graph builder with test mode from environment
    builder = SessionGraphBuilder(test_mode=TEST_MODE)
    
    # Build graph from sessions
    builder.build_from_sessions('sessions_train.csv')
    
    # Print top transitions
    print("\nTop 10 item transitions:")
    for source, target, weight in builder.get_top_transitions(10):
        print(f"{source} -> {target}: {weight}")
    
    # Save graph data
    builder.save_graph_data()
    
    # Create visualization
    builder.visualize_graph()
    
    # Train and save Node2Vec model
    builder.train_node2vec()
    
    # Example of loading and using vectors
    vectors = SessionGraphBuilder.load_node2vec_vectors()
    print("\nLoaded vectors for", len(vectors), "nodes")
    if vectors:
        sample_node = next(iter(vectors))
        print(f"Sample vector shape for node {sample_node}: {vectors[sample_node].shape}")

if __name__ == "__main__":
    main()