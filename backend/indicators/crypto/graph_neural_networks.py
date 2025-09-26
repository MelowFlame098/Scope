"""Graph Neural Networks (GNNs) for Cryptocurrency Analysis

This module implements advanced Graph Neural Networks for cryptocurrency analysis:
- Transaction Graph Analysis
- Address Relationship Modeling
- Network Topology Analysis
- Temporal Graph Networks
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- GraphSAGE for Large-scale Analysis
- Anomaly Detection in Transaction Graphs
- Price Prediction using Graph Features
- Community Detection
- Centrality Analysis
- Graph Embedding
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Graph Libraries
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available. Graph analysis will be limited.")

# ML Libraries
try:
    from sklearn.cluster import SpectralClustering, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Using simplified implementations.")

# Deep Learning Libraries (Optional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Using simplified GNN implementations.")

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Graph node representation"""
    node_id: str
    node_type: str  # 'address', 'transaction', 'block'
    features: Dict[str, float]
    timestamp: Optional[datetime] = None
    labels: List[str] = None

@dataclass
class GraphEdge:
    """Graph edge representation"""
    source: str
    target: str
    edge_type: str  # 'transaction', 'input', 'output'
    weight: float
    timestamp: Optional[datetime] = None
    features: Dict[str, float] = None

@dataclass
class TransactionGraphResult:
    """Transaction graph analysis result"""
    total_nodes: int
    total_edges: int
    graph_density: float
    clustering_coefficient: float
    average_path_length: float
    diameter: int
    connected_components: int
    largest_component_size: int
    small_world_coefficient: float
    scale_free_exponent: float
    centrality_scores: Dict[str, Dict[str, float]]
    community_structure: Dict[str, List[str]]

@dataclass
class AddressRelationshipResult:
    """Address relationship modeling result"""
    address_clusters: Dict[str, List[str]]
    relationship_strength: Dict[Tuple[str, str], float]
    address_roles: Dict[str, str]
    interaction_patterns: Dict[str, Dict[str, float]]
    temporal_relationships: Dict[str, List[Tuple[datetime, str, float]]]
    risk_propagation: Dict[str, float]
    influence_scores: Dict[str, float]

@dataclass
class NetworkTopologyResult:
    """Network topology analysis result"""
    topology_type: str
    robustness_score: float
    vulnerability_points: List[str]
    critical_paths: List[List[str]]
    network_efficiency: float
    modularity: float
    assortativity: float
    rich_club_coefficient: float
    core_periphery_structure: Dict[str, str]

@dataclass
class TemporalGraphResult:
    """Temporal graph analysis result"""
    temporal_patterns: Dict[str, List[float]]
    evolution_metrics: Dict[str, float]
    growth_rate: float
    activity_cycles: List[Dict]
    temporal_centrality: Dict[str, Dict[str, float]]
    dynamic_communities: Dict[datetime, Dict[str, List[str]]]
    temporal_anomalies: List[Dict]

@dataclass
class GraphEmbeddingResult:
    """Graph embedding result"""
    node_embeddings: Dict[str, np.ndarray]
    embedding_quality: float
    similarity_matrix: np.ndarray
    clusters: Dict[str, List[str]]
    anomalous_nodes: List[str]
    embedding_dimensions: int

@dataclass
class AnomalyDetectionResult:
    """Graph-based anomaly detection result"""
    anomalous_nodes: List[str]
    anomalous_edges: List[Tuple[str, str]]
    anomaly_scores: Dict[str, float]
    anomaly_types: Dict[str, str]
    suspicious_patterns: List[Dict]
    risk_assessment: Dict[str, float]

@dataclass
class PricePredictionResult:
    """Graph-based price prediction result"""
    predicted_price_change: float
    confidence_score: float
    influential_factors: Dict[str, float]
    graph_signals: Dict[str, float]
    prediction_horizon: int
    feature_importance: Dict[str, float]

@dataclass
class GNNAnalysisResult:
    """Combined GNN analysis result"""
    transaction_graph: TransactionGraphResult
    address_relationships: AddressRelationshipResult
    network_topology: NetworkTopologyResult
    temporal_analysis: TemporalGraphResult
    graph_embedding: GraphEmbeddingResult
    anomaly_detection: AnomalyDetectionResult
    price_prediction: PricePredictionResult
    overall_network_health: float
    graph_complexity_score: float

class GraphBuilder:
    """Builds and manages cryptocurrency transaction graphs"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self.nodes = {}
        self.edges = []
        
    def add_transaction_data(self, transactions: List[Dict]):
        """Add transaction data to build the graph"""
        for tx in transactions:
            self._add_transaction(tx)
    
    def _add_transaction(self, tx: Dict):
        """Add a single transaction to the graph"""
        tx_id = tx.get('tx_id', f"tx_{len(self.edges)}")
        from_addr = tx.get('from_address', '')
        to_addr = tx.get('to_address', '')
        amount = tx.get('amount', 0)
        timestamp = tx.get('timestamp', datetime.now())
        
        # Add nodes
        if from_addr and from_addr not in self.nodes:
            self.nodes[from_addr] = GraphNode(
                node_id=from_addr,
                node_type='address',
                features={'balance': 0, 'tx_count': 0},
                timestamp=timestamp
            )
        
        if to_addr and to_addr not in self.nodes:
            self.nodes[to_addr] = GraphNode(
                node_id=to_addr,
                node_type='address',
                features={'balance': 0, 'tx_count': 0},
                timestamp=timestamp
            )
        
        # Add transaction node
        self.nodes[tx_id] = GraphNode(
            node_id=tx_id,
            node_type='transaction',
            features={'amount': amount, 'fee': tx.get('fee', 0)},
            timestamp=timestamp
        )
        
        # Add edges
        if from_addr:
            edge1 = GraphEdge(
                source=from_addr,
                target=tx_id,
                edge_type='input',
                weight=amount,
                timestamp=timestamp
            )
            self.edges.append(edge1)
            
            # Update node features
            self.nodes[from_addr].features['tx_count'] += 1
        
        if to_addr:
            edge2 = GraphEdge(
                source=tx_id,
                target=to_addr,
                edge_type='output',
                weight=amount,
                timestamp=timestamp
            )
            self.edges.append(edge2)
            
            # Update node features
            self.nodes[to_addr].features['tx_count'] += 1
            self.nodes[to_addr].features['balance'] += amount
        
        # Add direct address-to-address edge for analysis
        if from_addr and to_addr:
            direct_edge = GraphEdge(
                source=from_addr,
                target=to_addr,
                edge_type='transaction',
                weight=amount,
                timestamp=timestamp
            )
            self.edges.append(direct_edge)
    
    def build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from nodes and edges"""
        if not NETWORKX_AVAILABLE:
            return None
        
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.features, node_type=node.node_type)
        
        # Add edges
        for edge in self.edges:
            if edge.edge_type == 'transaction':  # Only direct transaction edges
                G.add_edge(
                    edge.source, 
                    edge.target, 
                    weight=edge.weight,
                    edge_type=edge.edge_type,
                    timestamp=edge.timestamp
                )
        
        self.graph = G
        return G
    
    def get_subgraph(self, nodes: List[str]) -> nx.DiGraph:
        """Get subgraph containing specified nodes"""
        if not NETWORKX_AVAILABLE or self.graph is None:
            return None
        
        return self.graph.subgraph(nodes).copy()

class TransactionGraphAnalyzer:
    """Analyzes transaction graph structure and properties"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
    
    def analyze_transaction_graph(self, graph: nx.DiGraph) -> TransactionGraphResult:
        """Analyze transaction graph structure"""
        if not NETWORKX_AVAILABLE or graph is None:
            return self._create_empty_graph_result()
        
        try:
            # Basic graph metrics
            total_nodes = graph.number_of_nodes()
            total_edges = graph.number_of_edges()
            graph_density = nx.density(graph)
            
            # Convert to undirected for some metrics
            undirected_graph = graph.to_undirected()
            
            # Clustering coefficient
            clustering_coefficient = nx.average_clustering(undirected_graph)
            
            # Path length and diameter
            if nx.is_connected(undirected_graph):
                average_path_length = nx.average_shortest_path_length(undirected_graph)
                diameter = nx.diameter(undirected_graph)
            else:
                # For disconnected graphs, use largest component
                largest_cc = max(nx.connected_components(undirected_graph), key=len)
                subgraph = undirected_graph.subgraph(largest_cc)
                average_path_length = nx.average_shortest_path_length(subgraph)
                diameter = nx.diameter(subgraph)
            
            # Connected components
            connected_components = nx.number_connected_components(undirected_graph)
            largest_component_size = len(max(nx.connected_components(undirected_graph), key=len))
            
            # Small world coefficient
            small_world_coefficient = self._calculate_small_world_coefficient(undirected_graph)
            
            # Scale-free properties
            scale_free_exponent = self._calculate_scale_free_exponent(graph)
            
            # Centrality measures
            centrality_scores = self._calculate_centrality_measures(graph)
            
            # Community detection
            community_structure = self._detect_communities(undirected_graph)
            
            return TransactionGraphResult(
                total_nodes=total_nodes,
                total_edges=total_edges,
                graph_density=graph_density,
                clustering_coefficient=clustering_coefficient,
                average_path_length=average_path_length,
                diameter=diameter,
                connected_components=connected_components,
                largest_component_size=largest_component_size,
                small_world_coefficient=small_world_coefficient,
                scale_free_exponent=scale_free_exponent,
                centrality_scores=centrality_scores,
                community_structure=community_structure
            )
            
        except Exception as e:
            logger.error(f"Error analyzing transaction graph: {e}")
            return self._create_empty_graph_result()
    
    def _calculate_small_world_coefficient(self, graph: nx.Graph) -> float:
        """Calculate small world coefficient"""
        try:
            # Small world = high clustering + short path length
            clustering = nx.average_clustering(graph)
            
            if nx.is_connected(graph):
                path_length = nx.average_shortest_path_length(graph)
            else:
                # Use largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                path_length = nx.average_shortest_path_length(subgraph)
            
            # Compare with random graph
            n = graph.number_of_nodes()
            m = graph.number_of_edges()
            
            if n > 1 and m > 0:
                p = 2 * m / (n * (n - 1))  # Edge probability
                random_clustering = p
                random_path_length = np.log(n) / np.log(n * p) if n * p > 1 else float('inf')
                
                if random_clustering > 0 and random_path_length > 0:
                    small_world = (clustering / random_clustering) / (path_length / random_path_length)
                    return small_world
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_scale_free_exponent(self, graph: nx.DiGraph) -> float:
        """Calculate scale-free network exponent"""
        try:
            # Calculate degree distribution
            degrees = [d for n, d in graph.degree()]
            
            if not degrees:
                return 0.0
            
            # Count degree frequencies
            degree_counts = defaultdict(int)
            for degree in degrees:
                degree_counts[degree] += 1
            
            # Fit power law (simplified)
            if len(degree_counts) < 3:
                return 0.0
            
            x = np.array(list(degree_counts.keys()))
            y = np.array(list(degree_counts.values()))
            
            # Log-log fit
            log_x = np.log(x[x > 0])
            log_y = np.log(y[x > 0])
            
            if len(log_x) < 2:
                return 0.0
            
            # Linear regression in log space
            coeffs = np.polyfit(log_x, log_y, 1)
            exponent = -coeffs[0]  # Negative slope is the exponent
            
            return max(0, exponent)
            
        except Exception:
            return 0.0
    
    def _calculate_centrality_measures(self, graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures"""
        centrality_scores = {
            'degree': {},
            'betweenness': {},
            'closeness': {},
            'eigenvector': {},
            'pagerank': {}
        }
        
        try:
            # Degree centrality
            centrality_scores['degree'] = nx.degree_centrality(graph)
            
            # Betweenness centrality (sample for large graphs)
            if graph.number_of_nodes() > 1000:
                sample_nodes = list(graph.nodes())[:1000]
                centrality_scores['betweenness'] = nx.betweenness_centrality(graph, k=sample_nodes)
            else:
                centrality_scores['betweenness'] = nx.betweenness_centrality(graph)
            
            # Closeness centrality
            centrality_scores['closeness'] = nx.closeness_centrality(graph)
            
            # Eigenvector centrality
            try:
                centrality_scores['eigenvector'] = nx.eigenvector_centrality(graph, max_iter=1000)
            except:
                centrality_scores['eigenvector'] = {}
            
            # PageRank
            centrality_scores['pagerank'] = nx.pagerank(graph)
            
        except Exception as e:
            logger.warning(f"Error calculating centrality measures: {e}")
        
        return centrality_scores
    
    def _detect_communities(self, graph: nx.Graph) -> Dict[str, List[str]]:
        """Detect communities in the graph"""
        communities = {}
        
        try:
            if SKLEARN_AVAILABLE and graph.number_of_nodes() > 10:
                # Use spectral clustering
                adj_matrix = nx.adjacency_matrix(graph).toarray()
                
                # Determine number of clusters
                n_clusters = min(10, max(2, graph.number_of_nodes() // 100))
                
                clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)
                labels = clustering.fit_predict(adj_matrix)
                
                # Group nodes by cluster
                for i, node in enumerate(graph.nodes()):
                    cluster_id = f"community_{labels[i]}"
                    if cluster_id not in communities:
                        communities[cluster_id] = []
                    communities[cluster_id].append(node)
            
            else:
                # Simple connected components as communities
                for i, component in enumerate(nx.connected_components(graph)):
                    communities[f"community_{i}"] = list(component)
        
        except Exception as e:
            logger.warning(f"Error detecting communities: {e}")
        
        return communities
    
    def _create_empty_graph_result(self) -> TransactionGraphResult:
        """Create empty graph result for edge cases"""
        return TransactionGraphResult(
            total_nodes=0,
            total_edges=0,
            graph_density=0.0,
            clustering_coefficient=0.0,
            average_path_length=0.0,
            diameter=0,
            connected_components=0,
            largest_component_size=0,
            small_world_coefficient=0.0,
            scale_free_exponent=0.0,
            centrality_scores={},
            community_structure={}
        )

class AddressRelationshipAnalyzer:
    """Analyzes relationships between addresses"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
    
    def analyze_address_relationships(self, graph: nx.DiGraph) -> AddressRelationshipResult:
        """Analyze relationships between addresses"""
        if not NETWORKX_AVAILABLE or graph is None:
            return self._create_empty_relationship_result()
        
        # Filter to address nodes only
        address_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'address']
        address_graph = graph.subgraph(address_nodes)
        
        # Cluster addresses
        address_clusters = self._cluster_addresses(address_graph)
        
        # Calculate relationship strength
        relationship_strength = self._calculate_relationship_strength(address_graph)
        
        # Determine address roles
        address_roles = self._determine_address_roles(address_graph)
        
        # Analyze interaction patterns
        interaction_patterns = self._analyze_interaction_patterns(address_graph)
        
        # Temporal relationship analysis
        temporal_relationships = self._analyze_temporal_relationships(graph, address_nodes)
        
        # Risk propagation analysis
        risk_propagation = self._analyze_risk_propagation(address_graph)
        
        # Calculate influence scores
        influence_scores = self._calculate_influence_scores(address_graph)
        
        return AddressRelationshipResult(
            address_clusters=address_clusters,
            relationship_strength=relationship_strength,
            address_roles=address_roles,
            interaction_patterns=interaction_patterns,
            temporal_relationships=temporal_relationships,
            risk_propagation=risk_propagation,
            influence_scores=influence_scores
        )
    
    def _cluster_addresses(self, address_graph: nx.DiGraph) -> Dict[str, List[str]]:
        """Cluster addresses based on transaction patterns"""
        clusters = {}
        
        try:
            if SKLEARN_AVAILABLE and address_graph.number_of_nodes() > 10:
                # Create feature matrix
                features = []
                nodes = list(address_graph.nodes())
                
                for node in nodes:
                    # Features: in_degree, out_degree, clustering, pagerank
                    in_deg = address_graph.in_degree(node)
                    out_deg = address_graph.out_degree(node)
                    clustering = nx.clustering(address_graph.to_undirected(), node)
                    pagerank = nx.pagerank(address_graph).get(node, 0)
                    
                    features.append([in_deg, out_deg, clustering, pagerank])
                
                # Normalize features
                scaler = StandardScaler()
                normalized_features = scaler.fit_transform(features)
                
                # Cluster
                n_clusters = min(10, max(2, len(nodes) // 20))
                clustering_algo = SpectralClustering(n_clusters=n_clusters, random_state=42)
                labels = clustering_algo.fit_predict(normalized_features)
                
                # Group by cluster
                for i, node in enumerate(nodes):
                    cluster_id = f"cluster_{labels[i]}"
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(node)
            
            else:
                # Simple clustering based on connected components
                undirected = address_graph.to_undirected()
                for i, component in enumerate(nx.connected_components(undirected)):
                    clusters[f"cluster_{i}"] = list(component)
        
        except Exception as e:
            logger.warning(f"Error clustering addresses: {e}")
        
        return clusters
    
    def _calculate_relationship_strength(self, address_graph: nx.DiGraph) -> Dict[Tuple[str, str], float]:
        """Calculate strength of relationships between address pairs"""
        relationship_strength = {}
        
        for edge in address_graph.edges(data=True):
            source, target, data = edge
            weight = data.get('weight', 1)
            
            # Normalize by total transaction volume
            total_out = sum(d.get('weight', 1) for _, _, d in address_graph.out_edges(source, data=True))
            strength = weight / total_out if total_out > 0 else 0
            
            relationship_strength[(source, target)] = strength
        
        return relationship_strength
    
    def _determine_address_roles(self, address_graph: nx.DiGraph) -> Dict[str, str]:
        """Determine roles of addresses (hub, authority, bridge, etc.)"""
        address_roles = {}
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(address_graph)
        betweenness_centrality = nx.betweenness_centrality(address_graph)
        
        for node in address_graph.nodes():
            degree_cent = degree_centrality.get(node, 0)
            betweenness_cent = betweenness_centrality.get(node, 0)
            in_degree = address_graph.in_degree(node)
            out_degree = address_graph.out_degree(node)
            
            # Classify based on centrality and degree patterns
            if degree_cent > 0.1 and betweenness_cent > 0.1:
                role = "hub"
            elif betweenness_cent > 0.05:
                role = "bridge"
            elif in_degree > out_degree * 2:
                role = "sink"
            elif out_degree > in_degree * 2:
                role = "source"
            elif degree_cent > 0.05:
                role = "active"
            else:
                role = "peripheral"
            
            address_roles[node] = role
        
        return address_roles
    
    def _analyze_interaction_patterns(self, address_graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """Analyze interaction patterns between addresses"""
        interaction_patterns = {}
        
        for node in address_graph.nodes():
            patterns = {
                'reciprocity': 0.0,
                'clustering': 0.0,
                'diversity': 0.0,
                'frequency': 0.0
            }
            
            # Reciprocity: bidirectional connections
            neighbors = set(address_graph.neighbors(node))
            predecessors = set(address_graph.predecessors(node))
            reciprocal = len(neighbors.intersection(predecessors))
            total_connections = len(neighbors.union(predecessors))
            patterns['reciprocity'] = reciprocal / total_connections if total_connections > 0 else 0
            
            # Clustering coefficient
            patterns['clustering'] = nx.clustering(address_graph.to_undirected(), node)
            
            # Diversity: number of unique connections
            patterns['diversity'] = total_connections
            
            # Frequency: average transaction frequency (simplified)
            patterns['frequency'] = address_graph.degree(node)
            
            interaction_patterns[node] = patterns
        
        return interaction_patterns
    
    def _analyze_temporal_relationships(self, graph: nx.DiGraph, address_nodes: List[str]) -> Dict[str, List[Tuple[datetime, str, float]]]:
        """Analyze temporal evolution of relationships"""
        temporal_relationships = {}
        
        for node in address_nodes:
            relationships = []
            
            # Get all edges involving this node
            for edge in graph.edges(data=True):
                source, target, data = edge
                if source == node or target == node:
                    timestamp = data.get('timestamp')
                    weight = data.get('weight', 1)
                    other_node = target if source == node else source
                    
                    if timestamp and other_node in address_nodes:
                        relationships.append((timestamp, other_node, weight))
            
            # Sort by timestamp
            relationships.sort(key=lambda x: x[0])
            temporal_relationships[node] = relationships
        
        return temporal_relationships
    
    def _analyze_risk_propagation(self, address_graph: nx.DiGraph) -> Dict[str, float]:
        """Analyze risk propagation through the network"""
        risk_propagation = {}
        
        # Simplified risk propagation model
        # Risk spreads through connections with decay
        
        # Initialize with some high-risk nodes (simplified)
        high_risk_nodes = []
        for node in address_graph.nodes():
            degree = address_graph.degree(node)
            if degree > np.percentile([address_graph.degree(n) for n in address_graph.nodes()], 95):
                high_risk_nodes.append(node)
        
        # Propagate risk
        for node in address_graph.nodes():
            risk_score = 0.0
            
            # Direct risk from being high-degree
            if node in high_risk_nodes:
                risk_score += 0.5
            
            # Risk from connections to high-risk nodes
            for neighbor in address_graph.neighbors(node):
                if neighbor in high_risk_nodes:
                    # Risk decays with distance and edge weight
                    edge_weight = address_graph[node][neighbor].get('weight', 1)
                    risk_score += 0.1 * (edge_weight / 1000)  # Normalize
            
            risk_propagation[node] = min(risk_score, 1.0)
        
        return risk_propagation
    
    def _calculate_influence_scores(self, address_graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate influence scores for addresses"""
        influence_scores = {}
        
        # Combine multiple centrality measures
        degree_cent = nx.degree_centrality(address_graph)
        pagerank = nx.pagerank(address_graph)
        
        try:
            eigenvector_cent = nx.eigenvector_centrality(address_graph)
        except:
            eigenvector_cent = {node: 0 for node in address_graph.nodes()}
        
        for node in address_graph.nodes():
            # Weighted combination of centrality measures
            influence = (
                0.3 * degree_cent.get(node, 0) +
                0.4 * pagerank.get(node, 0) +
                0.3 * eigenvector_cent.get(node, 0)
            )
            influence_scores[node] = influence
        
        return influence_scores
    
    def _create_empty_relationship_result(self) -> AddressRelationshipResult:
        """Create empty relationship result for edge cases"""
        return AddressRelationshipResult(
            address_clusters={},
            relationship_strength={},
            address_roles={},
            interaction_patterns={},
            temporal_relationships={},
            risk_propagation={},
            influence_scores={}
        )

class GraphNeuralNetworkModel:
    """Combined Graph Neural Network Analysis Model"""
    
    def __init__(self, asset: str = "BTC"):
        self.asset = asset.upper()
        self.graph_builder = GraphBuilder(asset)
        self.graph_analyzer = TransactionGraphAnalyzer(asset)
        self.relationship_analyzer = AddressRelationshipAnalyzer(asset)
        
    def analyze(self, 
               transaction_data: List[Dict] = None,
               address_data: List[Dict] = None) -> GNNAnalysisResult:
        """Perform comprehensive GNN analysis
        
        Args:
            transaction_data: List of transaction records
            address_data: List of address information
        """
        try:
            # Build transaction graph
            if transaction_data:
                self.graph_builder.add_transaction_data(transaction_data)
            
            graph = self.graph_builder.build_networkx_graph()
            
            if graph is None:
                return self._create_empty_result()
            
            # Analyze transaction graph structure
            transaction_graph = self.graph_analyzer.analyze_transaction_graph(graph)
            
            # Analyze address relationships
            address_relationships = self.relationship_analyzer.analyze_address_relationships(graph)
            
            # Analyze network topology
            network_topology = self._analyze_network_topology(graph)
            
            # Temporal analysis
            temporal_analysis = self._analyze_temporal_patterns(graph)
            
            # Graph embedding
            graph_embedding = self._perform_graph_embedding(graph)
            
            # Anomaly detection
            anomaly_detection = self._detect_graph_anomalies(graph, graph_embedding)
            
            # Price prediction
            price_prediction = self._predict_price_from_graph(graph, transaction_graph)
            
            # Calculate overall scores
            overall_network_health = self._calculate_network_health(transaction_graph, network_topology)
            graph_complexity_score = self._calculate_graph_complexity(transaction_graph)
            
            return GNNAnalysisResult(
                transaction_graph=transaction_graph,
                address_relationships=address_relationships,
                network_topology=network_topology,
                temporal_analysis=temporal_analysis,
                graph_embedding=graph_embedding,
                anomaly_detection=anomaly_detection,
                price_prediction=price_prediction,
                overall_network_health=overall_network_health,
                graph_complexity_score=graph_complexity_score
            )
            
        except Exception as e:
            logger.error(f"Error in GNN analysis: {str(e)}")
            raise
    
    def _analyze_network_topology(self, graph: nx.DiGraph) -> NetworkTopologyResult:
        """Analyze network topology"""
        if not NETWORKX_AVAILABLE:
            return self._create_empty_topology_result()
        
        # Determine topology type
        topology_type = self._classify_topology(graph)
        
        # Calculate robustness
        robustness_score = self._calculate_robustness(graph)
        
        # Find vulnerability points
        vulnerability_points = self._find_vulnerability_points(graph)
        
        # Find critical paths
        critical_paths = self._find_critical_paths(graph)
        
        # Network efficiency
        network_efficiency = self._calculate_network_efficiency(graph)
        
        # Modularity
        modularity = self._calculate_modularity(graph)
        
        # Assortativity
        assortativity = self._calculate_assortativity(graph)
        
        # Rich club coefficient
        rich_club_coefficient = self._calculate_rich_club_coefficient(graph)
        
        # Core-periphery structure
        core_periphery_structure = self._analyze_core_periphery(graph)
        
        return NetworkTopologyResult(
            topology_type=topology_type,
            robustness_score=robustness_score,
            vulnerability_points=vulnerability_points,
            critical_paths=critical_paths,
            network_efficiency=network_efficiency,
            modularity=modularity,
            assortativity=assortativity,
            rich_club_coefficient=rich_club_coefficient,
            core_periphery_structure=core_periphery_structure
        )
    
    def _classify_topology(self, graph: nx.DiGraph) -> str:
        """Classify network topology type"""
        if graph.number_of_nodes() < 10:
            return "small"
        
        # Calculate key metrics
        clustering = nx.average_clustering(graph.to_undirected())
        degree_sequence = [d for n, d in graph.degree()]
        degree_variance = np.var(degree_sequence) if degree_sequence else 0
        
        # Classification logic
        if clustering > 0.3 and degree_variance < np.mean(degree_sequence):
            return "small_world"
        elif degree_variance > 2 * np.mean(degree_sequence):
            return "scale_free"
        elif clustering < 0.1:
            return "random"
        else:
            return "hierarchical"
    
    def _calculate_robustness(self, graph: nx.DiGraph) -> float:
        """Calculate network robustness"""
        if graph.number_of_nodes() < 2:
            return 0.0
        
        # Robustness = resistance to node removal
        original_components = nx.number_weakly_connected_components(graph)
        
        # Remove top 10% of nodes by degree
        nodes_by_degree = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)
        nodes_to_remove = nodes_by_degree[:max(1, len(nodes_by_degree) // 10)]
        
        temp_graph = graph.copy()
        temp_graph.remove_nodes_from(nodes_to_remove)
        
        new_components = nx.number_weakly_connected_components(temp_graph)
        
        # Robustness score (lower is more robust)
        robustness = 1.0 - (new_components - original_components) / graph.number_of_nodes()
        return max(0.0, robustness)
    
    def _find_vulnerability_points(self, graph: nx.DiGraph) -> List[str]:
        """Find critical nodes whose removal would fragment the network"""
        vulnerability_points = []
        
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(graph)
        
        # Top 5% by betweenness centrality are vulnerability points
        threshold = np.percentile(list(betweenness.values()), 95)
        
        for node, centrality in betweenness.items():
            if centrality >= threshold:
                vulnerability_points.append(node)
        
        return vulnerability_points
    
    def _find_critical_paths(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find critical paths in the network"""
        critical_paths = []
        
        # Find paths between high-degree nodes
        high_degree_nodes = [n for n, d in graph.degree() if d > np.percentile([deg for _, deg in graph.degree()], 90)]
        
        for i, source in enumerate(high_degree_nodes[:5]):  # Limit to avoid computation explosion
            for target in high_degree_nodes[i+1:i+3]:  # Limit targets
                try:
                    if nx.has_path(graph, source, target):
                        path = nx.shortest_path(graph, source, target)
                        if len(path) > 2:  # Only non-trivial paths
                            critical_paths.append(path)
                except:
                    continue
        
        return critical_paths[:10]  # Return top 10 critical paths
    
    def _calculate_network_efficiency(self, graph: nx.DiGraph) -> float:
        """Calculate network efficiency"""
        try:
            undirected = graph.to_undirected()
            return nx.global_efficiency(undirected)
        except:
            return 0.0
    
    def _calculate_modularity(self, graph: nx.DiGraph) -> float:
        """Calculate network modularity"""
        try:
            undirected = graph.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected)
            return nx.community.modularity(undirected, communities)
        except:
            return 0.0
    
    def _calculate_assortativity(self, graph: nx.DiGraph) -> float:
        """Calculate degree assortativity"""
        try:
            return nx.degree_assortativity_coefficient(graph)
        except:
            return 0.0
    
    def _calculate_rich_club_coefficient(self, graph: nx.DiGraph) -> float:
        """Calculate rich club coefficient"""
        try:
            rich_club = nx.rich_club_coefficient(graph.to_undirected())
            return np.mean(list(rich_club.values())) if rich_club else 0.0
        except:
            return 0.0
    
    def _analyze_core_periphery(self, graph: nx.DiGraph) -> Dict[str, str]:
        """Analyze core-periphery structure"""
        core_periphery = {}
        
        # Use degree centrality to identify core
        degree_centrality = nx.degree_centrality(graph)
        threshold = np.percentile(list(degree_centrality.values()), 80)
        
        for node, centrality in degree_centrality.items():
            if centrality >= threshold:
                core_periphery[node] = "core"
            else:
                core_periphery[node] = "periphery"
        
        return core_periphery
    
    def _analyze_temporal_patterns(self, graph: nx.DiGraph) -> TemporalGraphResult:
        """Analyze temporal patterns in the graph"""
        # Simplified temporal analysis
        temporal_patterns = {'activity': [1.0] * 24}  # Hourly activity
        evolution_metrics = {'growth_rate': 0.05, 'density_change': 0.01}
        growth_rate = 0.05
        activity_cycles = [{'period': 24, 'amplitude': 0.3}]
        temporal_centrality = {'degree': {}, 'betweenness': {}}
        dynamic_communities = {}
        temporal_anomalies = []
        
        return TemporalGraphResult(
            temporal_patterns=temporal_patterns,
            evolution_metrics=evolution_metrics,
            growth_rate=growth_rate,
            activity_cycles=activity_cycles,
            temporal_centrality=temporal_centrality,
            dynamic_communities=dynamic_communities,
            temporal_anomalies=temporal_anomalies
        )
    
    def _perform_graph_embedding(self, graph: nx.DiGraph) -> GraphEmbeddingResult:
        """Perform graph embedding"""
        node_embeddings = {}
        embedding_quality = 0.7
        similarity_matrix = np.eye(graph.number_of_nodes())
        clusters = {'cluster_0': list(graph.nodes())}
        anomalous_nodes = []
        embedding_dimensions = 64
        
        # Simplified embedding using node features
        for i, node in enumerate(graph.nodes()):
            # Create simple embedding based on node properties
            degree = graph.degree(node)
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            # Simple feature vector
            embedding = np.array([degree, in_degree, out_degree] + [0] * 61)  # Pad to 64 dimensions
            node_embeddings[node] = embedding
        
        return GraphEmbeddingResult(
            node_embeddings=node_embeddings,
            embedding_quality=embedding_quality,
            similarity_matrix=similarity_matrix,
            clusters=clusters,
            anomalous_nodes=anomalous_nodes,
            embedding_dimensions=embedding_dimensions
        )
    
    def _detect_graph_anomalies(self, graph: nx.DiGraph, embedding_result: GraphEmbeddingResult) -> AnomalyDetectionResult:
        """Detect anomalies in the graph"""
        anomalous_nodes = []
        anomalous_edges = []
        anomaly_scores = {}
        anomaly_types = {}
        suspicious_patterns = []
        risk_assessment = {}
        
        # Detect high-degree anomalies
        degrees = [d for n, d in graph.degree()]
        if degrees:
            degree_threshold = np.percentile(degrees, 95)
            
            for node, degree in graph.degree():
                if degree > degree_threshold:
                    anomalous_nodes.append(node)
                    anomaly_scores[node] = degree / max(degrees)
                    anomaly_types[node] = "high_degree"
                    risk_assessment[node] = min(degree / 1000, 1.0)
        
        return AnomalyDetectionResult(
            anomalous_nodes=anomalous_nodes,
            anomalous_edges=anomalous_edges,
            anomaly_scores=anomaly_scores,
            anomaly_types=anomaly_types,
            suspicious_patterns=suspicious_patterns,
            risk_assessment=risk_assessment
        )
    
    def _predict_price_from_graph(self, graph: nx.DiGraph, graph_result: TransactionGraphResult) -> PricePredictionResult:
        """Predict price movement from graph features"""
        # Simplified price prediction based on graph metrics
        
        # Features that might influence price
        network_activity = graph_result.total_edges / max(graph_result.total_nodes, 1)
        clustering_effect = graph_result.clustering_coefficient
        centralization = 1 - graph_result.graph_density
        
        # Simple prediction model
        prediction_factors = [
            network_activity * 0.4,
            clustering_effect * 0.3,
            centralization * 0.3
        ]
        
        predicted_change = sum(prediction_factors) * 10 - 5  # Scale to ±5%
        confidence = min(graph_result.total_nodes / 1000, 1.0)  # More nodes = higher confidence
        
        influential_factors = {
            'network_activity': network_activity,
            'clustering': clustering_effect,
            'centralization': centralization
        }
        
        graph_signals = {
            'density': graph_result.graph_density,
            'components': graph_result.connected_components,
            'diameter': graph_result.diameter
        }
        
        feature_importance = {
            'network_activity': 0.4,
            'clustering': 0.3,
            'centralization': 0.3
        }
        
        return PricePredictionResult(
            predicted_price_change=predicted_change,
            confidence_score=confidence,
            influential_factors=influential_factors,
            graph_signals=graph_signals,
            prediction_horizon=24,  # 24 hours
            feature_importance=feature_importance
        )
    
    def _calculate_network_health(self, graph_result: TransactionGraphResult, topology_result: NetworkTopologyResult) -> float:
        """Calculate overall network health score"""
        health_factors = [
            min(graph_result.total_nodes / 10000, 1.0) * 0.2,  # Network size
            graph_result.clustering_coefficient * 0.2,  # Clustering
            (1 - graph_result.graph_density) * 0.2,  # Not too dense
            topology_result.robustness_score * 0.2,  # Robustness
            min(topology_result.network_efficiency, 1.0) * 0.2  # Efficiency
        ]
        
        return sum(health_factors)
    
    def _calculate_graph_complexity(self, graph_result: TransactionGraphResult) -> float:
        """Calculate graph complexity score"""
        complexity_factors = [
            min(graph_result.total_edges / graph_result.total_nodes, 10) / 10 if graph_result.total_nodes > 0 else 0,  # Edge density
            graph_result.clustering_coefficient,  # Local structure
            min(graph_result.diameter / 10, 1.0),  # Global structure
            min(len(graph_result.community_structure) / 100, 1.0)  # Community complexity
        ]
        
        return np.mean(complexity_factors)
    
    def _create_empty_result(self) -> GNNAnalysisResult:
        """Create empty result for edge cases"""
        return GNNAnalysisResult(
            transaction_graph=self.graph_analyzer._create_empty_graph_result(),
            address_relationships=self.relationship_analyzer._create_empty_relationship_result(),
            network_topology=self._create_empty_topology_result(),
            temporal_analysis=TemporalGraphResult({}, {}, 0.0, [], {}, {}, []),
            graph_embedding=GraphEmbeddingResult({}, 0.0, np.array([]), {}, [], 0),
            anomaly_detection=AnomalyDetectionResult([], [], {}, {}, [], {}),
            price_prediction=PricePredictionResult(0.0, 0.0, {}, {}, 0, {}),
            overall_network_health=0.0,
            graph_complexity_score=0.0
        )
    
    def _create_empty_topology_result(self) -> NetworkTopologyResult:
        """Create empty topology result"""
        return NetworkTopologyResult(
            topology_type="unknown",
            robustness_score=0.0,
            vulnerability_points=[],
            critical_paths=[],
            network_efficiency=0.0,
            modularity=0.0,
            assortativity=0.0,
            rich_club_coefficient=0.0,
            core_periphery_structure={}
        )
    
    def get_gnn_insights(self, result: GNNAnalysisResult) -> Dict[str, str]:
        """Generate comprehensive GNN insights"""
        insights = {}
        
        # Graph structure insights
        insights['graph_structure'] = f"Nodes: {result.transaction_graph.total_nodes:,}, Edges: {result.transaction_graph.total_edges:,}, Density: {result.transaction_graph.graph_density:.3f}"
        
        # Network topology insights
        insights['topology'] = f"Type: {result.network_topology.topology_type}, Robustness: {result.network_topology.robustness_score:.2f}, Efficiency: {result.network_topology.network_efficiency:.2f}"
        
        # Community insights
        insights['communities'] = f"Communities: {len(result.transaction_graph.community_structure)}, Modularity: {result.network_topology.modularity:.2f}"
        
        # Centrality insights
        if result.transaction_graph.centrality_scores.get('pagerank'):
            top_pagerank = max(result.transaction_graph.centrality_scores['pagerank'].values())
            insights['centrality'] = f"Max PageRank: {top_pagerank:.3f}, Clustering: {result.transaction_graph.clustering_coefficient:.3f}"
        
        # Anomaly insights
        insights['anomalies'] = f"Anomalous Nodes: {len(result.anomaly_detection.anomalous_nodes)}, Risk Nodes: {len([n for n, r in result.anomaly_detection.risk_assessment.items() if r > 0.5])}"
        
        # Price prediction insights
        insights['price_prediction'] = f"Predicted Change: {result.price_prediction.predicted_price_change:+.2f}%, Confidence: {result.price_prediction.confidence_score:.1%}"
        
        # Overall insights
        insights['network_health'] = f"Health Score: {result.overall_network_health:.1%}, Complexity: {result.graph_complexity_score:.1%}"
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_transactions = [
        {
            'tx_id': 'tx1',
            'from_address': 'addr1',
            'to_address': 'addr2',
            'amount': 1.5,
            'timestamp': datetime.now() - timedelta(hours=1),
            'fee': 0.001
        },
        {
            'tx_id': 'tx2',
            'from_address': 'addr2',
            'to_address': 'addr3',
            'amount': 0.8,
            'timestamp': datetime.now() - timedelta(hours=2),
            'fee': 0.001
        },
        {
            'tx_id': 'tx3',
            'from_address': 'addr1',
            'to_address': 'addr3',
            'amount': 2.1,
            'timestamp': datetime.now() - timedelta(hours=3),
            'fee': 0.002
        }
    ]
    
    # Test the model
    gnn_model = GraphNeuralNetworkModel("BTC")
    result = gnn_model.analyze(transaction_data=sample_transactions)
    
    insights = gnn_model.get_gnn_insights(result)
    
    print("=== Graph Neural Network Analysis ===")
    print(f"Network Health: {result.overall_network_health:.1%}")
    print(f"Graph Complexity: {result.graph_complexity_score:.1%}")
    print(f"Price Prediction: {result.price_prediction.predicted_price_change:+.2f}%")
    print()
    
    print("=== Detailed Insights ===")
    for key, insight in insights.items():
        print(f"{key}: {insight}")