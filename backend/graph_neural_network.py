import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class CryptoIndicatorResult:
    """Result container for crypto indicator calculations"""
    indicator_name: str
    value: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1 signal strength

class GraphNeuralNetwork:
    """Graph Neural Network for blockchain transaction analysis"""
    
    def __init__(self):
        self.transaction_graph = {}
        self.node_features = {}
        
    def build_transaction_graph(self, 
                               transactions: List[Dict[str, Any]]) -> bool:
        """Build transaction graph from blockchain data"""
        try:
            self.transaction_graph = {}
            self.node_features = {}
            
            for tx in transactions:
                from_addr = tx.get('from_address', '')
                to_addr = tx.get('to_address', '')
                amount = tx.get('amount', 0)
                timestamp = tx.get('timestamp', datetime.now())
                
                # Add nodes
                if from_addr not in self.transaction_graph:
                    self.transaction_graph[from_addr] = []
                    self.node_features[from_addr] = {
                        'total_sent': 0,
                        'total_received': 0,
                        'tx_count': 0,
                        'first_seen': timestamp
                    }
                    
                if to_addr not in self.transaction_graph:
                    self.transaction_graph[to_addr] = []
                    self.node_features[to_addr] = {
                        'total_sent': 0,
                        'total_received': 0,
                        'tx_count': 0,
                        'first_seen': timestamp
                    }
                
                # Add edge
                self.transaction_graph[from_addr].append({
                    'to': to_addr,
                    'amount': amount,
                    'timestamp': timestamp
                })
                
                # Update node features
                self.node_features[from_addr]['total_sent'] += amount
                self.node_features[from_addr]['tx_count'] += 1
                self.node_features[to_addr]['total_received'] += amount
                
            return True
            
        except Exception as e:
            logger.error(f"Error building transaction graph: {e}")
            return False
    
    def calculate_network_health(self) -> CryptoIndicatorResult:
        """Calculate network health metrics using graph analysis"""
        try:
            if not self.transaction_graph:
                raise ValueError("Transaction graph not built")
            
            # Calculate basic graph metrics
            num_nodes = len(self.transaction_graph)
            num_edges = sum(len(edges) for edges in self.transaction_graph.values())
            
            if num_nodes == 0:
                raise ValueError("Empty transaction graph")
            
            # Calculate degree centrality
            degree_centralities = []
            for node in self.transaction_graph:
                out_degree = len(self.transaction_graph[node])
                in_degree = sum(1 for edges in self.transaction_graph.values() 
                              for edge in edges if edge['to'] == node)
                total_degree = out_degree + in_degree
                centrality = total_degree / (num_nodes - 1) if num_nodes > 1 else 0
                degree_centralities.append(centrality)
            
            # Calculate network density
            max_edges = num_nodes * (num_nodes - 1)
            density = num_edges / max_edges if max_edges > 0 else 0
            
            # Calculate clustering coefficient (simplified)
            clustering_coeff = self._calculate_clustering_coefficient()
            
            # Calculate activity distribution
            activities = [features['tx_count'] for features in self.node_features.values()]
            activity_variance = np.var(activities) if activities else 0
            activity_mean = np.mean(activities) if activities else 0
            
            # Combine metrics into health score
            health_score = (
                0.3 * min(density * 10, 1.0) +  # Network connectivity
                0.3 * min(clustering_coeff, 1.0) +  # Local clustering
                0.2 * min(np.mean(degree_centralities), 1.0) +  # Centralization
                0.2 * min(activity_mean / 100, 1.0)  # Activity level
            )
            
            # Generate signals based on health score
            if health_score > 0.7:
                signal = 'buy'
                strength = health_score
            elif health_score < 0.3:
                signal = 'sell'
                strength = 1.0 - health_score
            else:
                signal = 'hold'
                strength = 0.5
            
            confidence = min(0.8, num_nodes / 1000)  # Higher confidence with more data
            
            return CryptoIndicatorResult(
                indicator_name='Network Health (GNN)',
                value=health_score,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'num_nodes': num_nodes,
                    'num_edges': num_edges,
                    'density': density,
                    'clustering_coefficient': clustering_coeff,
                    'avg_degree_centrality': np.mean(degree_centralities) if degree_centralities else 0,
                    'activity_variance': activity_variance,
                    'activity_mean': activity_mean
                },
                signal=signal,
                strength=strength
            )
            
        except Exception as e:
            logger.error(f"Error calculating network health: {e}")
            return self._error_result('Network Health (GNN)', str(e))
    
    def _calculate_clustering_coefficient(self) -> float:
        """Calculate simplified clustering coefficient"""
        try:
            if len(self.transaction_graph) < 3:
                return 0.0
            
            clustering_coeffs = []
            
            for node in self.transaction_graph:
                neighbors = set()
                # Get outgoing neighbors
                for edge in self.transaction_graph[node]:
                    neighbors.add(edge['to'])
                # Get incoming neighbors
                for other_node, edges in self.transaction_graph.items():
                    for edge in edges:
                        if edge['to'] == node:
                            neighbors.add(other_node)
                
                neighbors.discard(node)  # Remove self
                
                if len(neighbors) < 2:
                    clustering_coeffs.append(0.0)
                    continue
                
                # Count triangles (simplified)
                triangles = 0
                possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
                
                for neighbor1 in neighbors:
                    for neighbor2 in neighbors:
                        if neighbor1 != neighbor2:
                            # Check if neighbor1 and neighbor2 are connected
                            if neighbor1 in self.transaction_graph:
                                for edge in self.transaction_graph[neighbor1]:
                                    if edge['to'] == neighbor2:
                                        triangles += 0.5  # Count each triangle once
                                        break
                
                clustering_coeff = triangles / possible_triangles if possible_triangles > 0 else 0
                clustering_coeffs.append(clustering_coeff)
            
            return np.mean(clustering_coeffs) if clustering_coeffs else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating clustering coefficient: {e}")
            return 0.0
    
    def _error_result(self, indicator_name: str, error_msg: str) -> CryptoIndicatorResult:
        return CryptoIndicatorResult(
            indicator_name=indicator_name,
            value=0.0,
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={'error': error_msg},
            signal='hold',
            strength=0.0
        )