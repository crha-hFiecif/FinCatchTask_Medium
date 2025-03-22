import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from typing import List, Dict, Any, Tuple
import spacy
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.neo4j_handler import neo4jHandler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import networkx as nx
import json

class knowledgeCluster:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.neo4j = neo4jHandler(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="12345678"  # Replace with your actual Neo4j password
        )
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for clustering"""
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower() for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        return " ".join(tokens)
    
    def get_causal_relationships(self, articles: List[Dict]) -> np.ndarray:
        """Get causal relationship matrix from Neo4j"""
        n = len(articles)
        relationship_matrix = np.zeros((n, n))
        
        with self.neo4j.driver.session() as session:
            # Get all relationships between articles
            result = session.run("""
                MATCH (a:Article)-[r]->(b:Article)
                RETURN a.title as source, b.title as target, 
                       r.confidence as confidence, type(r) as type
            """)
            
            # Create title to index mapping
            title_to_idx = {art['title']: i for i, art in enumerate(articles)}
            
            # Fill relationship matrix
            for record in result:
                source_idx = title_to_idx.get(record['source'])
                target_idx = title_to_idx.get(record['target'])
                if source_idx is not None and target_idx is not None:
                    # Weight different relationship types
                    weight = record['confidence']
                    if record['type'] == 'causes':
                        weight *= 1.5
                    elif record['type'] == 'correlates_with':
                        weight *= 1.0
                    elif record['type'] == 'influenced_by':
                        weight *= 1.2
                    
                    relationship_matrix[source_idx, target_idx] = weight
                    relationship_matrix[target_idx, source_idx] = weight  # Make it symmetric
        
        return relationship_matrix
    
    def evaluate_clustering(self, similarity_matrix: np.ndarray, clusters: np.ndarray) -> Dict:
        """Evaluate clustering quality using multiple metrics"""
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_ratio': n_noise / len(clusters),
        }
        
        if n_clusters > 1:
            # Convert similarity to distance and ensure diagonal is zero
            distances = 1 - similarity_matrix
            np.fill_diagonal(distances, 0)
            
            # Calculate various clustering metrics
            metrics['silhouette_score'] = silhouette_score(
                distances, clusters, metric='precomputed'
            )
            
            # Calculate average intra-cluster similarity
            intra_cluster_sim = []
            for cluster_id in set(clusters):
                if cluster_id != -1:
                    mask = clusters == cluster_id
                    cluster_sim = similarity_matrix[mask][:, mask]
                    intra_cluster_sim.append(cluster_sim.mean())
            
            metrics['avg_intra_cluster_similarity'] = np.mean(intra_cluster_sim)
            
            # Calculate average cluster size
            cluster_sizes = [
                list(clusters).count(i) 
                for i in set(clusters) if i != -1
            ]
            metrics['avg_cluster_size'] = np.mean(cluster_sizes)
            metrics['std_cluster_size'] = np.std(cluster_sizes)
            
        return metrics

    def visualize_clusters(self, similarity_matrix: np.ndarray, 
                         clusters: np.ndarray, 
                         articles: List[Dict],
                         output_dir: str = None):
        """Generate visualizations for the clustering results"""
        try:
            # Set default output directory using absolute path
            if output_dir is None:
                # Get the project root directory (Task)
                current_dir = os.path.abspath(os.path.dirname(__file__))  # Get absolute path
                print(f"Current directory: {current_dir}")
                
                # Navigate up to project root
                project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
                print(f"Project root: {project_root}")
                
                # Set output directory
                output_dir = os.path.join(project_root, 'data', 'visualizations')
                print(f"Output directory: {output_dir}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nVisualization directory created/exists: {output_dir}")
            
            # Verify directory is writable
            test_file = os.path.join(output_dir, 'test.txt')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                print("✅ Directory is writable")
            except Exception as e:
                print(f"❌ Directory is not writable: {str(e)}")
                return
            
            # Check if we have data to visualize
            print(f"\nData to visualize:")
            print(f"Number of articles: {len(articles)}")
            print(f"Similarity matrix shape: {similarity_matrix.shape}")
            print(f"Number of clusters: {len(set(clusters))}")
            
            if len(articles) == 0:
                print("❌ No articles to visualize")
                return
            
            if len(set(clusters)) <= 1:
                print("❌ No meaningful clusters found (all articles in one cluster)")
                return
            
            # 1. Cluster Size Distribution
            print("\nGenerating cluster size distribution...")
            plt.figure(figsize=(10, 6))
            cluster_sizes = [list(clusters).count(i) for i in set(clusters) if i != -1]
            if len(cluster_sizes) > 0:
                plt.bar(range(len(cluster_sizes)), sorted(cluster_sizes, reverse=True))
                plt.title('Cluster Size Distribution')
                plt.xlabel('Cluster Index')
                plt.ylabel('Number of Articles')
                output_file = os.path.join(output_dir, 'cluster_sizes.png')
                plt.savefig(output_file)
                plt.close()
                print(f"✅ Saved cluster_sizes.png to {output_file}")
            else:
                print("⚠️ No valid clusters to visualize")
            
            # 2. t-SNE Visualization of Clusters
            print("\nGenerating t-SNE visualization...")
            if similarity_matrix.shape[0] > 1:  # Need at least 2 points for t-SNE
                n_samples = len(clusters)
                perplexity = min(n_samples - 1, 15)  # Adjust perplexity based on sample size
                
                try:
                    # Convert similarity to distance
                    distances = 1 - similarity_matrix
                    np.fill_diagonal(distances, 0)
                    
                    tsne = TSNE(
                        n_components=2,
                        metric='precomputed',
                        random_state=42,
                        perplexity=perplexity,
                        n_iter=2000,
                        early_exaggeration=6,
                        init='random'  # Changed from 'pca' to 'random'
                    )
                    
                    coords = tsne.fit_transform(distances)
                    
                    plt.figure(figsize=(12, 8))
                    scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                                        c=clusters, cmap='tab20', alpha=0.6)
                    
                    # Add labels for each point
                    for i, (x, y) in enumerate(coords):
                        plt.annotate(
                            f"Article {i+1}",
                            (x, y),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            alpha=0.7
                        )
                    
                    plt.title('t-SNE Visualization of Clusters')
                    plt.colorbar(scatter)
                    output_file = os.path.join(output_dir, 'tsne_clusters.png')
                    plt.savefig(output_file)
                    plt.close()
                    print(f"✅ Saved tsne_clusters.png to {output_file}")
                except Exception as e:
                    print(f"⚠️ Could not generate t-SNE visualization: {str(e)}")
                    print("Continuing with other visualizations...")
            else:
                print("⚠️ Not enough data points for t-SNE visualization")
            
            # 3. Similarity Matrix Heatmap
            print("\nGenerating similarity matrix heatmap...")
            if similarity_matrix.size > 0:
                plt.figure(figsize=(12, 10))
                sns.heatmap(similarity_matrix, cmap='YlOrRd')
                plt.title('Article Similarity Matrix')
                output_file = os.path.join(output_dir, 'similarity_matrix.png')
                plt.savefig(output_file)
                plt.close()
                print(f"✅ Saved similarity_matrix.png to {output_file}")
            else:
                print("⚠️ Empty similarity matrix")
            
            # 4. Generate cluster relationship graph
            print("\nGenerating cluster relationships visualization...")
            self._visualize_cluster_relationships(clusters, articles, output_dir)
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _visualize_cluster_relationships(self, clusters: np.ndarray, 
                                       articles: List[Dict], 
                                       output_dir: str):
        """Visualize relationships between clusters"""
        try:
            with self.neo4j.driver.session() as session:
                # Get relationships between articles in different clusters
                print("Querying Neo4j for cluster relationships...")
                result = session.run("""
                    MATCH (c1:Cluster)<-[:BELONGS_TO]-(a1:Article)
                          -[r]->(a2:Article)-[:BELONGS_TO]->(c2:Cluster)
                    WHERE c1 <> c2  // Only between different clusters
                    RETURN c1.id as source_cluster, c2.id as target_cluster,
                           type(r) as relationship_type, count(*) as weight,
                           collect(distinct type(r)) as relationship_types
                """)
                
                # Create relationship graph
                plt.figure(figsize=(12, 8))
                G = nx.Graph()
                
                # Add nodes and edges with weights
                edge_weights = {}
                edge_types = {}
                for record in result:
                    source = f"Cluster {record['source_cluster']}"
                    target = f"Cluster {record['target_cluster']}"
                    weight = record['weight']
                    rel_types = record['relationship_types']
                    
                    if source != target:  # Ignore self-loops
                        edge_weights[(source, target)] = weight
                        edge_types[(source, target)] = rel_types
                        G.add_edge(source, target, weight=weight)
                
                print(f"Found {G.number_of_edges()} relationships between clusters")
                
                if G.number_of_edges() == 0:
                    print("⚠️ Warning: No relationships found between clusters")
                    plt.close()
                    return
                
                # Draw the graph
                pos = nx.spring_layout(G)
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, 
                                     node_color='lightblue',
                                     node_size=1000)
                
                # Draw edges with varying widths
                edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, width=edge_widths)
                
                # Add labels
                nx.draw_networkx_labels(G, pos, font_size=8)
                
                # Add edge labels showing relationship types
                edge_labels = {(u, v): '\n'.join(edge_types.get((u, v), [])) 
                             for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
                
                plt.title('Inter-cluster Relationships\n(Edge width = number of relationships)')
                plt.savefig(os.path.join(output_dir, 'cluster_relationships.png'),
                           bbox_inches='tight', dpi=300)
                plt.close()
                print("✅ Saved cluster_relationships.png")
                
        except Exception as e:
            print(f"❌ Error generating cluster relationships visualization: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def cluster_articles(self, articles: List[Dict], eps: float = 0.5, min_samples: int = 2, content_weight: float = 0.7) -> Tuple[List[int], Dict]:
        """Perform clustering on articles"""
        try:
            print(f"\nStarting clustering with {len(articles)} articles...")
            print(f"Parameters: eps={eps}, min_samples={min_samples}, content_weight={content_weight}")
            
            # Generate similarity matrices
            content_similarity = self._generate_content_similarity(articles)
            print(f"Generated content similarity matrix: {content_similarity.shape}")
            
            relationship_similarity = self._generate_relationship_similarity(articles)
            print(f"Generated relationship matrix: {relationship_similarity.shape}")
            
            # Combine similarity matrices
            combined_similarity = content_weight * content_similarity + (1 - content_weight) * relationship_similarity
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            clusters = dbscan.fit_predict(combined_similarity)
            
            # Add cluster assignments to articles
            for article, cluster_id in zip(articles, clusters):
                article['cluster'] = int(cluster_id)
            
            # Calculate metrics
            metrics = self._calculate_clustering_metrics(clusters, combined_similarity)
            
            # Save cluster assignments to Neo4j
            self.save_clusters_to_neo4j(articles)
            
            # Generate visualizations
            self._generate_visualizations(articles, clusters, combined_similarity)
            
            return clusters, metrics
            
        except Exception as e:
            print(f"Error in clustering process: {str(e)}")
            return [], {}

    def get_cluster_summary(self, articles: List[Dict]) -> Dict:
        """Get summary of clustering results"""
        try:
            # Group articles by cluster
            clusters = {}
            for article in articles:
                cluster_id = article.get('cluster', -1)  # Use get() with default value
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(article)
            
            # Create summary
            summary = {
                'num_clusters': len([c for c in clusters.keys() if c != -1]),
                'num_noise': len(clusters.get(-1, [])),
                'clusters': {}
            }
            
            for cluster_id, cluster_articles in clusters.items():
                if cluster_id != -1:
                    summary['clusters'][str(cluster_id)] = {
                        'size': len(cluster_articles),
                        'articles': [a['title'] for a in cluster_articles]
                    }
            
            return summary
            
        except Exception as e:
            print(f"Error generating cluster summary: {str(e)}")
            return {
                'num_clusters': 0,
                'num_noise': 0,
                'clusters': {}
            }

    def get_cluster_data(self) -> Dict[str, Any]:
        """Get the current cluster data from Neo4j"""
        try:
            with self.neo4j.driver.session() as session:
                # Get all clusters and their articles
                result = session.run("""
                    MATCH (c:Cluster)<-[:BELONGS_TO]-(a:Article)
                    RETURN c.id as cluster_id,
                           collect({
                               title: a.title,
                               summary: a.summary,
                               content: a.content
                           }) as articles
                    ORDER BY c.id
                """)
                
                # Convert to list of clusters
                clusters = []
                for record in result:
                    clusters.append({
                        'id': record['cluster_id'],
                        'articles': record['articles']
                    })
                
                return clusters
                
        except Exception as e:
            print(f"Error getting cluster data: {str(e)}")
            return None

    def save_clusters_to_neo4j(self, articles: List[Dict]):
        """Save cluster assignments to Neo4j"""
        try:
            with self.neo4j.driver.session() as session:
                # First, remove old cluster assignments
                session.run("MATCH (a:Article)-[r:BELONGS_TO]->() DELETE r")
                session.run("MATCH (c:Cluster) DELETE c")
                
                # Create cluster nodes
                unique_clusters = set(article['cluster'] for article in articles if article['cluster'] != -1)
                for cluster_id in unique_clusters:
                    session.run(
                        "CREATE (c:Cluster {id: $cluster_id})",
                        cluster_id=int(cluster_id)
                    )
                
                # Create BELONGS_TO relationships
                for article in articles:
                    if article['cluster'] != -1:
                        session.run("""
                            MATCH (a:Article {title: $title})
                            MATCH (c:Cluster {id: $cluster_id})
                            CREATE (a)-[:BELONGS_TO]->(c)
                        """, title=article['title'], cluster_id=int(article['cluster']))
            
            print("✅ Saved cluster assignments to Neo4j")
            
        except Exception as e:
            print(f"❌ Error saving cluster assignments to Neo4j: {str(e)}")

    def _generate_content_similarity(self, articles: List[Dict]) -> np.ndarray:
        """Generate content-based similarity matrix"""
        try:
            # Preprocess texts
            processed_texts = [
                self.preprocess_text(article['title'] + " " + article['content'])
                for article in articles
            ]
            
            # Generate TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Calculate cosine similarity
            content_similarity = cosine_similarity(tfidf_matrix.toarray())
            
            # Ensure diagonal is 1
            np.fill_diagonal(content_similarity, 1)
            
            return content_similarity
            
        except Exception as e:
            print(f"Error generating content similarity: {str(e)}")
            return np.zeros((len(articles), len(articles)))

    def _generate_relationship_similarity(self, articles: List[Dict]) -> np.ndarray:
        """Generate relationship-based similarity matrix"""
        try:
            # Get relationship matrix
            relationship_matrix = self.get_causal_relationships(articles)
            
            # Normalize to [0,1] range
            if relationship_matrix.max() > 0:
                relationship_matrix = relationship_matrix / relationship_matrix.max()
            
            # Ensure diagonal is 1
            np.fill_diagonal(relationship_matrix, 1)
            
            return relationship_matrix
            
        except Exception as e:
            print(f"Error generating relationship similarity: {str(e)}")
            return np.zeros((len(articles), len(articles)))

    def _calculate_clustering_metrics(self, clusters: List[int], similarity_matrix: np.ndarray) -> Dict:
        """Calculate clustering evaluation metrics"""
        try:
            metrics = {}
            
            # Only calculate metrics if we have clusters
            if len(set(clusters)) > 1:
                # Convert similarity to distance
                distance_matrix = 1 - similarity_matrix
                np.fill_diagonal(distance_matrix, 0)
                
                # Calculate metrics
                metrics['silhouette'] = silhouette_score(distance_matrix, clusters, metric='precomputed')
                metrics['calinski_harabasz'] = calinski_harabasz_score(distance_matrix, clusters)
                metrics['davies_bouldin'] = davies_bouldin_score(distance_matrix, clusters)
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {}

    def _generate_visualizations(self, articles: List[Dict], clusters: List[int], similarity_matrix: np.ndarray):
        """Generate clustering visualizations"""
        try:
            # Get the project root directory
            current_dir = os.path.abspath(os.path.dirname(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
            output_dir = os.path.join(project_root, 'data', 'visualizations')
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate visualizations
            self.visualize_clusters(similarity_matrix, clusters, articles)
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")

if __name__ == "__main__":
    try:
        # Create clusterer instance
        clusterer = knowledgeCluster()
        
        # Get articles from Neo4j
        with clusterer.neo4j.driver.session() as session:
            result = session.run("""
                MATCH (a:Article)
                RETURN a.title as title,
                       a.content as content,
                       a.summary as summary
            """)
            articles = [dict(record) for record in result]
        
        if not articles:
            print("No articles found in Neo4j")
            sys.exit(0)  # Exit gracefully
            
        # Perform clustering
        clusters, metrics = clusterer.cluster_articles(
            articles,
            eps=0.5,
            min_samples=2,
            content_weight=0.7
        )
        
        # Generate cluster summary
        cluster_summary = clusterer.get_cluster_summary(articles)
        print("\nCluster Summary:")
        print(json.dumps(cluster_summary, indent=2))
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        print(f"Error in clustering process: {str(e)}")
        sys.exit(0)  # Exit gracefully even on error 