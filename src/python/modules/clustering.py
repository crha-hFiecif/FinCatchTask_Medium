import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from typing import List, Dict
import spacy
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.neo4j_handler import Neo4jHandler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import networkx as nx

class KnowledgeClusterer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.neo4j = Neo4jHandler(
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
                
                tsne = TSNE(
                    n_components=2,
                    metric='precomputed',
                    random_state=42,
                    perplexity=perplexity,
                    n_iter=2000,
                    early_exaggeration=6
                )
                distances = 1 - similarity_matrix
                try:
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

    def cluster_articles(self, articles: List[Dict], eps: float = 0.5, min_samples: int = 2,
                        content_weight: float = 0.7):
        """Cluster articles using both content and causal relationships"""
        print(f"\nStarting clustering with {len(articles)} articles...")
        print(f"Parameters: eps={eps}, min_samples={min_samples}, content_weight={content_weight}")
        
        # Get content-based similarity
        processed_texts = [
            self.preprocess_text(article['title'] + " " + article['content'])
            for article in articles
        ]
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        content_similarity = cosine_similarity(tfidf_matrix.toarray())
        print(f"Generated content similarity matrix: {content_similarity.shape}")
        
        # Get relationship-based similarity
        relationship_matrix = self.get_causal_relationships(articles)
        print(f"Generated relationship matrix: {relationship_matrix.shape}")
        print(f"Number of relationships found: {np.count_nonzero(relationship_matrix)}")
        
        # Combine both similarities
        combined_similarity = (
            content_weight * content_similarity +
            (1 - content_weight) * relationship_matrix
        )
        
        # Convert similarity to distance and ensure diagonal is zero
        distance_matrix = 1 - combined_similarity
        np.fill_diagonal(distance_matrix, 0)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed'  # Use our pre-computed similarity matrix
        )
        clusters = clustering.fit_predict(distance_matrix)  # Use distance matrix
        
        # Print clustering results
        unique_clusters = set(clusters)
        print(f"\nClustering Results:")
        print(f"Number of clusters: {len(unique_clusters) - (1 if -1 in clusters else 0)}")
        print(f"Number of noise points: {list(clusters).count(-1)}")
        for cluster_id in sorted(unique_clusters):
            if cluster_id != -1:
                print(f"Cluster {cluster_id}: {list(clusters).count(cluster_id)} articles")
        
        # Save cluster assignments to Neo4j
        with self.neo4j.driver.session() as session:
            # First, remove old cluster assignments
            session.run("MATCH (a:Article)-[r:BELONGS_TO]->() DELETE r")
            session.run("MATCH (c:Cluster) DELETE c")
            
            # Create cluster nodes
            unique_clusters = set(clusters)
            for cluster_id in unique_clusters:
                if cluster_id != -1:  # Skip noise points
                    session.run(
                        "CREATE (c:Cluster {id: $cluster_id})",
                        cluster_id=int(cluster_id)
                    )
            
            # Create BELONGS_TO relationships
            for i, (cluster_id, article) in enumerate(zip(clusters, articles)):
                if cluster_id != -1:  # Skip noise points
                    session.run("""
                        MATCH (a:Article {title: $title})
                        MATCH (c:Cluster {id: $cluster_id})
                        CREATE (a)-[:BELONGS_TO]->(c)
                    """, title=article['title'], cluster_id=int(cluster_id))
            
            # Verify cluster assignments
            result = session.run("""
                MATCH (a:Article)-[:BELONGS_TO]->(c:Cluster)
                RETURN c.id as cluster_id, count(a) as article_count
            """)
            print("\nCluster assignments in Neo4j:")
            for record in result:
                print(f"Cluster {record['cluster_id']}: {record['article_count']} articles")
            
            # Check for relationships between clusters
            result = session.run("""
                MATCH (c1:Cluster)<-[:BELONGS_TO]-(a1:Article)
                      -[r]->(a2:Article)-[:BELONGS_TO]->(c2:Cluster)
                WHERE c1 <> c2
                RETURN count(r) as relationship_count
            """)
            relationship_count = result.single()['relationship_count']
            print(f"\nFound {relationship_count} relationships between articles in different clusters")
        
        # Evaluate clustering
        metrics = self.evaluate_clustering(combined_similarity, clusters)
        print("\nClustering Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        
        # Generate visualizations
        self.visualize_clusters(combined_similarity, clusters, articles)
        
        return clusters, metrics
    
    def get_cluster_summary(self, articles: List[Dict]) -> Dict:
        """Generate summary statistics for clusters"""
        cluster_stats = {}
        for article in articles:
            cluster_id = article['cluster']
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = {
                    'size': 0,
                    'articles': [],
                    'common_terms': set(),
                    'relationships': []
                }
            
            cluster_stats[cluster_id]['size'] += 1
            cluster_stats[cluster_id]['articles'].append(article['title'])
            
            # Extract key terms
            doc = self.nlp(article['content'])
            terms = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
            cluster_stats[cluster_id]['common_terms'].update(set(terms))
        
        # Add relationship information
        with self.neo4j.driver.session() as session:
            for cluster_id in cluster_stats:
                if cluster_id != -1:
                    result = session.run("""
                        MATCH (a:Article)-[r]->(b:Article)
                        WHERE (a)-[:BELONGS_TO]->(:Cluster {id: $cluster_id})
                        AND (b)-[:BELONGS_TO]->(:Cluster {id: $cluster_id})
                        RETURN type(r) as type, count(r) as count
                    """, cluster_id=cluster_id)
                    cluster_stats[cluster_id]['relationships'] = [
                        {'type': record['type'], 'count': record['count']}
                        for record in result
                    ]
        
        # Convert sets to lists for JSON serialization
        for cluster_id in cluster_stats:
            cluster_stats[cluster_id]['common_terms'] = list(
                cluster_stats[cluster_id]['common_terms']
            )[:10]  # Keep only top 10 terms
        
        return cluster_stats 