from neo4j import GraphDatabase
from typing import List, Dict
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class neo4jHandler:
    def __init__(self, uri="bolt://localhost:7687", 
                 user="neo4j",  # Always use "neo4j" as username
                 password="12345678"):  # Please replace with your actual Neo4j password
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load("en_core_web_sm")
    
    def test_connection(self):
        """Test if the Neo4j connection is working"""
        try:
            with self.driver.session(database="financial25News") as session:
                # Try a simple query
                result = session.run("RETURN 1")
                result.single()
                return True
        except Exception as e:
            raise Exception(f"Failed to connect to Neo4j: {str(e)}")
    
    def close(self):
        self.driver.close()
    
    def __del__(self):
        self.close()
    
    def create_knowledge_node(self, title: str, content: str, summary: str):
        with self.driver.session(database="financial25News") as session:
            session.run(
                "CREATE (n:Knowledge {title: $title, content: $content, summary: $summary})",
                title=title, content=content, summary=summary
            )
    
    def create_causal_relationship(self, source_title: str, target_title: str, relationship_type: str, confidence: float):
        with self.driver.session(database="financial25News") as session:
            session.run(
                """
                MATCH (source:Knowledge {title: $source_title})
                MATCH (target:Knowledge {title: $target_title})
                CREATE (source)-[r:CAUSES {type: $relationship_type, confidence: $confidence}]->(target)
                """,
                source_title=source_title, target_title=target_title,
                relationship_type=relationship_type, confidence=confidence
            )
    
    def identify_causal_relationships(self, articles: List[Dict]):
        """Identify potential causal relationships between articles using NLP"""
        # Create nodes for all articles
        for article in articles:
            self.create_knowledge_node(
                article['title'],
                article['content'],
                article['summary']
            )
        
        # Generate embeddings for all articles
        embeddings = []
        for article in articles:
            doc = self.nlp(article['content'])
            embedding = doc.vector
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Calculate similarity matrix
        similarities = cosine_similarity(embeddings)
        
        # Create relationships for similar articles
        threshold = 0.7  # Similarity threshold
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                if similarities[i][j] > threshold:
                    # Check for causal indicators in content
                    combined_text = articles[i]['content'] + " " + articles[j]['content']
                    doc = self.nlp(combined_text)
                    
                    # Look for causal indicators
                    causal_indicators = ["because", "therefore", "thus", "hence", "consequently"]
                    if any(indicator in combined_text.lower() for indicator in causal_indicators):
                        self.create_causal_relationship(
                            articles[i]['title'],
                            articles[j]['title'],
                            "CAUSES",
                            similarities[i][j]
                        )
    
    def get_knowledge_graph(self):
        """Retrieve the entire knowledge graph"""
        with self.driver.session(database="financial25News") as session:
            result = session.run("""
                MATCH (n:Knowledge)
                OPTIONAL MATCH (n)-[r:CAUSES]->(m:Knowledge)
                RETURN collect(distinct {
                    id: id(n),
                    label: n.title,
                    content: n.content,
                    summary: n.summary
                }) as nodes,
                collect(distinct {
                    source: id(n),
                    target: id(m),
                    type: type(r),
                    confidence: r.confidence
                }) as relationships
            """)
            return result.single() 