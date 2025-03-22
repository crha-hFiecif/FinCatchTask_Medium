import pandas as pd
import spacy
import logging
from neo4j import GraphDatabase
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class relationshipBuilder:
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "12345678"):
        """Initialize the relationship builder with Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load('en_core_web_sm')
        
        # Causal indicators for relationship detection
        self.causal_indicators = {
            'causes': ['cause', 'lead to', 'result in', 'drive', 'affect'],
            'influenced_by': ['depend on', 'influenced by', 'affected by', 'determined by'],
            'correlates_with': ['correlate', 'associated with', 'related to', 'linked to']
        }

    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()

    def create_article_node(self, tx, article: Dict):
        """Create a node for an article in Neo4j"""
        query = """
        MERGE (a:Article {title: $title})
        SET a.content = $content,
            a.summary = $summary,
            a.url = $url,
            a.source = $source
        RETURN a
        """
        tx.run(query, title=article['title'], content=article['content'],
               summary=article['summary'], url=article['url'],
               source=article['source'])

    def find_causal_relationships(self, text1: str, text2: str) -> List[Tuple[str, float]]:
        """Find causal relationships between two texts"""
        relationships = []
        combined_text = f"{text1} {text2}"
        doc = self.nlp(combined_text.lower())
        
        # Check for causal indicators
        for rel_type, indicators in self.causal_indicators.items():
            for indicator in indicators:
                if indicator in combined_text.lower():
                    # Calculate confidence based on proximity of terms
                    confidence = 0.7  # Base confidence
                    # Adjust confidence based on document similarity
                    doc1 = self.nlp(text1)
                    doc2 = self.nlp(text2)
                    similarity = doc1.similarity(doc2)
                    confidence = (confidence + similarity) / 2
                    relationships.append((rel_type, confidence))
        
        return relationships

    def calculate_content_similarity(self, articles: List[Dict]) -> np.ndarray:
        """Calculate content similarity matrix between articles"""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([art['content'] for art in articles])
        return cosine_similarity(tfidf_matrix)

    def create_relationships(self, articles: List[Dict]):
        """Create relationships between articles in Neo4j"""
        with self.driver.session() as session:
            # First, create all article nodes
            for article in articles:
                session.execute_write(self.create_article_node, article)
            
            # Calculate content similarity
            similarity_matrix = self.calculate_content_similarity(articles)
            
            # Create relationships based on similarity and causal indicators
            for i, article1 in enumerate(articles):
                for j, article2 in enumerate(articles[i+1:], i+1):
                    # Check if articles are similar enough
                    if similarity_matrix[i, j] > 0.3:  # Similarity threshold
                        # Find causal relationships
                        relationships = self.find_causal_relationships(
                            article1['content'],
                            article2['content']
                        )
                        
                        # Create relationships in Neo4j
                        for rel_type, confidence in relationships:
                            session.run("""
                            MATCH (a1:Article {title: $title1})
                            MATCH (a2:Article {title: $title2})
                            MERGE (a1)-[r:%s {confidence: $confidence}]->(a2)
                            """ % rel_type,
                            title1=article1['title'],
                            title2=article2['title'],
                            confidence=confidence)
                            
                            logging.info(f"Created {rel_type} relationship between "
                                       f"'{article1['title']}' and '{article2['title']}' "
                                       f"with confidence {confidence:.2f}")

def main():
    """Main function to build relationships"""
    try:
        # Load processed articles
        df = pd.read_csv('../../data/processed_articles.csv')
        articles = df.to_dict('records')
        
        # Initialize and run relationship builder
        builder = relationshipBuilder()
        builder.create_relationships(articles)
        
        logging.info("Successfully built relationships in Neo4j")
        
    except Exception as e:
        logging.error(f"Error building relationships: {str(e)}")
    finally:
        if 'builder' in locals():
            builder.close()

if __name__ == "__main__":
    main() 