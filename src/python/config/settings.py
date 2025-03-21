"""
Configuration settings for the Financial Knowledge System
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
VISUALIZATIONS_DIR = os.path.join(DATA_DIR, 'visualizations')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Neo4j settings
NEO4J_CONFIG = {
    'uri': "bolt://localhost:7687",
    'user': "neo4j",
    'password': "12345678"  # Please replace with your actual password
}

# Extractor settings
EXTRACTOR_CONFIG = {
    'max_workers': 5,
    'rate_limit_delay': 0.5,
    'timeout': 10,
    'retries': 3
}

# Clustering settings
CLUSTERING_CONFIG = {
    'eps': 0.5,
    'min_samples': 2,
    'content_weight': 0.7
}

# Web server settings
WEB_CONFIG = {
    'host': 'localhost',
    'port': 3000
}

# Input/Output files
FILES = {
    'input_csv': os.path.join(DATA_DIR, 'FinCatch_Sources_Medium.csv'),
    'processed_articles': os.path.join(DATA_DIR, 'processed_articles.csv'),
    'cluster_summaries': os.path.join(VISUALIZATIONS_DIR, 'cluster_summaries.json')
} 