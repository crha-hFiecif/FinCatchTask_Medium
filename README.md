# FinCatchTask_Medium

## System Design

### Interactive Menu System
The system features an interactive menu-driven interface that allows users to:
1. Configure system settings at startup
2. Choose specific tasks to run
3. View results and visualizations

#### Initial Setup
When starting the system, users are prompted to provide:
- Neo4j database credentials:
  - Database name (defaults to "neo4j")
  - Password
- CSV file path for article sources (defaults to `data/FinCatch_Sources_Medium.csv`)
  - After providing the path (or pressing Enter for default), the system automatically Q1:
    - Validates the CSV file exists
    - Runs the article extraction process
    - Shows progress and completion status

#### Main Menu Options
```
=== FinCatchTask System Menu ===
0: Run Q2 - Build Relationships and Start Visualization Server
1: Run Q3 - Perform Clustering Analysis
2: Exit System
```

#### Task Descriptions

##### Q2 - Relationship Building and Visualization
- Builds causal relationships between articles
- Starts a web visualization server
- Access visualization at http://localhost:3000
- Shows real-time relationship graphs
- Can be stopped with Ctrl+C

##### Q3 - Clustering Analysis
- Performs DBSCAN clustering on articles
- Generates multiple visualizations:
  - Cluster size distribution
  - t-SNE visualization
  - Similarity matrix heatmap
  - Cluster relationship graph
- Saves visualizations to `data/visualizations/`
- Shows clear output paths in terminal

### Output Locations
- Q2 Visualizations: Web interface at http://localhost:3000
- Q3 Visualizations: `data/visualizations/` directory
  - `cluster_sizes.png`: Distribution of articles across clusters
  - `tsne_clusters.png`: 2D visualization of article groupings
  - `similarity_matrix.png`: Article similarity heatmap
  - `cluster_relationships.png`: Cluster connection graph
- Detailed logs will be saved to logs/ 
  - system_YYYYMMDD_HHMMSS.log:
  - All system operations
  - Debug information
  - Error details
  - Timestamps and log levels

### Error Handling
- Graceful handling of Neo4j connection issues
- Automatic retries for failed operations
- Clear error messages and logging
- Safe cleanup of resources on exit


## Quick Start

### Prerequisites
- Python 3.11
- Neo4j Desktop installed locally

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Source\activate    # On Window
```

3. Install required packages:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Set up Neo4j:
- Open Neo4j Desktop
- Create a new database (if not already created)
- Set the password to "fincatch"
- Start the database
- Note: The default connection string is `bolt://localhost:7687`

### Running the System
 Run the complete system:
```bash
cd ../..
python3 run_system.py
```
--
### Requirements
1. Q1 Extractor - ✅ COMPLETED
- Parallel processing with ThreadPoolExecutor
- Rate limiting and error handling
- Scalable design for large numbers of articles
- Well-commented code
- Progress tracking and logging

Q2 Causal Relationship Visualizer - ✅ COMPLETED
-  Neo4j integration
- Causal relationship identification
-  Visual interface with D3.js
-  Graph database storage and querying

Q3 Clustering Module - ✅ COMPLETED
- DBSCAN clustering implementation
-  Cluster quality evaluation
-  Integration with Neo4j
-  Visualization support

--

### Challenges and Solutions:
1. Scalability:
- Challenge: Processing large numbers of articles efficiently
- Solution: Implemented parallel processing with ThreadPoolExecutor and rate limiting
- Trade-off: Balanced between speed and server load

2. Accuracy:
- Challenge: Extracting meaningful content and relationships
- Solution: Used spaCy for NLP and DBSCAN for clustering
- Trade-off: Chose DBSCAN over k-means because it doesn't require predefined cluster numbers

3. Performance:
- Challenge: Handling network requests and database operations
- Solution: Implemented connection pooling, retries, and error handling
- Trade-off: Added complexity for better reliability
