# FinCatch MediumQ: Financial News Causal Analysis System

> Below is a causal gui demo

https://github.com/user-attachments/assets/5ea393b7-d9cc-4dd7-9513-a57770734c6d


## System Design

### Interactive Menu System
The system features an interactive menu-driven interface that allows users to:
1. Prompt for Neo4j credentials at startup
2. Extract articles from the CSV file
3. Show an interactive menu for running Q2 (relationship visualization) and Q3 (clustering analysis)
4. Create necessary directories (logs, visualizations)

#### Initial Setup
When starting the system, users are prompted to provide:
- Neo4j database credentials:
  - Database name (or push `enter`, defaults to "financial25News")
  - Password
- CSV file path for article sources (or push `enter`, defaults to `data/FinCatch_Sources_Medium.csv`)
  - After providing the path (or pressing Enter for default), the system automatically Q1:
    - Validates the CSV file exists
    - Runs the article extraction process
    - Shows progress and completion status
    - Saves processed article to `data/processed_articles.csv`

#### Main Menu Options
```
=== FinCatch System Menu ===
1: Run relationship analysis (Q2)
2: Run clustering analysis (Q3)
q: Quit

Note: Please run options in order (1 → 2)
```

#### Task Descriptions

##### Q2 - Relationship Building and Visualization
- Builds causal relationships between articles
- Starts a web visualization server
- Access visualization at http://localhost:3000
- Shows real-time relationship graphs
- Press '1' to start Q2
- Press 'q' to return to menu

##### Q3 - Clustering Analysis
- Performs DBSCAN clustering on articles
- Generates multiple visualizations:
  - Cluster size distribution
  - t-SNE visualization
  - Similarity matrix heatmap
  - Cluster relationship graph
- Saves visualizations to `.data/visualizations/`
- Saves cluster details to `.logs/uuid/visualizations`
- Shows clear output paths in terminal
- Press '2' to start Q3

### Output Locations
- Q2 Visualizations: Web interface at http://localhost:3000
- Q3 Visualizations: `data/visualizations/` 
  - `cluster_sizes.png`: Distribution of articles across clusters
  - `tsne_clusters.png`: 2D visualization of article groupings
  - `similarity_matrix.png`: Article similarity heatmap
  - `cluster_relationships.png`: Cluster connection graph
- Detailed logs will be saved to `.logs/ `
  - system_YYYYMMDD_HHMMSS.log:
  - All system operations
  - Debug information
  - Error details
  - Q3 cluster idx details
  - Timestamps and log levels

### Error Handling
- Graceful handling of Neo4j connection issues
- Automatic retries for failed operations
- Clear error messages and logging
- Safe cleanup of resources on exit


## Quick Start

### Prerequisites
- Python Requirements: *Python(3.11)*
    - All required packages installed via `pip install -r requirements.txt` in the root
- Node.js Requirements: *Node.js(v23.5.0)*, *npm(10.9.2)*
    - Web dependencies installed via npm install in the `src/web` directory
- Neo4j Desktop installed locally: *Neo4j(5.24.0)*
    - Default port: 7687
    - Default username: "neo4j"
- Operation System: Windows / Linux / MacOS Linux / Window(WSL)


1. Python Setup:
```bash
# Create and activate a virtual environment in the root:
python -m venv venv

source venv/bin/activate # On macOS/Linux

venv\Source\activate  # On Window 

# Install required packages:
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Web Interface Setup:
```bash
# Navigate to web directory
cd src/web

# Install Node.js dependencies
npm install
```

3. Set up Neo4j:
- Open Neo4j Desktop
- Create a new database (if not already created)
- Set the password to "the password of Your Database"
- Start the database
- Note: The default connection string is `bolt://localhost:7687`

4. Running the System:
```bash
cd ../..
python3 run_system.py
```

### TroubleShoot
- Check on the `.logs/uuid/process_datetime.json` to debug

#### Common Issues
1. **Port 3000 in use**
   - The system will automatically attempt to kill any existing process
   - If issues persist, manually check and kill the process

2. **Neo4j Connection**
   - Ensure Neo4j is running
   - Verify credentials
   - Check if port 7687 is accessible

3. **Web Server**
   - If the web interface doesn't start, check Node.js and npm installation
   - Verify all dependencies are installed in `src/web`

#### Platform-Specific Notes

##### Windows
- Uses `npm.cmd` instead of `npm`
- Uses Windows-specific process management
- Paths are automatically handled for Windows compatibility

----

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

----

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

4. Visualization Generation
- Challenge: t-SNE visualization failing with precomputed distance matrices
- Solution: Modified initialization parameters and added graceful error handling
- Trade-off: Slightly longer computation time for more reliable visualization

5. Cross-Platform Compatibility
- Challenge: Ensuring the system works consistently across Windows and Unix-based systems
- Solution: Implemented OS-specific checks and commands, unified path handling
- Trade-off: Additional code complexity for better platform support

----

### Improvement

1. Enhanced Error Recovery and System Resilience
- Implement automatic retry mechanisms for failed API calls or database operations
- Add system state persistence to allow resuming from the last successful state


2. Improved Visualization and User Interface

3. Advanced Analysis Features
