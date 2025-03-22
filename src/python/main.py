"""
Main entry point for the Financial Knowledge System
"""
import logging
import subprocess
import sys
import time
from pathlib import Path
import signal
import os
from typing import Tuple, Optional

from modules import articleExtractor, relationshipBuilder, knowledgeCluster
from utils.neo4j_handler import neo4jHandler

import select
from datetime import datetime

from utils.logger_handler import loggerHandler

# Create logs directory
project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)

# Setup logging
log_file = logs_dir / f'system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a separate logger for terminal output
terminal_logger = logging.getLogger('terminal')
terminal_logger.setLevel(logging.INFO)
terminal_handler = logging.StreamHandler(sys.stdout)
terminal_handler.setFormatter(logging.Formatter('%(message)s'))
terminal_logger.addHandler(terminal_handler)

class SystemError(Exception):
    """Base exception for system errors"""
    pass

class Neo4jConnectionError(SystemError):
    """Raised when Neo4j connection fails"""
    pass

class ComponentError(SystemError):
    """Raised when a component fails"""
    pass

class WebServerError(SystemError):
    """Raised when web server fails"""
    pass

class FinancialKnowledgeSystem:
    def __init__(self):
        self.logger_handler = loggerHandler()
        self.logger = self.logger_handler.logger
        self.logger_handler.log_process_start()
        self.project_root = Path(__file__).parent.parent.parent
        self.python_dir = self.project_root / 'src' / 'python'
        self.web_dir = self.project_root / 'src' / 'web'
        self.web_process = None
        self.neo4j = None  # Will be initialized with user credentials
        self.csv_path = None  # Will be set by user input
        
    def get_user_input(self):
        """Get Neo4j credentials and CSV path from user"""
        terminal_logger.info("\n=== Task System Setup ===")
        
        # Get Neo4j credentials
        terminal_logger.info("\nPlease enter your Neo4j database credentials:")
        db_name = input("Database name (default: financial25News): ").strip() or "financial25News"
        password = input("Password: ").strip()
        
        # Initialize Neo4j handler with user credentials
        self.neo4j = neo4jHandler(
            uri="bolt://localhost:7687",
            user="neo4j",  # Always use "neo4j" as username
            password=password
        )
        
        # Get CSV path
        default_csv = self.project_root / 'data' / 'FinCatch_Sources_Medium.csv'
        terminal_logger.info(f"\nEnter the path to your CSV file (press Enter for default: {default_csv}):")
        csv_input = input().strip()
        
        if csv_input:
            self.csv_path = Path(csv_input)
        else:
            self.csv_path = default_csv
            
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        terminal_logger.info(f"\nUsing CSV file: {self.csv_path}")
        
        # Run extraction automatically
        terminal_logger.info("\nStarting article extraction...")
        if not self.run_python_component('extractor.py'):
            raise ComponentError("Failed to extract articles")
        terminal_logger.info("✅ Article extraction completed!")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle system signals gracefully"""
        logging.info(f"Received signal {signum}")
        self.cleanup()
        sys.exit(0)
        
    def cleanup(self):
        """Clean up system resources"""
        logging.info("Cleaning up resources...")
        if self.web_process and self.web_process.poll() is None:
            try:
                logging.info("Attempting to terminate web server gracefully...")
                if os.name == 'nt':  # Windows
                    self.web_process.send_signal(signal.CTRL_BREAK_EVENT)  # Windows-specific
                else:  # Unix-like systems
                    self.web_process.terminate()
                self.web_process.wait(timeout=5)
                logging.info("Web server terminated successfully")
            except subprocess.TimeoutExpired:
                logging.warning("Web server did not terminate gracefully, forcing kill...")
                if os.name == 'nt':  # Windows
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.web_process.pid)])
                else:  # Unix-like systems
                    self.web_process.kill()
                logging.info("Web server killed successfully")
            except Exception as e:
                logging.error(f"Error during cleanup: {str(e)}")
        else:
            logging.info("No web server process to clean up")
        
    def check_neo4j(self) -> bool:
        """Check if Neo4j is running"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.neo4j.test_connection()
                logging.info("✅ Neo4j connection successful")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Neo4j connection attempt {attempt + 1} failed: {str(e)}")
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"❌ Neo4j connection failed after {max_retries} attempts: {str(e)}")
                    logging.error("Please make sure Neo4j is running and credentials are correct")
                    raise Neo4jConnectionError(str(e))
        return False

    def run_python_component(self, script_name: str) -> bool:
        """Run a Python component and return success status"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                script_path = self.python_dir / 'modules' / script_name
                if not script_path.exists():
                    raise ComponentError(f"Script not found: {script_path}")
                
                logging.info(f"Running {script_name}...")
                
                # Use appropriate Python executable path
                python_executable = sys.executable
                if os.name == 'nt':  # Windows
                    python_executable = str(python_executable).replace('\\', '/')
                
                result = subprocess.run(
                    [python_executable, str(script_path)],
                    cwd=self.python_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    shell=True if os.name == 'nt' else False  # Use shell on Windows
                )
                
                if result.stdout:
                    logging.info(result.stdout)
                if result.stderr:
                    logging.warning(result.stderr)
                    
                logging.info(f"✅ {script_name} completed successfully")
                return True
                
            except subprocess.CalledProcessError as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Component {script_name} failed attempt {attempt + 1}: {str(e)}")
                    logging.info("Retrying...")
                    continue
                logging.error(f"❌ {script_name} failed: {str(e)}")
                if e.output:
                    logging.error(e.output)
                raise ComponentError(f"Component {script_name} failed: {str(e)}")
            except Exception as e:
                logging.error(f"❌ Unexpected error in {script_name}: {str(e)}")
                raise ComponentError(str(e))
        return False

    def start_web_server(self) -> Tuple[bool, Optional[subprocess.Popen]]:
        """Start the web visualization server"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logging.info("Starting web server...")
                # Kill any existing process on port 3000
                try:
                    logging.info("Checking for existing web server process...")
                    if os.name == 'nt':  # Windows
                        subprocess.run(
                            ["netstat -ano | findstr :3000"],
                            shell=True,
                            capture_output=True,
                            text=True
                        )
                    else:  # Unix-like systems
                        subprocess.run(
                            ["lsof -ti:3000 | xargs kill -9"],
                            shell=True,
                            stderr=subprocess.DEVNULL
                        )
                    logging.info("Any existing web server process has been terminated")
                except Exception:
                    logging.info("No existing web server process found")
                
                # Check if package.json exists
                if not (self.web_dir / 'package.json').exists():
                    raise WebServerError("package.json not found in web directory")
                
                # Use appropriate command based on OS
                if os.name == 'nt':  # Windows
                    web_process = subprocess.Popen(
                        'npm.cmd start',  # Use npm.cmd on Windows
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=self.web_dir,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # Windows-specific
                    )
                else:  # Unix-like systems
                    web_process = subprocess.Popen(
                        'npm start',
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=self.web_dir
                    )
                
                # Wait for server to start
                for _ in range(30):  # Wait up to 30 seconds
                    if web_process.poll() is not None:
                        stdout, stderr = web_process.communicate()
                        raise WebServerError(f"Server failed to start: {stderr.decode()}")
                    
                    # Try to connect to the server
                    try:
                        import socket
                        with socket.create_connection(('localhost', 3000), timeout=1):
                            logging.info("✅ Web server started successfully")
                            logging.info("Visit http://localhost:3000 to view the visualization")
                            self.web_process = web_process
                            return True, web_process
                    except Exception:
                        time.sleep(1)
                        continue
                
                raise WebServerError("Server startup timeout")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Web server start attempt {attempt + 1} failed: {str(e)}")
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                logging.error(f"❌ Web server failed to start: {str(e)}")
                return False, None
        
        return False, None

    def show_menu(self):
        """Display the main menu"""
        terminal_logger.info("\n=== Task System Menu ===")
        terminal_logger.info("0: Run Q2 - Build Relationships and Start Visualization Server")
        terminal_logger.info("1: Run Q3 - Perform Clustering Analysis")
        terminal_logger.info("2: Exit System")
        
    def run_q2(self):
        """Run Q2 - Build relationships and start visualization server"""
        try:
            self.logger_handler.log_q2_start()
            logging.info("Starting Q2 - Building relationships...")
            
            # Check Neo4j
            self.check_neo4j()
            
            # Run build_relationships.py
            if not self.run_python_component('build_relationships.py'):
                raise ComponentError("Failed to build relationships")
            
            # Start web server
            success, web_process = self.start_web_server()
            if not success:
                raise WebServerError("Failed to start web server")
            
            terminal_logger.info("\n🎉 Q2 is running!")
            terminal_logger.info("Visit http://localhost:3000 to view the visualization")
            terminal_logger.info("Press 'q' end the web server and return to menu")
            
            
            while web_process.poll() is None:
                if sys.stdin.isatty():  
                    if select.select([sys.stdin], [], [], 0.0)[0]: 
                        if sys.stdin.read(1) == 'q':
                            logging.info("Stopping web server and returning to menu...")
                            self.cleanup()
                            return
                time.sleep(1)
                
            # Save visualization
            if hasattr(self, 'graph_data'):
                self.logger_handler.save_visualization('q2_graph', self.graph_data)
            
            self.logger_handler.log_q2_result({
                'num_nodes': len(self.graph_data['nodes']),
                'num_edges': len(self.graph_data['links']),
                'visualization_path': str(self.logger_handler.viz_dir / f"q2_graph_{self.logger_handler.timestamp}.png")
            })
            
        except Exception as e:
            self.logger_handler.log_error(e)
            raise
            
    def run_q3(self):
        """Run Q3 - Perform clustering analysis"""
        try:
            self.logger_handler.log_q3_start()
            logging.info("Starting Q3 - Clustering analysis...")
            
            # Check Neo4j
            self.check_neo4j()
            
            # Create visualization directory
            viz_dir = self.project_root / 'data' / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"📊 Clustering visualizations will be saved to: {viz_dir}")
            
            # Run clustering.py
            if not self.run_python_component('clustering.py'):
                raise ComponentError("Failed to perform clustering")
            
            terminal_logger.info("\n✅ Clustering analysis completed!")
            terminal_logger.info(f"📈 Visualizations are available in: {viz_dir}")
            
            # Get cluster data from Neo4j
            try:
                from modules.clustering import knowledgeCluster
                import json
                
                clusterer = knowledgeCluster()
                cluster_data = clusterer.get_cluster_data()
                
                # Save visualization if cluster data exists
                if cluster_data:
                    # Save cluster data as JSON
                    cluster_json_path = self.logger_handler.viz_dir / f"q3_clusters_{self.logger_handler.timestamp}.json"
                    with open(cluster_json_path, 'w') as f:
                        json.dump(cluster_data, f, indent=2)
                    
                    self.logger_handler.log_q3_result({
                        'num_clusters': len(cluster_data),
                        'visualization_path': str(cluster_json_path)
                    })
                else:
                    self.logger_handler.log_q3_result({
                        'status': 'completed',
                        'message': 'No cluster data available'
                    })
            except Exception as e:
                self.logger_handler.log_error(e, {'context': 'Failed to process cluster data'})
                raise
            
        except Exception as e:
            self.logger_handler.log_error(e)
            raise
            
    def run_menu(self):
        """Run the interactive menu"""
        while True:
            print("\n=== FinCatch System Menu ===")
            print("1: Run relationship analysis (Q2)")
            print("2: Run clustering analysis (Q3)")
            print("q: Quit")
            print("\nNote: Please run options in order (1 → 2)")
            
            choice = input("\nEnter your choice: ").strip().lower()
            
            try:
                if choice == '1':
                    # Check if extraction has been run
                    with self.neo4j.driver.session() as session:
                        result = session.run("MATCH (a:Article) RETURN count(a) as count")
                        article_count = result.single()['count']
                        if article_count == 0:
                            print("\n❌ Error: No articles found in database.")
                            print("Please restart the system to extract articles.")
                            continue
                    self.run_q2()
                elif choice == '2':
                    # Check if both extraction and relationship analysis have been run
                    with self.neo4j.driver.session() as session:
                        # Check for articles
                        result = session.run("MATCH (a:Article) RETURN count(a) as count")
                        article_count = result.single()['count']
                        if article_count == 0:
                            print("\n❌ Error: No articles found in database.")
                            print("Please restart the system to extract articles.")
                            continue
                        
                        # Check for relationships
                        result = session.run("""
                            MATCH (a:Article)-[r]->(b:Article)
                            RETURN count(r) as count
                        """)
                        relationship_count = result.single()['count']
                        if relationship_count == 0:
                            print("\n❌ Error: No relationships found between articles.")
                            print("Please run option '1' (Relationship analysis) first.")
                            continue
                    self.run_q3()
                elif choice == 'q':
                    print("\nStopping system...")
                    break
                else:
                    print("\n❌ Invalid choice. Please enter one of the following:")
                    print("- 1: Run relationship analysis")
                    print("- 2: Run clustering analysis")
                    print("- q: Quit")
                
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or enter 'q' to quit.")

    def run(self):
        """Run the complete system with interactive menu"""
        try:
            terminal_logger.info(f"\nLog file: {log_file}")
            terminal_logger.info("Detailed logs will be saved to this file")
            
            # Get user input first
            self.get_user_input()
            
            self.run_menu()
            
        except KeyboardInterrupt:
            terminal_logger.info("\nShutting down gracefully...")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        finally:
            self.cleanup()
            terminal_logger.info("System stopped")

if __name__ == "__main__":
    system = FinancialKnowledgeSystem()
    system.run() 