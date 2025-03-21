#!/usr/bin/env python3
"""
Root-level script to run the Financial Knowledge System
"""
import os
import sys
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent
    python_dir = project_root / 'src' / 'python'
    
    # Add the python directory to the path
    sys.path.append(str(python_dir))
    
    # Import and run the system
    from main import FinancialKnowledgeSystem
    system = FinancialKnowledgeSystem()
    system.run()

if __name__ == "__main__":
    main() 