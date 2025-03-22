import logging
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Union
from pathlib import Path
import platform


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "event_type": getattr(record, "event_type", "log"),
            "process_id": getattr(record, "process_id", None),
            "timestamp": datetime.utcnow().isoformat(),
            "message": record.getMessage(),
            "data": getattr(record,  "data", None)
        }
        return json.dumps(log_data, indent=2)

class loggerHandler:
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = Path(base_dir)
        self.process_id = str(uuid.uuid4())
        self.process_dir = self.base_dir / self.process_id
        self.process_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this process
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up logger
        self.logger = logging.getLogger(f"fincatch_{self.process_id}")
        self.logger.setLevel(logging.INFO)
        
        # File handler for JSON logs
        self.file_handler = logging.FileHandler(
            self.process_dir / f"process_{self.timestamp}.json"
        )
        self.file_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(self.file_handler)
        
        # Console handler for clean output
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.console_handler)
        
        # Create visualization directory
        self.viz_dir = self.process_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)

    def log_process_start(self) -> None:
        self.logger.info(
            "Process started",
            extra={
                "event_type": "process_start",
                "process_id": self.process_id,
                "data": {
                    "system_info": {
                        "python_version": platform.python_version(),
                        "platform": platform.platform()
                    }
                }
            }
        )

    def log_q2_start(self) -> None:
        self.logger.info(
            "Starting Q2 analysis",
            extra={
                "event_type": "q2_start",
                "process_id": self.process_id,
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "initiated"
                }
            }
        )

    def log_q2_result(self, result: Dict[str, Any]) -> None:
        self.logger.info(
            "Q2 analysis completed",
            extra={
                "event_type": "q2_result",
                "process_id": self.process_id,
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    **result
                }
            }
        )

    def log_q3_start(self) -> None:
        self.logger.info(
            "Starting Q3 analysis",
            extra={
                "event_type": "q3_start",
                "process_id": self.process_id,
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "initiated"
                }
            }
        )

    def log_q3_result(self, result: Dict[str, Any]) -> None:
        self.logger.info(
            "Q3 analysis completed",
            extra={
                "event_type": "q3_result",
                "process_id": self.process_id,
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    **result
                }
            }
        )

    def log_error(self, error: Exception, context: Dict[str, Any] | None = None) -> None:
        self.logger.error(
            str(error),
            extra={
                "event_type": "error",
                "process_id": self.process_id,
                "data": {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "context": context or {},
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

    def log_process_end(self) -> None:
        self.logger.info(
            "Process completed",
            extra={
                "event_type": "process_end",
                "process_id": self.process_id,
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
            }
        )

    def save_visualization(self, name: str, data: bytes) -> None:
        """Save visualization data (e.g., PNG) to the process's visualization directory."""
        file_path = self.viz_dir / f"{name}_{self.timestamp}.png"
        with open(file_path, "wb") as f:
            f.write(data)
        self.logger.info(
            f"Saved visualization: {name}",
            extra={
                "event_type": "visualization_saved",
                "process_id": self.process_id,
                "data": {
                    "name": name,
                    "path": str(file_path),
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
