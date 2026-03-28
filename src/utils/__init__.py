"""
Utility functions and helpers for project duration estimation.
"""

import logging
import random
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import yaml
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("project_duration")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set environment variables for other libraries
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def create_output_directory(base_dir: Union[str, Path], 
                           subdirs: Optional[List[str]] = None) -> Path:
    """
    Create output directory structure.
    
    Args:
        base_dir: Base output directory
        subdirs: Optional list of subdirectories to create
        
    Returns:
        Path to base directory
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    if subdirs:
        for subdir in subdirs:
            (base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    return base_path


def format_duration(days: float) -> str:
    """
    Format duration in days to human-readable string.
    
    Args:
        days: Duration in days
        
    Returns:
        Formatted duration string
    """
    if days < 1:
        hours = days * 24
        return f"{hours:.1f} hours"
    elif days < 7:
        return f"{days:.1f} days"
    elif days < 30:
        weeks = days / 7
        return f"{weeks:.1f} weeks"
    elif days < 365:
        months = days / 30
        return f"{months:.1f} months"
    else:
        years = days / 365
        return f"{years:.1f} years"


def calculate_percentile_rank(value: float, data: List[float]) -> float:
    """
    Calculate percentile rank of a value in a dataset.
    
    Args:
        value: Value to rank
        data: Dataset
        
    Returns:
        Percentile rank (0-100)
    """
    if not data:
        return 0.0
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Count values less than the given value
    count_less = sum(1 for x in sorted_data if x < value)
    
    # Calculate percentile rank
    percentile_rank = (count_less / n) * 100
    
    return percentile_rank


def validate_project_data(project_data: Any) -> bool:
    """
    Validate project data structure.
    
    Args:
        project_data: Project data to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if project_data has required attributes
        if not hasattr(project_data, 'tasks'):
            return False
        
        if not hasattr(project_data, 'project_name'):
            return False
        
        # Check tasks
        if not isinstance(project_data.tasks, list):
            return False
        
        for task in project_data.tasks:
            if not hasattr(task, 'task_id'):
                return False
            if not hasattr(task, 'duration'):
                return False
            if not hasattr(task, 'dependencies'):
                return False
            
            # Check duration is positive
            if task.duration <= 0:
                return False
            
            # Check dependencies are valid task IDs
            if not isinstance(task.dependencies, list):
                return False
        
        return True
    
    except Exception:
        return False


def export_results_to_json(results: Dict[str, Any], 
                          output_path: Union[str, Path]) -> None:
    """
    Export results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    converted_results = convert_numpy_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(converted_results, f, indent=2, default=str)


def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Timestamp string in YYYY-MM-DD_HH-MM-SS format
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def calculate_risk_score(duration: float, 
                        variance: float, 
                        critical_path_length: int,
                        total_tasks: int) -> float:
    """
    Calculate project risk score based on various factors.
    
    Args:
        duration: Project duration
        variance: Duration variance
        critical_path_length: Length of critical path
        total_tasks: Total number of tasks
        
    Returns:
        Risk score between 0 and 1
    """
    # Normalize factors
    duration_factor = min(1.0, duration / 100)  # Assume 100 days is high risk
    variance_factor = min(1.0, variance / 25)  # Assume variance of 25 is high risk
    criticality_factor = critical_path_length / total_tasks if total_tasks > 0 else 0
    
    # Weighted combination
    risk_score = (
        0.3 * duration_factor +
        0.4 * variance_factor +
        0.3 * criticality_factor
    )
    
    return min(1.0, max(0.0, risk_score))


def generate_project_summary(project_data: Any, 
                           cpm_result: Any,
                           pert_result: Optional[Any] = None,
                           mc_result: Optional[Any] = None) -> Dict[str, Any]:
    """
    Generate comprehensive project summary.
    
    Args:
        project_data: Project data
        cpm_result: CPM analysis results
        pert_result: Optional PERT analysis results
        mc_result: Optional Monte Carlo simulation results
        
    Returns:
        Project summary dictionary
    """
    summary = {
        "project_info": {
            "name": project_data.project_name,
            "total_tasks": len(project_data.tasks),
            "total_duration": sum(task.duration for task in project_data.tasks),
            "avg_task_duration": np.mean([task.duration for task in project_data.tasks]),
            "generated_at": get_timestamp()
        },
        "cpm_analysis": {
            "project_duration": cpm_result.duration,
            "critical_path_length": len(cpm_result.critical_path),
            "critical_path": cpm_result.critical_path,
            "total_slack_time": sum(cpm_result.slack_times.values()),
            "critical_tasks": sum(1 for slack in cpm_result.slack_times.values() if slack == 0)
        }
    }
    
    if pert_result:
        summary["pert_analysis"] = {
            "expected_duration": pert_result.project_expected_duration,
            "standard_deviation": pert_result.project_standard_deviation,
            "variance": pert_result.project_variance,
            "critical_path_probability": pert_result.critical_path_probability,
            "confidence_intervals": pert_result.confidence_intervals
        }
    
    if mc_result:
        summary["monte_carlo_analysis"] = {
            "mean_duration": mc_result.mean_duration,
            "median_duration": mc_result.median_duration,
            "standard_deviation": mc_result.std_duration,
            "percentiles": mc_result.percentiles,
            "confidence_intervals": mc_result.confidence_intervals
        }
    
    # Calculate overall risk score
    variance = pert_result.project_variance if pert_result else mc_result.std_duration ** 2 if mc_result else 0
    risk_score = calculate_risk_score(
        cpm_result.duration,
        variance,
        len(cpm_result.critical_path),
        len(project_data.tasks)
    )
    
    summary["risk_assessment"] = {
        "overall_risk_score": risk_score,
        "risk_level": "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"
    }
    
    return summary
