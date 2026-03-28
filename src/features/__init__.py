"""
Feature engineering utilities for project duration estimation.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from src.data import ProjectData, Task


class FeatureEngineer:
    """Feature engineering for project duration estimation."""
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def extract_task_features(self, project_data: ProjectData) -> pd.DataFrame:
        """
        Extract features from project tasks.
        
        Args:
            project_data: Project data
            
        Returns:
            DataFrame with task features
        """
        features = []
        
        for task in project_data.tasks:
            feature_dict = {
                "task_id": task.task_id,
                "duration": task.duration,
                "n_dependencies": len(task.dependencies),
                "risk_level_numeric": self._encode_risk_level(task.risk_level),
                "has_dependencies": len(task.dependencies) > 0,
                "optimistic_duration": task.optimistic_duration or task.duration * 0.7,
                "most_likely_duration": task.most_likely_duration or task.duration,
                "pessimistic_duration": task.pessimistic_duration or task.duration * 1.5,
                "duration_range": (task.pessimistic_duration or task.duration * 1.5) - 
                                (task.optimistic_duration or task.duration * 0.7),
                "duration_variance": self._calculate_task_variance(task)
            }
            
            # Add resource features if available
            if task.resource_requirements:
                feature_dict.update(self._extract_resource_features(task.resource_requirements))
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def extract_project_features(self, project_data: ProjectData) -> Dict[str, Any]:
        """
        Extract project-level features.
        
        Args:
            project_data: Project data
            
        Returns:
            Dictionary with project features
        """
        tasks = project_data.tasks
        
        features = {
            "n_tasks": len(tasks),
            "total_duration": sum(task.duration for task in tasks),
            "avg_task_duration": np.mean([task.duration for task in tasks]),
            "max_task_duration": max(task.duration for task in tasks),
            "min_task_duration": min(task.duration for task in tasks),
            "duration_std": np.std([task.duration for task in tasks]),
            "n_dependencies": sum(len(task.dependencies) for task in tasks),
            "avg_dependencies_per_task": np.mean([len(task.dependencies) for task in tasks]),
            "max_dependencies_per_task": max(len(task.dependencies) for task in tasks),
            "tasks_with_dependencies": sum(1 for task in tasks if len(task.dependencies) > 0),
            "dependency_density": sum(len(task.dependencies) for task in tasks) / (len(tasks) * (len(tasks) - 1)) if len(tasks) > 1 else 0,
            "risk_level_distribution": self._calculate_risk_distribution(tasks)
        }
        
        return features
    
    def _encode_risk_level(self, risk_level: str) -> int:
        """Encode risk level as numeric value."""
        risk_mapping = {"Low": 1, "Medium": 2, "High": 3}
        return risk_mapping.get(risk_level, 2)
    
    def _calculate_task_variance(self, task: Task) -> float:
        """Calculate variance for a task based on PERT estimates."""
        if (task.optimistic_duration and task.most_likely_duration and 
            task.pessimistic_duration):
            return ((task.pessimistic_duration - task.optimistic_duration) / 6) ** 2
        else:
            # Estimate variance based on duration
            return (task.duration * 0.1) ** 2
    
    def _extract_resource_features(self, resource_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from resource requirements."""
        features = {}
        
        if "team_size" in resource_requirements:
            features["team_size"] = resource_requirements["team_size"]
        
        if "skill_level" in resource_requirements:
            features["skill_level"] = resource_requirements["skill_level"]
        
        return features
    
    def _calculate_risk_distribution(self, tasks: List[Task]) -> Dict[str, float]:
        """Calculate distribution of risk levels."""
        risk_counts = {"Low": 0, "Medium": 0, "High": 0}
        
        for task in tasks:
            risk_counts[task.risk_level] += 1
        
        total = len(tasks)
        return {
            f"risk_{level.lower()}_ratio": count / total 
            for level, count in risk_counts.items()
        }
