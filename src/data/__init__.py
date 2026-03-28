"""
Data loading and synthetic data generation for project duration estimation.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass
import random
from pathlib import Path
import json


@dataclass
class Task:
    """Represents a project task with its properties."""
    task_id: str
    name: str
    duration: float
    dependencies: List[str]
    resource_requirements: Optional[Dict[str, Any]] = None
    risk_level: str = "Medium"
    optimistic_duration: Optional[float] = None
    most_likely_duration: Optional[float] = None
    pessimistic_duration: Optional[float] = None


@dataclass
class ProjectData:
    """Container for project data including tasks and metadata."""
    tasks: List[Task]
    project_name: str
    metadata: Dict[str, Any]


class SyntheticDataGenerator:
    """Generates synthetic project data for testing and demonstration."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_sample_project(self, 
                              n_tasks: int = 10,
                              min_duration: float = 1.0,
                              max_duration: float = 10.0,
                              dependency_probability: float = 0.3) -> ProjectData:
        """
        Generate a synthetic project with random tasks and dependencies.
        
        Args:
            n_tasks: Number of tasks to generate
            min_duration: Minimum task duration
            max_duration: Maximum task duration
            dependency_probability: Probability of creating dependencies between tasks
            
        Returns:
            ProjectData object containing generated tasks
        """
        tasks = []
        task_ids = [f"Task_{i:02d}" for i in range(n_tasks)]
        
        # Generate tasks with random durations
        for i, task_id in enumerate(task_ids):
            duration = np.random.uniform(min_duration, max_duration)
            
            # Generate dependencies (only to previous tasks to avoid cycles)
            dependencies = []
            if i > 0 and np.random.random() < dependency_probability:
                # Randomly select 1-3 previous tasks as dependencies
                n_deps = np.random.randint(1, min(4, i + 1))
                dependencies = np.random.choice(task_ids[:i], n_deps, replace=False).tolist()
            
            # Generate PERT estimates
            optimistic = duration * 0.7
            most_likely = duration
            pessimistic = duration * 1.5
            
            # Assign risk levels based on duration and dependencies
            risk_level = "Low"
            if duration > (max_duration - min_duration) * 0.7 + min_duration:
                risk_level = "High"
            elif len(dependencies) > 2:
                risk_level = "Medium"
            
            task = Task(
                task_id=task_id,
                name=f"Task {i+1}",
                duration=duration,
                dependencies=dependencies,
                resource_requirements={"team_size": np.random.randint(1, 5)},
                risk_level=risk_level,
                optimistic_duration=optimistic,
                most_likely_duration=most_likely,
                pessimistic_duration=pessimistic
            )
            tasks.append(task)
        
        metadata = {
            "generation_params": {
                "n_tasks": n_tasks,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "dependency_probability": dependency_probability,
                "seed": self.seed
            },
            "total_duration": sum(task.duration for task in tasks),
            "avg_duration": np.mean([task.duration for task in tasks]),
            "max_dependencies": max(len(task.dependencies) for task in tasks)
        }
        
        return ProjectData(
            tasks=tasks,
            project_name="Synthetic Project",
            metadata=metadata
        )


class DataLoader:
    """Handles loading and saving of project data."""
    
    @staticmethod
    def project_to_dataframe(project_data: ProjectData) -> pd.DataFrame:
        """Convert ProjectData to pandas DataFrame."""
        data = []
        for task in project_data.tasks:
            data.append({
                "task_id": task.task_id,
                "name": task.name,
                "duration": task.duration,
                "dependencies": ",".join(task.dependencies) if task.dependencies else "",
                "resource_requirements": json.dumps(task.resource_requirements) if task.resource_requirements else "",
                "risk_level": task.risk_level,
                "optimistic_duration": task.optimistic_duration,
                "most_likely_duration": task.most_likely_duration,
                "pessimistic_duration": task.pessimistic_duration
            })
        return pd.DataFrame(data)
    
    @staticmethod
    def dataframe_to_project(df: pd.DataFrame, project_name: str = "Project") -> ProjectData:
        """Convert pandas DataFrame to ProjectData."""
        tasks = []
        for _, row in df.iterrows():
            dependencies = row["dependencies"].split(",") if row["dependencies"] else []
            dependencies = [dep.strip() for dep in dependencies if dep.strip()]
            
            resource_requirements = None
            if row["resource_requirements"]:
                try:
                    resource_requirements = json.loads(row["resource_requirements"])
                except json.JSONDecodeError:
                    resource_requirements = None
            
            task = Task(
                task_id=row["task_id"],
                name=row["name"],
                duration=float(row["duration"]),
                dependencies=dependencies,
                resource_requirements=resource_requirements,
                risk_level=row["risk_level"],
                optimistic_duration=row.get("optimistic_duration"),
                most_likely_duration=row.get("most_likely_duration"),
                pessimistic_duration=row.get("pessimistic_duration")
            )
            tasks.append(task)
        
        return ProjectData(tasks=tasks, project_name=project_name, metadata={})
    
    @staticmethod
    def save_project(project_data: ProjectData, filepath: str) -> None:
        """Save project data to file."""
        df = DataLoader.project_to_dataframe(project_data)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
    
    @staticmethod
    def load_project(filepath: str, project_name: str = None) -> ProjectData:
        """Load project data from file."""
        df = pd.read_csv(filepath)
        if project_name is None:
            project_name = Path(filepath).stem
        return DataLoader.dataframe_to_project(df, project_name)


def generate_sample_project(**kwargs) -> ProjectData:
    """Convenience function to generate sample project data."""
    generator = SyntheticDataGenerator()
    return generator.generate_sample_project(**kwargs)
