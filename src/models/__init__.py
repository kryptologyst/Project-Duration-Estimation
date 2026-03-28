"""
Critical Path Method (CPM) implementation for project duration estimation.
"""

from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
import numpy as np
from dataclasses import dataclass
from src.data import ProjectData, Task


@dataclass
class CPMAnalysisResult:
    """Results from CPM analysis."""
    critical_path: List[str]
    duration: float
    slack_times: Dict[str, float]
    early_start: Dict[str, float]
    early_finish: Dict[str, float]
    late_start: Dict[str, float]
    late_finish: Dict[str, float]
    task_details: Dict[str, Dict[str, Any]]


class CriticalPathMethod:
    """Implementation of Critical Path Method for project scheduling."""
    
    def __init__(self, weight_attribute: str = "duration"):
        """
        Initialize CPM analyzer.
        
        Args:
            weight_attribute: Attribute to use as task weight/duration
        """
        self.weight_attribute = weight_attribute
    
    def analyze(self, project_data: ProjectData) -> CPMAnalysisResult:
        """
        Perform CPM analysis on project data.
        
        Args:
            project_data: Project data containing tasks and dependencies
            
        Returns:
            CPMAnalysisResult with critical path and timing information
        """
        # Create directed graph
        G = self._create_dependency_graph(project_data)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Project contains circular dependencies")
        
        # Calculate forward pass (early start/finish)
        early_start, early_finish = self._forward_pass(G)
        
        # Calculate backward pass (late start/finish)
        late_start, late_finish = self._backward_pass(G, early_finish)
        
        # Calculate slack times
        slack_times = self._calculate_slack_times(early_finish, late_finish)
        
        # Find critical path
        critical_path = self._find_critical_path(G, slack_times)
        
        # Calculate total project duration
        duration = max(early_finish.values()) if early_finish else 0.0
        
        # Create detailed task information
        task_details = self._create_task_details(
            project_data, early_start, early_finish, 
            late_start, late_finish, slack_times
        )
        
        return CPMAnalysisResult(
            critical_path=critical_path,
            duration=duration,
            slack_times=slack_times,
            early_start=early_start,
            early_finish=early_finish,
            late_start=late_start,
            late_finish=late_finish,
            task_details=task_details
        )
    
    def _create_dependency_graph(self, project_data: ProjectData) -> nx.DiGraph:
        """Create NetworkX directed graph from project data."""
        G = nx.DiGraph()
        
        # Add tasks as nodes
        for task in project_data.tasks:
            G.add_node(task.task_id, duration=getattr(task, self.weight_attribute))
        
        # Add dependencies as edges
        for task in project_data.tasks:
            for dep in task.dependencies:
                G.add_edge(dep, task.task_id)
        
        return G
    
    def _forward_pass(self, G: nx.DiGraph) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate early start and early finish times."""
        early_start = {}
        early_finish = {}
        
        # Topological sort ensures we process tasks in dependency order
        for task_id in nx.topological_sort(G):
            # Early start is max of predecessors' early finish times
            if G.in_degree(task_id) == 0:
                early_start[task_id] = 0.0
            else:
                max_predecessor_finish = max(
                    early_finish[pred] for pred in G.predecessors(task_id)
                )
                early_start[task_id] = max_predecessor_finish
            
            # Early finish = early start + duration
            duration = G.nodes[task_id]["duration"]
            early_finish[task_id] = early_start[task_id] + duration
        
        return early_start, early_finish
    
    def _backward_pass(self, G: nx.DiGraph, early_finish: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate late start and late finish times."""
        late_start = {}
        late_finish = {}
        
        # Project completion time
        project_duration = max(early_finish.values())
        
        # Process tasks in reverse topological order
        for task_id in reversed(list(nx.topological_sort(G))):
            # Late finish is min of successors' late start times
            if G.out_degree(task_id) == 0:
                late_finish[task_id] = project_duration
            else:
                min_successor_start = min(
                    late_start[succ] for succ in G.successors(task_id)
                )
                late_finish[task_id] = min_successor_start
            
            # Late start = late finish - duration
            duration = G.nodes[task_id]["duration"]
            late_start[task_id] = late_finish[task_id] - duration
        
        return late_start, late_finish
    
    def _calculate_slack_times(self, early_finish: Dict[str, float], late_finish: Dict[str, float]) -> Dict[str, float]:
        """Calculate slack (float) time for each task."""
        return {
            task_id: late_finish[task_id] - early_finish[task_id]
            for task_id in early_finish.keys()
        }
    
    def _find_critical_path(self, G: nx.DiGraph, slack_times: Dict[str, float]) -> List[str]:
        """Find the critical path (tasks with zero slack)."""
        critical_tasks = [task_id for task_id, slack in slack_times.items() if slack == 0]
        
        # Find the longest path through critical tasks
        critical_subgraph = G.subgraph(critical_tasks)
        if len(critical_subgraph.nodes) == 0:
            return []
        
        # Find longest path in critical subgraph
        try:
            longest_path = nx.dag_longest_path(critical_subgraph, weight="duration")
            return longest_path
        except nx.NetworkXError:
            # Fallback: return critical tasks in topological order
            return list(nx.topological_sort(critical_subgraph))
    
    def _create_task_details(self, 
                           project_data: ProjectData,
                           early_start: Dict[str, float],
                           early_finish: Dict[str, float],
                           late_start: Dict[str, float],
                           late_finish: Dict[str, float],
                           slack_times: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Create detailed information for each task."""
        task_details = {}
        
        for task in project_data.tasks:
            task_details[task.task_id] = {
                "name": task.name,
                "duration": task.duration,
                "dependencies": task.dependencies,
                "risk_level": task.risk_level,
                "early_start": early_start.get(task.task_id, 0),
                "early_finish": early_finish.get(task.task_id, 0),
                "late_start": late_start.get(task.task_id, 0),
                "late_finish": late_finish.get(task.task_id, 0),
                "slack_time": slack_times.get(task.task_id, 0),
                "is_critical": slack_times.get(task.task_id, 0) == 0
            }
        
        return task_details
    
    def get_critical_path_length(self, project_data: ProjectData) -> float:
        """Get the length of the critical path."""
        result = self.analyze(project_data)
        return result.duration
    
    def get_task_slack(self, project_data: ProjectData, task_id: str) -> float:
        """Get slack time for a specific task."""
        result = self.analyze(project_data)
        return result.slack_times.get(task_id, 0)
    
    def is_task_critical(self, project_data: ProjectData, task_id: str) -> bool:
        """Check if a task is on the critical path."""
        result = self.analyze(project_data)
        return result.slack_times.get(task_id, 0) == 0
