"""
PERT (Program Evaluation and Review Technique) analysis for project duration estimation.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from scipy import stats
from src.data import ProjectData, Task
from src.models.cpm import CriticalPathMethod, CPMAnalysisResult


@dataclass
class PERTEstimate:
    """PERT estimate for a single task."""
    task_id: str
    optimistic: float
    most_likely: float
    pessimistic: float
    expected: float
    variance: float
    standard_deviation: float


@dataclass
class PERTAnalysisResult:
    """Results from PERT analysis."""
    task_estimates: Dict[str, PERTEstimate]
    project_expected_duration: float
    project_variance: float
    project_standard_deviation: float
    confidence_intervals: Dict[float, Tuple[float, float]]
    critical_path_probability: float


class PERTAnalyzer:
    """PERT analysis for project duration estimation with uncertainty."""
    
    def __init__(self, confidence_levels: List[float] = [0.8, 0.9, 0.95]):
        """
        Initialize PERT analyzer.
        
        Args:
            confidence_levels: List of confidence levels for interval estimation
        """
        self.confidence_levels = confidence_levels
        self.cpm = CriticalPathMethod()
    
    def analyze(self, project_data: ProjectData) -> PERTAnalysisResult:
        """
        Perform PERT analysis on project data.
        
        Args:
            project_data: Project data with PERT estimates
            
        Returns:
            PERTAnalysisResult with probabilistic duration estimates
        """
        # Calculate PERT estimates for each task
        task_estimates = self._calculate_task_estimates(project_data)
        
        # Perform CPM analysis to find critical path
        cpm_result = self.cpm.analyze(project_data)
        
        # Calculate project-level statistics
        project_stats = self._calculate_project_statistics(
            task_estimates, cpm_result.critical_path
        )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            project_stats["expected"], project_stats["variance"]
        )
        
        # Calculate probability of completing on critical path duration
        critical_path_probability = self._calculate_critical_path_probability(
            project_stats["expected"], project_stats["variance"], cpm_result.duration
        )
        
        return PERTAnalysisResult(
            task_estimates=task_estimates,
            project_expected_duration=project_stats["expected"],
            project_variance=project_stats["variance"],
            project_standard_deviation=project_stats["standard_deviation"],
            confidence_intervals=confidence_intervals,
            critical_path_probability=critical_path_probability
        )
    
    def _calculate_task_estimates(self, project_data: ProjectData) -> Dict[str, PERTEstimate]:
        """Calculate PERT estimates for each task."""
        task_estimates = {}
        
        for task in project_data.tasks:
            # Use provided PERT estimates or generate from duration
            if (task.optimistic_duration is not None and 
                task.most_likely_duration is not None and 
                task.pessimistic_duration is not None):
                optimistic = task.optimistic_duration
                most_likely = task.most_likely_duration
                pessimistic = task.pessimistic_duration
            else:
                # Generate PERT estimates from single duration estimate
                optimistic = task.duration * 0.7
                most_likely = task.duration
                pessimistic = task.duration * 1.5
            
            # PERT formula: Expected = (Optimistic + 4*Most_Likely + Pessimistic) / 6
            expected = (optimistic + 4 * most_likely + pessimistic) / 6
            
            # Variance = ((Pessimistic - Optimistic) / 6)^2
            variance = ((pessimistic - optimistic) / 6) ** 2
            standard_deviation = np.sqrt(variance)
            
            task_estimates[task.task_id] = PERTEstimate(
                task_id=task.task_id,
                optimistic=optimistic,
                most_likely=most_likely,
                pessimistic=pessimistic,
                expected=expected,
                variance=variance,
                standard_deviation=standard_deviation
            )
        
        return task_estimates
    
    def _calculate_project_statistics(self, 
                                    task_estimates: Dict[str, PERTEstimate],
                                    critical_path: List[str]) -> Dict[str, float]:
        """Calculate project-level statistics based on critical path."""
        if not critical_path:
            return {
                "expected": 0.0,
                "variance": 0.0,
                "standard_deviation": 0.0
            }
        
        # Sum expected durations and variances along critical path
        expected_duration = sum(
            task_estimates[task_id].expected for task_id in critical_path
        )
        variance = sum(
            task_estimates[task_id].variance for task_id in critical_path
        )
        standard_deviation = np.sqrt(variance)
        
        return {
            "expected": expected_duration,
            "variance": variance,
            "standard_deviation": standard_deviation
        }
    
    def _calculate_confidence_intervals(self, 
                                       expected: float, 
                                       variance: float) -> Dict[float, Tuple[float, float]]:
        """Calculate confidence intervals for project duration."""
        standard_deviation = np.sqrt(variance)
        confidence_intervals = {}
        
        for confidence_level in self.confidence_levels:
            # Calculate z-score for confidence level
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            # Calculate interval bounds
            margin_of_error = z_score * standard_deviation
            lower_bound = max(0, expected - margin_of_error)
            upper_bound = expected + margin_of_error
            
            confidence_intervals[confidence_level] = (lower_bound, upper_bound)
        
        return confidence_intervals
    
    def _calculate_critical_path_probability(self, 
                                           expected: float, 
                                           variance: float, 
                                           critical_path_duration: float) -> float:
        """Calculate probability of completing project within critical path duration."""
        if variance == 0:
            return 1.0 if expected <= critical_path_duration else 0.0
        
        standard_deviation = np.sqrt(variance)
        z_score = (critical_path_duration - expected) / standard_deviation
        probability = stats.norm.cdf(z_score)
        
        return probability
    
    def get_task_expected_duration(self, project_data: ProjectData, task_id: str) -> float:
        """Get expected duration for a specific task."""
        result = self.analyze(project_data)
        return result.task_estimates[task_id].expected
    
    def get_project_confidence_interval(self, 
                                      project_data: ProjectData, 
                                      confidence_level: float = 0.9) -> Tuple[float, float]:
        """Get confidence interval for project duration."""
        result = self.analyze(project_data)
        return result.confidence_intervals[confidence_level]
    
    def get_completion_probability(self, 
                                 project_data: ProjectData, 
                                 target_duration: float) -> float:
        """Get probability of completing project within target duration."""
        result = self.analyze(project_data)
        expected = result.project_expected_duration
        variance = result.project_variance
        
        if variance == 0:
            return 1.0 if expected <= target_duration else 0.0
        
        standard_deviation = np.sqrt(variance)
        z_score = (target_duration - expected) / standard_deviation
        probability = stats.norm.cdf(z_score)
        
        return probability