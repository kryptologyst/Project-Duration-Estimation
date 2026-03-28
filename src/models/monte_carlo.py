"""
Monte Carlo simulation for project duration estimation with uncertainty analysis.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from scipy import stats
from src.data import ProjectData, Task
from src.models.cpm import CriticalPathMethod, CPMAnalysisResult
from src.models.pert import PERTAnalyzer


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    simulated_durations: List[float]
    mean_duration: float
    median_duration: float
    std_duration: float
    percentiles: Dict[float, float]
    confidence_intervals: Dict[float, Tuple[float, float]]
    probability_distribution: Dict[str, Any]


class MonteCarloSimulator:
    """Monte Carlo simulation for project duration estimation."""
    
    def __init__(self, 
                 n_simulations: int = 10000,
                 confidence_levels: List[float] = [0.8, 0.9, 0.95],
                 random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulation runs
            confidence_levels: List of confidence levels for interval estimation
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.confidence_levels = confidence_levels
        self.random_seed = random_seed
        self.cpm = CriticalPathMethod()
        self.pert = PERTAnalyzer()
    
    def simulate(self, project_data: ProjectData) -> MonteCarloResult:
        """
        Perform Monte Carlo simulation on project data.
        
        Args:
            project_data: Project data with PERT estimates
            
        Returns:
            MonteCarloResult with simulation statistics
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Get PERT estimates for each task
        pert_result = self.pert.analyze(project_data)
        
        # Perform simulations
        simulated_durations = self._run_simulations(project_data, pert_result)
        
        # Calculate statistics
        mean_duration = np.mean(simulated_durations)
        median_duration = np.median(simulated_durations)
        std_duration = np.std(simulated_durations)
        
        # Calculate percentiles
        percentiles = self._calculate_percentiles(simulated_durations)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(simulated_durations)
        
        # Analyze probability distribution
        probability_distribution = self._analyze_distribution(simulated_durations)
        
        return MonteCarloResult(
            simulated_durations=simulated_durations,
            mean_duration=mean_duration,
            median_duration=median_duration,
            std_duration=std_duration,
            percentiles=percentiles,
            confidence_intervals=confidence_intervals,
            probability_distribution=probability_distribution
        )
    
    def _run_simulations(self, 
                        project_data: ProjectData, 
                        pert_result) -> List[float]:
        """Run Monte Carlo simulations."""
        simulated_durations = []
        
        for _ in range(self.n_simulations):
            # Create modified project data with simulated durations
            simulated_project = self._create_simulated_project(project_data, pert_result)
            
            # Calculate project duration using CPM
            cpm_result = self.cpm.analyze(simulated_project)
            simulated_durations.append(cpm_result.duration)
        
        return simulated_durations
    
    def _create_simulated_project(self, 
                                project_data: ProjectData, 
                                pert_result) -> ProjectData:
        """Create a project with simulated task durations."""
        simulated_tasks = []
        
        for task in project_data.tasks:
            # Get PERT estimates for this task
            pert_estimate = pert_result.task_estimates[task.task_id]
            
            # Generate random duration using beta distribution
            # Beta distribution is commonly used for PERT simulation
            alpha = 4  # Shape parameter
            beta = 4   # Shape parameter
            
            # Scale beta distribution to PERT range
            beta_sample = np.random.beta(alpha, beta)
            simulated_duration = (pert_estimate.optimistic + 
                                beta_sample * (pert_estimate.pessimistic - pert_estimate.optimistic))
            
            # Create new task with simulated duration
            simulated_task = Task(
                task_id=task.task_id,
                name=task.name,
                duration=simulated_duration,
                dependencies=task.dependencies,
                resource_requirements=task.resource_requirements,
                risk_level=task.risk_level,
                optimistic_duration=task.optimistic_duration,
                most_likely_duration=task.most_likely_duration,
                pessimistic_duration=task.pessimistic_duration
            )
            simulated_tasks.append(simulated_task)
        
        return ProjectData(
            tasks=simulated_tasks,
            project_name=project_data.project_name,
            metadata=project_data.metadata
        )
    
    def _calculate_percentiles(self, durations: List[float]) -> Dict[float, float]:
        """Calculate percentiles for simulated durations."""
        percentiles = {}
        percentile_values = [5, 10, 25, 50, 75, 90, 95, 99]
        
        for p in percentile_values:
            percentiles[p] = np.percentile(durations, p)
        
        return percentiles
    
    def _calculate_confidence_intervals(self, durations: List[float]) -> Dict[float, Tuple[float, float]]:
        """Calculate confidence intervals for simulated durations."""
        confidence_intervals = {}
        
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(durations, lower_percentile)
            upper_bound = np.percentile(durations, upper_percentile)
            
            confidence_intervals[confidence_level] = (lower_bound, upper_bound)
        
        return confidence_intervals
    
    def _analyze_distribution(self, durations: List[float]) -> Dict[str, Any]:
        """Analyze the probability distribution of simulated durations."""
        # Test for normality
        shapiro_stat, shapiro_p = stats.shapiro(durations)
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(durations)
        kurtosis = stats.kurtosis(durations)
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(durations)
        
        return {
            "is_normal": shapiro_p > 0.05,
            "shapiro_statistic": shapiro_stat,
            "shapiro_p_value": shapiro_p,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "fitted_normal_mean": mu,
            "fitted_normal_std": sigma
        }
    
    def get_completion_probability(self, 
                                 project_data: ProjectData, 
                                 target_duration: float) -> float:
        """Get probability of completing project within target duration."""
        result = self.simulate(project_data)
        
        # Count simulations that completed within target duration
        completed_within_target = sum(1 for d in result.simulated_durations if d <= target_duration)
        probability = completed_within_target / len(result.simulated_durations)
        
        return probability
    
    def get_risk_metrics(self, project_data: ProjectData) -> Dict[str, float]:
        """Calculate risk metrics for the project."""
        result = self.simulate(project_data)
        
        # Calculate various risk metrics
        risk_metrics = {
            "coefficient_of_variation": result.std_duration / result.mean_duration,
            "probability_over_mean": self.get_completion_probability(project_data, result.mean_duration),
            "probability_over_median": self.get_completion_probability(project_data, result.median_duration),
            "range_95": result.percentiles[95] - result.percentiles[5],
            "interquartile_range": result.percentiles[75] - result.percentiles[25]
        }
        
        return risk_metrics
