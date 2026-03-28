"""
Unit tests for project duration estimation modules.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import generate_sample_project, ProjectData, Task, DataLoader
from src.models.cpm import CriticalPathMethod, CPMAnalysisResult
from src.models.pert import PERTAnalyzer, PERTAnalysisResult
from src.models.monte_carlo import MonteCarloSimulator, MonteCarloResult
from src.evaluation import ProjectDurationEvaluator, EvaluationMetrics


class TestDataGeneration:
    """Test data generation functionality."""
    
    def test_generate_sample_project(self):
        """Test synthetic project generation."""
        project_data = generate_sample_project(n_tasks=5, seed=42)
        
        assert isinstance(project_data, ProjectData)
        assert len(project_data.tasks) == 5
        assert project_data.project_name == "Synthetic Project"
        
        # Check that all tasks have required attributes
        for task in project_data.tasks:
            assert isinstance(task.task_id, str)
            assert isinstance(task.name, str)
            assert isinstance(task.duration, float)
            assert task.duration > 0
            assert isinstance(task.dependencies, list)
            assert isinstance(task.risk_level, str)
    
    def test_task_creation(self):
        """Test Task dataclass creation."""
        task = Task(
            task_id="T1",
            name="Test Task",
            duration=5.0,
            dependencies=["T0"],
            risk_level="Medium"
        )
        
        assert task.task_id == "T1"
        assert task.name == "Test Task"
        assert task.duration == 5.0
        assert task.dependencies == ["T0"]
        assert task.risk_level == "Medium"
    
    def test_data_loader_conversion(self):
        """Test data conversion between ProjectData and DataFrame."""
        project_data = generate_sample_project(n_tasks=3, seed=42)
        
        # Convert to DataFrame
        df = DataLoader.project_to_dataframe(project_data)
        assert len(df) == 3
        assert "task_id" in df.columns
        assert "duration" in df.columns
        
        # Convert back to ProjectData
        project_data_2 = DataLoader.dataframe_to_project(df, "Test Project")
        assert len(project_data_2.tasks) == 3
        assert project_data_2.project_name == "Test Project"


class TestCPMAnalysis:
    """Test Critical Path Method analysis."""
    
    def test_cpm_initialization(self):
        """Test CPM analyzer initialization."""
        cpm = CriticalPathMethod()
        assert cpm.weight_attribute == "duration"
    
    def test_cpm_analysis(self):
        """Test CPM analysis on sample project."""
        project_data = generate_sample_project(n_tasks=5, seed=42)
        cpm = CriticalPathMethod()
        
        result = cpm.analyze(project_data)
        
        assert isinstance(result, CPMAnalysisResult)
        assert result.duration >= 0
        assert isinstance(result.critical_path, list)
        assert isinstance(result.slack_times, dict)
        assert len(result.slack_times) == len(project_data.tasks)
        
        # Check that critical path tasks have zero slack
        for task_id in result.critical_path:
            assert result.slack_times[task_id] == 0
    
    def test_cpm_with_simple_project(self):
        """Test CPM with a simple linear project."""
        # Create a simple linear project: A -> B -> C
        tasks = [
            Task("A", "Task A", 2.0, [], "Low"),
            Task("B", "Task B", 3.0, ["A"], "Medium"),
            Task("C", "Task C", 1.0, ["B"], "High")
        ]
        
        project_data = ProjectData(tasks=tasks, project_name="Linear Project", metadata={})
        cpm = CriticalPathMethod()
        result = cpm.analyze(project_data)
        
        # Should have duration of 6 (2+3+1)
        assert result.duration == 6.0
        # Critical path should be A -> B -> C
        assert result.critical_path == ["A", "B", "C"]
        # All tasks should be critical
        assert all(result.slack_times[task_id] == 0 for task_id in ["A", "B", "C"])


class TestPERTAnalysis:
    """Test PERT analysis functionality."""
    
    def test_pert_initialization(self):
        """Test PERT analyzer initialization."""
        pert = PERTAnalyzer()
        assert pert.confidence_levels == [0.8, 0.9, 0.95]
    
    def test_pert_analysis(self):
        """Test PERT analysis on sample project."""
        project_data = generate_sample_project(n_tasks=5, seed=42)
        pert = PERTAnalyzer()
        
        result = pert.analyze(project_data)
        
        assert isinstance(result, PERTAnalysisResult)
        assert result.project_expected_duration >= 0
        assert result.project_variance >= 0
        assert result.project_standard_deviation >= 0
        assert isinstance(result.confidence_intervals, dict)
        assert isinstance(result.task_estimates, dict)
        
        # Check confidence intervals
        for level in [0.8, 0.9, 0.95]:
            assert level in result.confidence_intervals
            lower, upper = result.confidence_intervals[level]
            assert lower <= upper
            assert lower >= 0
    
    def test_pert_task_estimates(self):
        """Test PERT task-level estimates."""
        project_data = generate_sample_project(n_tasks=3, seed=42)
        pert = PERTAnalyzer()
        result = pert.analyze(project_data)
        
        for task_id, estimate in result.task_estimates.items():
            assert estimate.optimistic <= estimate.most_likely <= estimate.pessimistic
            assert estimate.expected >= 0
            assert estimate.variance >= 0
            assert estimate.standard_deviation >= 0


class TestMonteCarloSimulation:
    """Test Monte Carlo simulation functionality."""
    
    def test_monte_carlo_initialization(self):
        """Test Monte Carlo simulator initialization."""
        mc = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        assert mc.n_simulations == 1000
        assert mc.random_seed == 42
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        project_data = generate_sample_project(n_tasks=5, seed=42)
        mc = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        
        result = mc.simulate(project_data)
        
        assert isinstance(result, MonteCarloResult)
        assert len(result.simulated_durations) == 1000
        assert result.mean_duration >= 0
        assert result.median_duration >= 0
        assert result.std_duration >= 0
        assert isinstance(result.percentiles, dict)
        assert isinstance(result.confidence_intervals, dict)
        
        # Check percentiles
        for p in [5, 25, 50, 75, 95]:
            assert p in result.percentiles
            assert result.percentiles[p] >= 0
    
    def test_monte_carlo_reproducibility(self):
        """Test Monte Carlo simulation reproducibility."""
        project_data = generate_sample_project(n_tasks=3, seed=42)
        
        # Run simulation twice with same seed
        mc1 = MonteCarloSimulator(n_simulations=100, random_seed=42)
        mc2 = MonteCarloSimulator(n_simulations=100, random_seed=42)
        
        result1 = mc1.simulate(project_data)
        result2 = mc2.simulate(project_data)
        
        # Results should be identical
        assert result1.mean_duration == result2.mean_duration
        assert result1.std_duration == result2.std_duration


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ProjectDurationEvaluator()
        assert evaluator is not None
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        evaluator = ProjectDurationEvaluator()
        
        # Mock model results
        model_results = {
            "duration": 10.0,
            "critical_path": ["A", "B", "C"],
            "critical_path_length": 3,
            "slack_times": {"A": 0, "B": 0, "C": 0, "D": 2},
            "task_details": {
                "A": {"duration": 3, "is_critical": True},
                "B": {"duration": 4, "is_critical": True},
                "C": {"duration": 3, "is_critical": True},
                "D": {"duration": 2, "is_critical": False}
            }
        }
        
        metrics = evaluator.evaluate_model(model_results)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.project_duration == 10.0
        assert metrics.critical_path_length == 3
        assert metrics.total_slack_time == 2.0
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        evaluator = ProjectDurationEvaluator()
        
        # Create mock model comparisons
        model_comparisons = [
            ModelComparison(
                model_name="Model1",
                metrics=EvaluationMetrics(
                    duration_mae=1.0, duration_rmse=1.2, duration_mape=10.0,
                    duration_smape=9.5, duration_r2=0.8, critical_path_accuracy=0.9,
                    critical_path_precision=0.85, critical_path_recall=0.9,
                    critical_path_f1=0.87, risk_calibration=0.8,
                    confidence_interval_coverage=0.85, project_duration=10.0,
                    critical_path_length=3, total_slack_time=2.0, risk_score=0.3
                ),
                predictions={},
                execution_time=0.1
            ),
            ModelComparison(
                model_name="Model2",
                metrics=EvaluationMetrics(
                    duration_mae=0.8, duration_rmse=1.0, duration_mape=8.0,
                    duration_smape=7.5, duration_r2=0.85, critical_path_accuracy=0.95,
                    critical_path_precision=0.9, critical_path_recall=0.95,
                    critical_path_f1=0.92, risk_calibration=0.85,
                    confidence_interval_coverage=0.9, project_duration=9.5,
                    critical_path_length=3, total_slack_time=1.5, risk_score=0.25
                ),
                predictions={},
                execution_time=0.15
            )
        ]
        
        leaderboard = evaluator.compare_models(model_comparisons)
        
        assert len(leaderboard) == 2
        assert "Model" in leaderboard.columns
        assert "Overall_Score" in leaderboard.columns
        # Model2 should rank higher (lower overall score)
        assert leaderboard.iloc[0]["Model"] == "Model2"


class TestIntegration:
    """Test integration between different modules."""
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis."""
        # Generate project data
        project_data = generate_sample_project(n_tasks=5, seed=42)
        
        # Run CPM analysis
        cpm = CriticalPathMethod()
        cpm_result = cpm.analyze(project_data)
        
        # Run PERT analysis
        pert = PERTAnalyzer()
        pert_result = pert.analyze(project_data)
        
        # Run Monte Carlo simulation
        mc = MonteCarloSimulator(n_simulations=100, random_seed=42)
        mc_result = mc.simulate(project_data)
        
        # Evaluate models
        evaluator = ProjectDurationEvaluator()
        
        model_comparisons = [
            ModelComparison(
                model_name="CPM",
                metrics=evaluator.evaluate_model({
                    "duration": cpm_result.duration,
                    "critical_path": cpm_result.critical_path,
                    "critical_path_length": len(cpm_result.critical_path),
                    "slack_times": cpm_result.slack_times
                }),
                predictions={"duration": cpm_result.duration},
                execution_time=0.01
            ),
            ModelComparison(
                model_name="PERT",
                metrics=evaluator.evaluate_model({
                    "duration": pert_result.project_expected_duration,
                    "critical_path": cpm_result.critical_path,
                    "critical_path_length": len(cpm_result.critical_path),
                    "confidence_intervals": pert_result.confidence_intervals
                }),
                predictions={"expected_duration": pert_result.project_expected_duration},
                execution_time=0.02
            ),
            ModelComparison(
                model_name="Monte Carlo",
                metrics=evaluator.evaluate_model({
                    "duration": mc_result.mean_duration,
                    "critical_path": cpm_result.critical_path,
                    "critical_path_length": len(cpm_result.critical_path),
                    "confidence_intervals": mc_result.confidence_intervals
                }),
                predictions={"mean_duration": mc_result.mean_duration},
                execution_time=0.1
            )
        ]
        
        # Create leaderboard
        leaderboard = evaluator.compare_models(model_comparisons)
        
        assert len(leaderboard) == 3
        assert all(model in leaderboard["Model"].values for model in ["CPM", "PERT", "Monte Carlo"])


if __name__ == "__main__":
    pytest.main([__file__])
