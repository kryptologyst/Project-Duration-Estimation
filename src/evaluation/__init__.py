"""
Evaluation metrics and leaderboard for project duration estimation models.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data import ProjectData
from src.models.cpm import CPMAnalysisResult
from src.models.pert import PERTAnalysisResult
from src.models.monte_carlo import MonteCarloResult


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Duration accuracy metrics
    duration_mae: float
    duration_rmse: float
    duration_mape: float
    duration_smape: float
    duration_r2: float
    
    # Critical path metrics
    critical_path_accuracy: float
    critical_path_precision: float
    critical_path_recall: float
    critical_path_f1: float
    
    # Risk assessment metrics
    risk_calibration: float
    confidence_interval_coverage: float
    
    # Business KPIs
    project_duration: float
    critical_path_length: float
    total_slack_time: float
    risk_score: float


@dataclass
class ModelComparison:
    """Comparison results between different models."""
    model_name: str
    metrics: EvaluationMetrics
    predictions: Dict[str, Any]
    execution_time: float


class ProjectDurationEvaluator:
    """Evaluator for project duration estimation models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def evaluate_model(self, 
                      model_results: Dict[str, Any],
                      actual_durations: Optional[Dict[str, float]] = None,
                      model_name: str = "Model") -> EvaluationMetrics:
        """
        Evaluate a single model's performance.
        
        Args:
            model_results: Results from model analysis
            actual_durations: Actual task durations (if available)
            model_name: Name of the model being evaluated
            
        Returns:
            EvaluationMetrics object with performance metrics
        """
        # Extract basic metrics
        project_duration = model_results.get("duration", 0.0)
        critical_path_length = model_results.get("critical_path_length", 0.0)
        
        # Calculate duration accuracy metrics
        duration_metrics = self._calculate_duration_metrics(
            model_results, actual_durations
        )
        
        # Calculate critical path metrics
        critical_path_metrics = self._calculate_critical_path_metrics(
            model_results, actual_durations
        )
        
        # Calculate risk assessment metrics
        risk_metrics = self._calculate_risk_metrics(model_results)
        
        # Calculate business KPIs
        business_kpis = self._calculate_business_kpis(model_results)
        
        return EvaluationMetrics(
            duration_mae=duration_metrics["mae"],
            duration_rmse=duration_metrics["rmse"],
            duration_mape=duration_metrics["mape"],
            duration_smape=duration_metrics["smape"],
            duration_r2=duration_metrics["r2"],
            critical_path_accuracy=critical_path_metrics["accuracy"],
            critical_path_precision=critical_path_metrics["precision"],
            critical_path_recall=critical_path_metrics["recall"],
            critical_path_f1=critical_path_metrics["f1"],
            risk_calibration=risk_metrics["calibration"],
            confidence_interval_coverage=risk_metrics["coverage"],
            project_duration=business_kpis["project_duration"],
            critical_path_length=business_kpis["critical_path_length"],
            total_slack_time=business_kpis["total_slack_time"],
            risk_score=business_kpis["risk_score"]
        )
    
    def compare_models(self, 
                      model_results: List[ModelComparison]) -> pd.DataFrame:
        """
        Compare multiple models and create a leaderboard.
        
        Args:
            model_results: List of ModelComparison objects
            
        Returns:
            DataFrame with model comparison results
        """
        comparison_data = []
        
        for model in model_results:
            row = {
                "Model": model.model_name,
                "Execution_Time": model.execution_time,
                "Project_Duration": model.metrics.project_duration,
                "Critical_Path_Length": model.metrics.critical_path_length,
                "Duration_MAE": model.metrics.duration_mae,
                "Duration_RMSE": model.metrics.duration_rmse,
                "Duration_MAPE": model.metrics.duration_mape,
                "Critical_Path_Accuracy": model.metrics.critical_path_accuracy,
                "Critical_Path_F1": model.metrics.critical_path_f1,
                "Risk_Calibration": model.metrics.risk_calibration,
                "Confidence_Coverage": model.metrics.confidence_interval_coverage,
                "Total_Slack_Time": model.metrics.total_slack_time,
                "Risk_Score": model.metrics.risk_score
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by overall performance (lower MAE and RMSE are better)
        df["Overall_Score"] = (
            df["Duration_MAE"] * 0.4 + 
            df["Duration_RMSE"] * 0.3 + 
            df["Critical_Path_Accuracy"] * 0.2 + 
            df["Risk_Calibration"] * 0.1
        )
        df = df.sort_values("Overall_Score")
        
        return df
    
    def _calculate_duration_metrics(self, 
                                  model_results: Dict[str, Any],
                                  actual_durations: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Calculate duration prediction accuracy metrics."""
        if actual_durations is None:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "mape": 0.0,
                "smape": 0.0,
                "r2": 0.0
            }
        
        # Extract predicted and actual durations
        predicted_durations = model_results.get("task_durations", {})
        
        if not predicted_durations:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "mape": 0.0,
                "smape": 0.0,
                "r2": 0.0
            }
        
        # Align predictions with actual durations
        common_tasks = set(predicted_durations.keys()) & set(actual_durations.keys())
        
        if not common_tasks:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "mape": 0.0,
                "smape": 0.0,
                "r2": 0.0
            }
        
        y_pred = [predicted_durations[task] for task in common_tasks]
        y_true = [actual_durations[task] for task in common_tasks]
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE calculation (handle division by zero)
        mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
        
        # SMAPE calculation
        smape = np.mean(2 * np.abs(np.array(y_true) - np.array(y_pred)) / 
                       (np.abs(y_true) + np.abs(y_pred))) * 100
        
        r2 = r2_score(y_true, y_pred)
        
        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "smape": smape,
            "r2": r2
        }
    
    def _calculate_critical_path_metrics(self, 
                                       model_results: Dict[str, Any],
                                       actual_durations: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Calculate critical path identification metrics."""
        predicted_critical_path = set(model_results.get("critical_path", []))
        
        if actual_durations is None:
            # Use heuristics for critical path (longest tasks)
            all_tasks = model_results.get("all_tasks", [])
            task_durations = model_results.get("task_durations", {})
            
            if not all_tasks or not task_durations:
                return {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                }
            
            # Assume top 30% longest tasks are critical
            sorted_tasks = sorted(all_tasks, key=lambda t: task_durations.get(t, 0), reverse=True)
            n_critical = max(1, len(sorted_tasks) // 3)
            actual_critical_path = set(sorted_tasks[:n_critical])
        else:
            # Use actual critical path if available
            actual_critical_path = set(model_results.get("actual_critical_path", []))
        
        if not actual_critical_path:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
        
        # Calculate metrics
        true_positives = len(predicted_critical_path & actual_critical_path)
        false_positives = len(predicted_critical_path - actual_critical_path)
        false_negatives = len(actual_critical_path - predicted_critical_path)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy is the proportion of correctly identified critical tasks
        all_tasks = predicted_critical_path | actual_critical_path
        accuracy = true_positives / len(all_tasks) if all_tasks else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _calculate_risk_metrics(self, model_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk assessment metrics."""
        # Risk calibration (how well confidence intervals match actual uncertainty)
        confidence_intervals = model_results.get("confidence_intervals", {})
        actual_variance = model_results.get("actual_variance", 0.0)
        
        calibration = 0.0
        if confidence_intervals and actual_variance > 0:
            # Simple calibration metric based on interval width vs actual variance
            interval_widths = []
            for level, (lower, upper) in confidence_intervals.items():
                interval_widths.append(upper - lower)
            
            avg_interval_width = np.mean(interval_widths)
            expected_width = 2 * np.sqrt(actual_variance)  # Approximate for 95% interval
            calibration = 1.0 - abs(avg_interval_width - expected_width) / expected_width
            calibration = max(0.0, min(1.0, calibration))
        
        # Confidence interval coverage (if we have actual data)
        coverage = 0.0
        if model_results.get("actual_duration") and confidence_intervals:
            actual_duration = model_results["actual_duration"]
            covered_intervals = 0
            total_intervals = len(confidence_intervals)
            
            for level, (lower, upper) in confidence_intervals.items():
                if lower <= actual_duration <= upper:
                    covered_intervals += 1
            
            coverage = covered_intervals / total_intervals if total_intervals > 0 else 0.0
        
        return {
            "calibration": calibration,
            "coverage": coverage
        }
    
    def _calculate_business_kpis(self, model_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate business key performance indicators."""
        project_duration = model_results.get("duration", 0.0)
        critical_path_length = model_results.get("critical_path_length", 0.0)
        
        # Calculate total slack time
        slack_times = model_results.get("slack_times", {})
        total_slack_time = sum(slack_times.values()) if slack_times else 0.0
        
        # Calculate risk score based on uncertainty and critical path length
        confidence_intervals = model_results.get("confidence_intervals", {})
        risk_score = 0.0
        
        if confidence_intervals:
            # Risk score based on interval width relative to mean duration
            interval_widths = []
            for level, (lower, upper) in confidence_intervals.items():
                interval_widths.append(upper - lower)
            
            avg_interval_width = np.mean(interval_widths)
            risk_score = min(1.0, avg_interval_width / project_duration) if project_duration > 0 else 0.0
        
        return {
            "project_duration": project_duration,
            "critical_path_length": critical_path_length,
            "total_slack_time": total_slack_time,
            "risk_score": risk_score
        }
    
    def create_leaderboard(self, 
                          model_comparisons: List[ModelComparison],
                          save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create a comprehensive leaderboard of model performance.
        
        Args:
            model_comparisons: List of model comparison results
            save_path: Optional path to save the leaderboard
            
        Returns:
            DataFrame with leaderboard results
        """
        leaderboard = self.compare_models(model_comparisons)
        
        if save_path:
            leaderboard.to_csv(save_path, index=False)
        
        return leaderboard
