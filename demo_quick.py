#!/usr/bin/env python3
"""
Quick demonstration of the modernized Project Duration Estimation system.

This script demonstrates the key features of the refactored system.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import generate_sample_project, DataLoader
from src.models.cpm import CriticalPathMethod
from src.models.pert import PERTAnalyzer
from src.models.monte_carlo import MonteCarloSimulator
from src.evaluation import ProjectDurationEvaluator, ModelComparison
from src.compliance import ComplianceManager, create_disclaimer_text


def main():
    """Run a quick demonstration of the system."""
    
    print("🚀 Project Duration Estimation - Quick Demo")
    print("=" * 50)
    
    # Generate sample project
    print("\n📊 Generating sample project...")
    project_data = generate_sample_project(
        n_tasks=8,
        min_duration=2.0,
        max_duration=10.0,
        dependency_probability=0.3
    )
    
    print(f"✅ Generated project with {len(project_data.tasks)} tasks")
    print(f"   Total duration: {sum(task.duration for task in project_data.tasks):.1f} days")
    
    # Initialize compliance manager
    compliance = ComplianceManager()
    compliance.log_data_access("synthetic_project_data", "demo_user")
    
    # CPM Analysis
    print("\n🔍 Running CPM Analysis...")
    start_time = time.time()
    cpm = CriticalPathMethod()
    cpm_result = cpm.analyze(project_data)
    cpm_time = time.time() - start_time
    
    print(f"✅ CPM completed in {cpm_time:.3f}s")
    print(f"   Project Duration: {cpm_result.duration:.1f} days")
    print(f"   Critical Path: {' → '.join(cpm_result.critical_path)}")
    
    compliance.log_model_execution("CPM", "input_hash", "output_hash", cpm_time, "demo_user")
    
    # PERT Analysis
    print("\n📊 Running PERT Analysis...")
    start_time = time.time()
    pert = PERTAnalyzer()
    pert_result = pert.analyze(project_data)
    pert_time = time.time() - start_time
    
    print(f"✅ PERT completed in {pert_time:.3f}s")
    print(f"   Expected Duration: {pert_result.project_expected_duration:.1f} days")
    print(f"   Standard Deviation: {pert_result.project_standard_deviation:.1f} days")
    print(f"   90% CI: {pert_result.confidence_intervals[0.9][0]:.1f} - {pert_result.confidence_intervals[0.9][1]:.1f} days")
    
    compliance.log_model_execution("PERT", "input_hash", "output_hash", pert_time, "demo_user")
    
    # Monte Carlo Simulation
    print("\n🎲 Running Monte Carlo Simulation (1,000 runs)...")
    start_time = time.time()
    mc = MonteCarloSimulator(n_simulations=1000, random_seed=42)
    mc_result = mc.simulate(project_data)
    mc_time = time.time() - start_time
    
    print(f"✅ Monte Carlo completed in {mc_time:.3f}s")
    print(f"   Mean Duration: {mc_result.mean_duration:.1f} days")
    print(f"   Median Duration: {mc_result.median_duration:.1f} days")
    print(f"   95th Percentile: {mc_result.percentiles[95]:.1f} days")
    
    compliance.log_model_execution("Monte Carlo", "input_hash", "output_hash", mc_time, "demo_user")
    
    # Model Comparison
    print("\n📈 Creating Model Comparison...")
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
            execution_time=cpm_time
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
            execution_time=pert_time
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
            execution_time=mc_time
        )
    ]
    
    leaderboard = evaluator.compare_models(model_comparisons)
    
    print("🏆 Model Comparison Results:")
    print(leaderboard[['Model', 'Project_Duration', 'Execution_Time', 'Overall_Score']].to_string(index=False))
    
    # Log decision support
    compliance.log_decision_support(
        decision_type="project_duration_estimation",
        recommendations=[
            f"Use CPM for deterministic scheduling: {cpm_result.duration:.1f} days",
            f"Consider PERT for risk assessment: {pert_result.project_expected_duration:.1f} ± {pert_result.project_standard_deviation:.1f} days",
            f"Use Monte Carlo for comprehensive analysis: {mc_result.mean_duration:.1f} days"
        ],
        confidence_scores={"CPM": 0.85, "PERT": 0.80, "Monte Carlo": 0.90},
        user_id="demo_user"
    )
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 30)
    print(f"• Project Duration (CPM): {cpm_result.duration:.1f} days")
    print(f"• Expected Duration (PERT): {pert_result.project_expected_duration:.1f} days")
    print(f"• Mean Duration (Monte Carlo): {mc_result.mean_duration:.1f} days")
    print(f"• Critical Path: {' → '.join(cpm_result.critical_path)}")
    print(f"• Total Slack Time: {sum(cpm_result.slack_times.values()):.1f} days")
    print(f"• Risk Level: {'High' if mc_result.std_duration / mc_result.mean_duration > 0.3 else 'Medium' if mc_result.std_duration / mc_result.mean_duration > 0.15 else 'Low'}")
    
    print("\n💡 KEY INSIGHTS:")
    print("• CPM provides deterministic project scheduling")
    print("• PERT incorporates uncertainty and risk assessment")
    print("• Monte Carlo offers comprehensive probabilistic analysis")
    print("• All methods complement each other for robust planning")
    
    print("\n⚠️  DISCLAIMER:")
    print("This is an experimental research tool. Always validate estimates")
    print("with experienced project managers before making decisions.")
    
    print("\n✅ Demo completed successfully!")
    print("   All models executed without errors")
    print("   Compliance logging completed")
    print("   Results ready for human review")


if __name__ == "__main__":
    main()
