"""
Main script for running project duration estimation analysis.
"""

import argparse
import time
from pathlib import Path
import sys
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import generate_sample_project, DataLoader
from src.models.cpm import CriticalPathMethod
from src.models.pert import PERTAnalyzer
from src.models.monte_carlo import MonteCarloSimulator
from src.visualization import ProjectVisualizer
from src.evaluation import ProjectDurationEvaluator, ModelComparison


def main():
    """Main function for running project duration estimation."""
    parser = argparse.ArgumentParser(description="Project Duration Estimation Analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--n-tasks", type=int, default=10, 
                       help="Number of tasks in synthetic project")
    parser.add_argument("--min-duration", type=float, default=1.0, 
                       help="Minimum task duration")
    parser.add_argument("--max-duration", type=float, default=10.0, 
                       help="Maximum task duration")
    parser.add_argument("--dependency-prob", type=float, default=0.3, 
                       help="Dependency probability")
    parser.add_argument("--n-simulations", type=int, default=10000, 
                       help="Number of Monte Carlo simulations")
    parser.add_argument("--output-dir", type=str, default="assets", 
                       help="Output directory for results")
    parser.add_argument("--enable-pert", action="store_true", 
                       help="Enable PERT analysis")
    parser.add_argument("--enable-monte-carlo", action="store_true", 
                       help="Enable Monte Carlo simulation")
    parser.add_argument("--save-plots", action="store_true", 
                       help="Save visualization plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Project Duration Estimation Analysis")
    print("=" * 50)
    
    # Generate project data
    print("📊 Generating synthetic project data...")
    project_data = generate_sample_project(
        n_tasks=args.n_tasks,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        dependency_probability=args.dependency_prob
    )
    
    print(f"✅ Generated project with {len(project_data.tasks)} tasks")
    print(f"   Total duration: {sum(task.duration for task in project_data.tasks):.1f} days")
    
    # Save project data
    DataLoader.save_project(project_data, output_dir / "project_data.csv")
    print(f"💾 Saved project data to {output_dir / 'project_data.csv'}")
    
    # Initialize models
    cpm = CriticalPathMethod()
    pert = PERTAnalyzer() if args.enable_pert else None
    mc = MonteCarloSimulator(n_simulations=args.n_simulations) if args.enable_monte_carlo else None
    visualizer = ProjectVisualizer()
    evaluator = ProjectDurationEvaluator()
    
    # Run analyses
    model_comparisons = []
    
    # CPM Analysis
    print("\n🔍 Running CPM Analysis...")
    start_time = time.time()
    cpm_result = cpm.analyze(project_data)
    cpm_time = time.time() - start_time
    
    print(f"✅ CPM Analysis completed in {cpm_time:.3f} seconds")
    print(f"   Project Duration: {cpm_result.duration:.1f} days")
    print(f"   Critical Path: {' → '.join(cpm_result.critical_path)}")
    
    # Add to model comparisons
    model_comparisons.append(ModelComparison(
        model_name="CPM",
        metrics=evaluator.evaluate_model({
            "duration": cpm_result.duration,
            "critical_path": cpm_result.critical_path,
            "critical_path_length": len(cpm_result.critical_path),
            "slack_times": cpm_result.slack_times,
            "task_details": cpm_result.task_details
        }),
        predictions={
            "duration": cpm_result.duration,
            "critical_path": cpm_result.critical_path,
            "slack_times": cpm_result.slack_times
        },
        execution_time=cpm_time
    ))
    
    # PERT Analysis
    if pert:
        print("\n📊 Running PERT Analysis...")
        start_time = time.time()
        pert_result = pert.analyze(project_data)
        pert_time = time.time() - start_time
        
        print(f"✅ PERT Analysis completed in {pert_time:.3f} seconds")
        print(f"   Expected Duration: {pert_result.project_expected_duration:.1f} days")
        print(f"   Standard Deviation: {pert_result.project_standard_deviation:.1f} days")
        print(f"   Critical Path Probability: {pert_result.critical_path_probability:.1%}")
        
        # Add to model comparisons
        model_comparisons.append(ModelComparison(
            model_name="PERT",
            metrics=evaluator.evaluate_model({
                "duration": pert_result.project_expected_duration,
                "critical_path": cpm_result.critical_path,
                "critical_path_length": len(cpm_result.critical_path),
                "confidence_intervals": pert_result.confidence_intervals,
                "variance": pert_result.project_variance
            }),
            predictions={
                "expected_duration": pert_result.project_expected_duration,
                "confidence_intervals": pert_result.confidence_intervals,
                "critical_path_probability": pert_result.critical_path_probability
            },
            execution_time=pert_time
        ))
    else:
        pert_result = None
    
    # Monte Carlo Analysis
    if mc:
        print(f"\n🎲 Running Monte Carlo Simulation ({args.n_simulations:,} runs)...")
        start_time = time.time()
        mc_result = mc.simulate(project_data)
        mc_time = time.time() - start_time
        
        print(f"✅ Monte Carlo Simulation completed in {mc_time:.3f} seconds")
        print(f"   Mean Duration: {mc_result.mean_duration:.1f} days")
        print(f"   Median Duration: {mc_result.median_duration:.1f} days")
        print(f"   Standard Deviation: {mc_result.std_duration:.1f} days")
        
        # Add to model comparisons
        model_comparisons.append(ModelComparison(
            model_name="Monte Carlo",
            metrics=evaluator.evaluate_model({
                "duration": mc_result.mean_duration,
                "critical_path": cpm_result.critical_path,
                "critical_path_length": len(cpm_result.critical_path),
                "confidence_intervals": mc_result.confidence_intervals,
                "variance": mc_result.std_duration ** 2
            }),
            predictions={
                "mean_duration": mc_result.mean_duration,
                "confidence_intervals": mc_result.confidence_intervals,
                "percentiles": mc_result.percentiles
            },
            execution_time=mc_time
        ))
    else:
        mc_result = None
    
    # Create model comparison leaderboard
    print("\n📈 Creating Model Comparison Leaderboard...")
    leaderboard = evaluator.create_leaderboard(model_comparisons)
    leaderboard_path = output_dir / "model_comparison.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"💾 Saved leaderboard to {leaderboard_path}")
    
    # Display leaderboard
    print("\n🏆 Model Comparison Results:")
    print(leaderboard.to_string(index=False))
    
    # Generate visualizations
    if args.save_plots:
        print("\n🎨 Generating visualizations...")
        plots_dir = output_dir / "plots"
        saved_plots = visualizer.save_all_plots(
            project_data, cpm_result, pert_result, mc_result, str(plots_dir)
        )
        
        print("✅ Generated visualizations:")
        for plot_name, plot_path in saved_plots.items():
            print(f"   {plot_name}: {plot_path}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Analysis Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("\n📋 Summary:")
    print(f"   • Project Duration (CPM): {cpm_result.duration:.1f} days")
    if pert_result:
        print(f"   • Expected Duration (PERT): {pert_result.project_expected_duration:.1f} days")
    if mc_result:
        print(f"   • Mean Duration (Monte Carlo): {mc_result.mean_duration:.1f} days")
    print(f"   • Critical Path Length: {len(cpm_result.critical_path)} tasks")
    print(f"   • Total Slack Time: {sum(cpm_result.slack_times.values()):.1f} days")
    
    print("\n⚠️  DISCLAIMER: These are estimates for research/educational purposes.")
    print("   Always validate with experienced project managers before making decisions.")


if __name__ == "__main__":
    main()
