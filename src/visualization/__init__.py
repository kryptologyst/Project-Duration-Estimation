"""
Visualization utilities for project duration estimation.
"""

from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
from src.data import ProjectData
from src.models.cpm import CPMAnalysisResult
from src.models.pert import PERTAnalysisResult
from src.models.monte_carlo import MonteCarloResult


class ProjectVisualizer:
    """Visualization utilities for project duration estimation."""
    
    def __init__(self, style: str = "seaborn-v0_8", figure_size: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figure_size: Default figure size (width, height)
        """
        self.style = style
        self.figure_size = figure_size
        plt.style.use(style)
    
    def plot_project_network(self, 
                           project_data: ProjectData,
                           cpm_result: CPMAnalysisResult,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a network visualization of the project tasks and dependencies.
        
        Args:
            project_data: Project data
            cpm_result: CPM analysis results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for task in project_data.tasks:
            G.add_node(task.task_id, duration=task.duration, risk_level=task.risk_level)
            for dep in task.dependencies:
                G.add_edge(dep, task.task_id)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Color nodes based on critical path
        critical_path = set(cpm_result.critical_path)
        node_colors = []
        for node in G.nodes():
            if node in critical_path:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        
        # Draw the graph
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=1500,
                with_labels=True,
                font_size=10,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                ax=ax)
        
        # Add duration labels
        labels = {node: f"{node}\n({G.nodes[node]['duration']:.1f}d)" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title("Project Task Dependency Network\n(Red = Critical Path)", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_gantt_chart(self, 
                        cpm_result: CPMAnalysisResult,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a Gantt chart showing task scheduling.
        
        Args:
            cpm_result: CPM analysis results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(self.figure_size[0], len(cpm_result.task_details) * 0.5))
        
        # Prepare data for Gantt chart
        tasks = []
        start_times = []
        durations = []
        colors = []
        
        for task_id, details in cpm_result.task_details.items():
            tasks.append(details['name'])
            start_times.append(details['early_start'])
            durations.append(details['duration'])
            
            # Color based on critical path
            if details['is_critical']:
                colors.append('red')
            else:
                colors.append('lightblue')
        
        # Create horizontal bar chart
        y_pos = np.arange(len(tasks))
        bars = ax.barh(y_pos, durations, left=start_times, color=colors, alpha=0.7)
        
        # Customize the chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tasks)
        ax.set_xlabel('Time (days)')
        ax.set_title('Project Gantt Chart\n(Red = Critical Path Tasks)', fontsize=16, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Critical Path'),
            Patch(facecolor='lightblue', alpha=0.7, label='Non-Critical')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pert_distribution(self, 
                              pert_result: PERTAnalysisResult,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a visualization of PERT estimates and confidence intervals.
        
        Args:
            pert_result: PERT analysis results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figure_size[0] * 1.5, self.figure_size[1]))
        
        # Plot 1: Task-level PERT estimates
        task_ids = list(pert_result.task_estimates.keys())
        expected_durations = [pert_result.task_estimates[tid].expected for tid in task_ids]
        optimistic_durations = [pert_result.task_estimates[tid].optimistic for tid in task_ids]
        pessimistic_durations = [pert_result.task_estimates[tid].pessimistic for tid in task_ids]
        
        x_pos = np.arange(len(task_ids))
        
        # Create error bars for PERT estimates
        lower_errors = [expected - optimistic for expected, optimistic in zip(expected_durations, optimistic_durations)]
        upper_errors = [pessimistic - expected for expected, pessimistic in zip(expected_durations, pessimistic_durations)]
        
        ax1.errorbar(x_pos, expected_durations, 
                    yerr=[lower_errors, upper_errors],
                    fmt='o', capsize=5, capthick=2, markersize=8)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(task_ids, rotation=45)
        ax1.set_ylabel('Duration (days)')
        ax1.set_title('PERT Estimates by Task', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Project-level confidence intervals
        confidence_levels = list(pert_result.confidence_intervals.keys())
        lower_bounds = [pert_result.confidence_intervals[level][0] for level in confidence_levels]
        upper_bounds = [pert_result.confidence_intervals[level][1] for level in confidence_levels]
        interval_widths = [upper - lower for upper, lower in zip(upper_bounds, lower_bounds)]
        
        ax2.bar(confidence_levels, interval_widths, 
                bottom=lower_bounds, alpha=0.7, color='skyblue')
        ax2.axhline(y=pert_result.project_expected_duration, color='red', 
                   linestyle='--', linewidth=2, label='Expected Duration')
        
        ax2.set_xlabel('Confidence Level')
        ax2.set_ylabel('Duration (days)')
        ax2.set_title('Project Duration Confidence Intervals', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_monte_carlo_results(self, 
                                monte_carlo_result: MonteCarloResult,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualizations of Monte Carlo simulation results.
        
        Args:
            monte_carlo_result: Monte Carlo simulation results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.figure_size[0] * 1.5, self.figure_size[1] * 1.5))
        
        # Plot 1: Distribution of simulated durations
        ax1.hist(monte_carlo_result.simulated_durations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(monte_carlo_result.mean_duration, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {monte_carlo_result.mean_duration:.2f}')
        ax1.axvline(monte_carlo_result.median_duration, color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {monte_carlo_result.median_duration:.2f}')
        ax1.set_xlabel('Project Duration (days)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Monte Carlo Simulation Results', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative distribution
        sorted_durations = np.sort(monte_carlo_result.simulated_durations)
        cumulative_prob = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
        ax2.plot(sorted_durations, cumulative_prob, linewidth=2, color='blue')
        ax2.set_xlabel('Project Duration (days)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Percentiles
        percentiles = list(monte_carlo_result.percentiles.keys())
        percentile_values = list(monte_carlo_result.percentiles.values())
        ax3.bar(percentiles, percentile_values, alpha=0.7, color='lightcoral')
        ax3.set_xlabel('Percentile')
        ax3.set_ylabel('Duration (days)')
        ax3.set_title('Duration Percentiles', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Confidence intervals
        confidence_levels = list(monte_carlo_result.confidence_intervals.keys())
        lower_bounds = [monte_carlo_result.confidence_intervals[level][0] for level in confidence_levels]
        upper_bounds = [monte_carlo_result.confidence_intervals[level][1] for level in confidence_levels]
        interval_widths = [upper - lower for upper, lower in zip(upper_bounds, lower_bounds)]
        
        ax4.bar(confidence_levels, interval_widths, 
                bottom=lower_bounds, alpha=0.7, color='lightgreen')
        ax4.axhline(y=monte_carlo_result.mean_duration, color='red', 
                   linestyle='--', linewidth=2, label='Mean Duration')
        
        ax4.set_xlabel('Confidence Level')
        ax4.set_ylabel('Duration (days)')
        ax4.set_title('Confidence Intervals', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, 
                                   project_data: ProjectData,
                                   cpm_result: CPMAnalysisResult,
                                   pert_result: Optional[PERTAnalysisResult] = None,
                                   monte_carlo_result: Optional[MonteCarloResult] = None) -> go.Figure:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            project_data: Project data
            cpm_result: CPM analysis results
            pert_result: Optional PERT analysis results
            monte_carlo_result: Optional Monte Carlo simulation results
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Project Network', 'Gantt Chart', 'PERT Estimates', 'Monte Carlo Results'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Plot 1: Project Network (simplified)
        task_ids = list(cpm_result.task_details.keys())
        durations = [cpm_result.task_details[tid]['duration'] for tid in task_ids]
        is_critical = [cpm_result.task_details[tid]['is_critical'] for tid in task_ids]
        
        colors = ['red' if critical else 'lightblue' for critical in is_critical]
        
        fig.add_trace(
            go.Scatter(
                x=task_ids,
                y=durations,
                mode='markers',
                marker=dict(size=15, color=colors),
                text=[f"Task: {tid}<br>Duration: {dur:.1f}d<br>Critical: {crit}" 
                      for tid, dur, crit in zip(task_ids, durations, is_critical)],
                hovertemplate='%{text}<extra></extra>',
                name='Tasks'
            ),
            row=1, col=1
        )
        
        # Plot 2: Gantt Chart
        tasks = [cpm_result.task_details[tid]['name'] for tid in task_ids]
        start_times = [cpm_result.task_details[tid]['early_start'] for tid in task_ids]
        
        fig.add_trace(
            go.Bar(
                y=tasks,
                x=durations,
                base=start_times,
                marker_color=colors,
                name='Task Duration',
                text=[f"{dur:.1f}d" for dur in durations],
                textposition='inside'
            ),
            row=1, col=2
        )
        
        # Plot 3: PERT Estimates (if available)
        if pert_result:
            expected_durations = [pert_result.task_estimates[tid].expected for tid in task_ids]
            optimistic_durations = [pert_result.task_estimates[tid].optimistic for tid in task_ids]
            pessimistic_durations = [pert_result.task_estimates[tid].pessimistic for tid in task_ids]
            
            fig.add_trace(
                go.Scatter(
                    x=task_ids,
                    y=expected_durations,
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    name='Expected Duration',
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[pess - exp for pess, exp in zip(pessimistic_durations, expected_durations)],
                        arrayminus=[exp - opt for exp, opt in zip(expected_durations, optimistic_durations)]
                    )
                ),
                row=2, col=1
            )
        
        # Plot 4: Monte Carlo Results (if available)
        if monte_carlo_result:
            fig.add_trace(
                go.Histogram(
                    x=monte_carlo_result.simulated_durations,
                    nbinsx=50,
                    name='Simulated Durations',
                    marker_color='skyblue'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Project Duration Estimation Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def save_all_plots(self, 
                      project_data: ProjectData,
                      cpm_result: CPMAnalysisResult,
                      pert_result: Optional[PERTAnalysisResult] = None,
                      monte_carlo_result: Optional[MonteCarloResult] = None,
                      output_dir: str = "assets/plots") -> Dict[str, str]:
        """
        Save all visualization plots to files.
        
        Args:
            project_data: Project data
            cpm_result: CPM analysis results
            pert_result: Optional PERT analysis results
            monte_carlo_result: Optional Monte Carlo simulation results
            output_dir: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # Save network plot
        network_path = f"{output_dir}/project_network.png"
        self.plot_project_network(project_data, cpm_result, network_path)
        saved_files["network"] = network_path
        
        # Save Gantt chart
        gantt_path = f"{output_dir}/gantt_chart.png"
        self.plot_gantt_chart(cpm_result, gantt_path)
        saved_files["gantt"] = gantt_path
        
        # Save PERT plot (if available)
        if pert_result:
            pert_path = f"{output_dir}/pert_analysis.png"
            self.plot_pert_distribution(pert_result, pert_path)
            saved_files["pert"] = pert_path
        
        # Save Monte Carlo plot (if available)
        if monte_carlo_result:
            mc_path = f"{output_dir}/monte_carlo_results.png"
            self.plot_monte_carlo_results(monte_carlo_result, mc_path)
            saved_files["monte_carlo"] = mc_path
        
        # Save interactive dashboard
        dashboard_path = f"{output_dir}/interactive_dashboard.html"
        dashboard_fig = self.create_interactive_dashboard(
            project_data, cpm_result, pert_result, monte_carlo_result
        )
        dashboard_fig.write_html(dashboard_path)
        saved_files["dashboard"] = dashboard_path
        
        return saved_files
