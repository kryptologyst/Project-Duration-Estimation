"""
Streamlit demo application for project duration estimation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data import generate_sample_project, ProjectData, Task
from src.models.cpm import CriticalPathMethod
from src.models.pert import PERTAnalyzer
from src.models.monte_carlo import MonteCarloSimulator
from src.visualization import ProjectVisualizer
from src.evaluation import ProjectDurationEvaluator, ModelComparison


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Project Duration Estimation",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header with disclaimer
    st.title("📊 Project Duration Estimation Dashboard")
    
    st.warning("""
    **IMPORTANT DISCLAIMER**: This is an experimental research and educational tool. 
    It is NOT intended for automated decision-making without human review. 
    All project duration estimates should be validated by experienced project managers 
    and stakeholders before making any business decisions.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Project parameters
    st.sidebar.subheader("Project Parameters")
    n_tasks = st.sidebar.slider("Number of Tasks", 5, 20, 10)
    min_duration = st.sidebar.slider("Minimum Duration (days)", 1, 5, 1)
    max_duration = st.sidebar.slider("Maximum Duration (days)", 5, 20, 10)
    dependency_prob = st.sidebar.slider("Dependency Probability", 0.1, 0.8, 0.3)
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    enable_pert = st.sidebar.checkbox("Enable PERT Analysis", True)
    enable_monte_carlo = st.sidebar.checkbox("Enable Monte Carlo Simulation", True)
    n_simulations = st.sidebar.slider("Monte Carlo Simulations", 1000, 50000, 10000)
    
    # Generate project data
    if st.sidebar.button("Generate New Project"):
        st.session_state.project_data = generate_sample_project(
            n_tasks=n_tasks,
            min_duration=min_duration,
            max_duration=max_duration,
            dependency_probability=dependency_prob
        )
        st.session_state.analysis_results = None
    
    # Initialize session state
    if 'project_data' not in st.session_state:
        st.session_state.project_data = generate_sample_project(
            n_tasks=n_tasks,
            min_duration=min_duration,
            max_duration=max_duration,
            dependency_probability=dependency_prob
        )
    
    project_data = st.session_state.project_data
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Project Overview", 
        "🔍 CPM Analysis", 
        "📊 PERT Analysis", 
        "🎲 Monte Carlo", 
        "📈 Comparison"
    ])
    
    with tab1:
        show_project_overview(project_data)
    
    with tab2:
        show_cpm_analysis(project_data)
    
    with tab3:
        if enable_pert:
            show_pert_analysis(project_data)
        else:
            st.info("PERT analysis is disabled. Enable it in the sidebar to see results.")
    
    with tab4:
        if enable_monte_carlo:
            show_monte_carlo_analysis(project_data, n_simulations)
        else:
            st.info("Monte Carlo simulation is disabled. Enable it in the sidebar to see results.")
    
    with tab5:
        show_model_comparison(project_data, enable_pert, enable_monte_carlo, n_simulations)


def show_project_overview(project_data: ProjectData):
    """Display project overview."""
    st.header("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tasks", len(project_data.tasks))
    
    with col2:
        total_duration = sum(task.duration for task in project_data.tasks)
        st.metric("Total Duration", f"{total_duration:.1f} days")
    
    with col3:
        avg_duration = total_duration / len(project_data.tasks)
        st.metric("Average Duration", f"{avg_duration:.1f} days")
    
    # Task details table
    st.subheader("Task Details")
    
    task_data = []
    for task in project_data.tasks:
        task_data.append({
            "Task ID": task.task_id,
            "Name": task.name,
            "Duration": f"{task.duration:.1f} days",
            "Dependencies": ", ".join(task.dependencies) if task.dependencies else "None",
            "Risk Level": task.risk_level,
            "Optimistic": f"{task.optimistic_duration:.1f}" if task.optimistic_duration else "N/A",
            "Most Likely": f"{task.most_likely_duration:.1f}" if task.most_likely_duration else "N/A",
            "Pessimistic": f"{task.pessimistic_duration:.1f}" if task.pessimistic_duration else "N/A"
        })
    
    df = pd.DataFrame(task_data)
    st.dataframe(df, use_container_width=True)
    
    # Project network visualization
    st.subheader("Project Network")
    
    # Create a simple network visualization
    cpm = CriticalPathMethod()
    cpm_result = cpm.analyze(project_data)
    
    # Create network plot
    import networkx as nx
    import matplotlib.pyplot as plt
    
    G = nx.DiGraph()
    for task in project_data.tasks:
        G.add_node(task.task_id, duration=task.duration)
        for dep in task.dependencies:
            G.add_edge(dep, task.task_id)
    
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color nodes based on critical path
    critical_path = set(cpm_result.critical_path)
    node_colors = ['red' if node in critical_path else 'lightblue' for node in G.nodes()]
    
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
    
    labels = {node: f"{node}\n({G.nodes[node]['duration']:.1f}d)" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title("Project Task Dependency Network\n(Red = Critical Path)", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    st.pyplot(fig)


def show_cpm_analysis(project_data: ProjectData):
    """Display CPM analysis results."""
    st.header("Critical Path Method (CPM) Analysis")
    
    # Perform CPM analysis
    with st.spinner("Performing CPM analysis..."):
        cpm = CriticalPathMethod()
        cpm_result = cpm.analyze(project_data)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Project Duration", f"{cpm_result.duration:.1f} days")
    
    with col2:
        st.metric("Critical Path Length", len(cpm_result.critical_path))
    
    with col3:
        total_slack = sum(cpm_result.slack_times.values())
        st.metric("Total Slack Time", f"{total_slack:.1f} days")
    
    with col4:
        critical_tasks = sum(1 for slack in cpm_result.slack_times.values() if slack == 0)
        st.metric("Critical Tasks", critical_tasks)
    
    # Critical path
    st.subheader("Critical Path")
    critical_path_str = " → ".join(cpm_result.critical_path)
    st.success(f"**Critical Path:** {critical_path_str}")
    
    # Task details
    st.subheader("Task Schedule")
    
    task_schedule_data = []
    for task_id, details in cpm_result.task_details.items():
        task_schedule_data.append({
            "Task ID": task_id,
            "Name": details['name'],
            "Duration": f"{details['duration']:.1f} days",
            "Early Start": f"{details['early_start']:.1f}",
            "Early Finish": f"{details['early_finish']:.1f}",
            "Late Start": f"{details['late_start']:.1f}",
            "Late Finish": f"{details['late_finish']:.1f}",
            "Slack Time": f"{details['slack_time']:.1f}",
            "Critical": "Yes" if details['is_critical'] else "No"
        })
    
    df = pd.DataFrame(task_schedule_data)
    st.dataframe(df, use_container_width=True)
    
    # Gantt chart
    st.subheader("Gantt Chart")
    
    # Create Gantt chart
    tasks = [details['name'] for details in cpm_result.task_details.values()]
    start_times = [details['early_start'] for details in cpm_result.task_details.values()]
    durations = [details['duration'] for details in cpm_result.task_details.values()]
    is_critical = [details['is_critical'] for details in cpm_result.task_details.values()]
    
    colors = ['red' if critical else 'lightblue' for critical in is_critical]
    
    fig = go.Figure()
    
    for i, (task, start, duration, critical) in enumerate(zip(tasks, start_times, durations, is_critical)):
        fig.add_trace(go.Bar(
            y=[task],
            x=[duration],
            base=[start],
            marker_color='red' if critical else 'lightblue',
            name='Critical' if critical else 'Non-Critical',
            text=[f"{duration:.1f}d"],
            textposition='inside',
            showlegend=False
        ))
    
    fig.update_layout(
        title="Project Gantt Chart",
        xaxis_title="Time (days)",
        yaxis_title="Tasks",
        height=400,
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_pert_analysis(project_data: ProjectData):
    """Display PERT analysis results."""
    st.header("PERT Analysis")
    
    # Perform PERT analysis
    with st.spinner("Performing PERT analysis..."):
        pert = PERTAnalyzer()
        pert_result = pert.analyze(project_data)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Expected Duration", f"{pert_result.project_expected_duration:.1f} days")
    
    with col2:
        st.metric("Standard Deviation", f"{pert_result.project_standard_deviation:.1f} days")
    
    with col3:
        st.metric("Variance", f"{pert_result.project_variance:.1f}")
    
    with col4:
        st.metric("Critical Path Probability", f"{pert_result.critical_path_probability:.1%}")
    
    # Confidence intervals
    st.subheader("Confidence Intervals")
    
    ci_data = []
    for level, (lower, upper) in pert_result.confidence_intervals.items():
        ci_data.append({
            "Confidence Level": f"{level:.0%}",
            "Lower Bound": f"{lower:.1f} days",
            "Upper Bound": f"{upper:.1f} days",
            "Interval Width": f"{upper - lower:.1f} days"
        })
    
    df = pd.DataFrame(ci_data)
    st.dataframe(df, use_container_width=True)
    
    # PERT estimates visualization
    st.subheader("PERT Estimates by Task")
    
    task_ids = list(pert_result.task_estimates.keys())
    expected_durations = [pert_result.task_estimates[tid].expected for tid in task_ids]
    optimistic_durations = [pert_result.task_estimates[tid].optimistic for tid in task_ids]
    pessimistic_durations = [pert_result.task_estimates[tid].pessimistic for tid in task_ids]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=task_ids,
        y=expected_durations,
        mode='markers',
        marker=dict(size=15, color='blue'),
        name='Expected Duration',
        error_y=dict(
            type='data',
            symmetric=False,
            array=[pess - exp for pess, exp in zip(pessimistic_durations, expected_durations)],
            arrayminus=[exp - opt for exp, opt in zip(expected_durations, optimistic_durations)]
        )
    ))
    
    fig.update_layout(
        title="PERT Estimates by Task",
        xaxis_title="Task ID",
        yaxis_title="Duration (days)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_monte_carlo_analysis(project_data: ProjectData, n_simulations: int):
    """Display Monte Carlo simulation results."""
    st.header("Monte Carlo Simulation")
    
    # Perform Monte Carlo simulation
    with st.spinner(f"Running {n_simulations:,} Monte Carlo simulations..."):
        mc = MonteCarloSimulator(n_simulations=n_simulations)
        mc_result = mc.simulate(project_data)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Duration", f"{mc_result.mean_duration:.1f} days")
    
    with col2:
        st.metric("Median Duration", f"{mc_result.median_duration:.1f} days")
    
    with col3:
        st.metric("Standard Deviation", f"{mc_result.std_duration:.1f} days")
    
    with col4:
        st.metric("Coefficient of Variation", f"{mc_result.std_duration / mc_result.mean_duration:.2f}")
    
    # Percentiles
    st.subheader("Duration Percentiles")
    
    percentile_data = []
    for p, value in mc_result.percentiles.items():
        percentile_data.append({
            "Percentile": f"{p}%",
            "Duration": f"{value:.1f} days"
        })
    
    df = pd.DataFrame(percentile_data)
    st.dataframe(df, use_container_width=True)
    
    # Confidence intervals
    st.subheader("Confidence Intervals")
    
    ci_data = []
    for level, (lower, upper) in mc_result.confidence_intervals.items():
        ci_data.append({
            "Confidence Level": f"{level:.0%}",
            "Lower Bound": f"{lower:.1f} days",
            "Upper Bound": f"{upper:.1f} days",
            "Interval Width": f"{upper - lower:.1f} days"
        })
    
    df = pd.DataFrame(ci_data)
    st.dataframe(df, use_container_width=True)
    
    # Distribution visualization
    st.subheader("Simulation Results")
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution of Simulated Durations', 'Cumulative Distribution'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=mc_result.simulated_durations,
            nbinsx=50,
            name='Simulated Durations',
            marker_color='skyblue'
        ),
        row=1, col=1
    )
    
    # Cumulative distribution
    sorted_durations = np.sort(mc_result.simulated_durations)
    cumulative_prob = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    
    fig.add_trace(
        go.Scatter(
            x=sorted_durations,
            y=cumulative_prob,
            mode='lines',
            name='Cumulative Probability',
            line=dict(color='blue', width=2)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Monte Carlo Simulation Results",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk analysis
    st.subheader("Risk Analysis")
    
    risk_metrics = mc.get_risk_metrics(project_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Coefficient of Variation", f"{risk_metrics['coefficient_of_variation']:.3f}")
    
    with col2:
        st.metric("95% Range", f"{risk_metrics['range_95']:.1f} days")
    
    with col3:
        st.metric("Interquartile Range", f"{risk_metrics['interquartile_range']:.1f} days")


def show_model_comparison(project_data: ProjectData, enable_pert: bool, enable_monte_carlo: bool, n_simulations: int):
    """Display model comparison results."""
    st.header("Model Comparison")
    
    # Perform all analyses
    with st.spinner("Running all analyses for comparison..."):
        start_time = time.time()
        
        # CPM Analysis
        cpm = CriticalPathMethod()
        cpm_result = cpm.analyze(project_data)
        cpm_time = time.time() - start_time
        
        # PERT Analysis
        pert_result = None
        pert_time = 0
        if enable_pert:
            pert_start = time.time()
            pert = PERTAnalyzer()
            pert_result = pert.analyze(project_data)
            pert_time = time.time() - pert_start
        
        # Monte Carlo Analysis
        mc_result = None
        mc_time = 0
        if enable_monte_carlo:
            mc_start = time.time()
            mc = MonteCarloSimulator(n_simulations=n_simulations)
            mc_result = mc.simulate(project_data)
            mc_time = time.time() - mc_start
    
    # Create comparison table
    comparison_data = []
    
    # CPM
    comparison_data.append({
        "Model": "CPM",
        "Project Duration": f"{cpm_result.duration:.1f} days",
        "Critical Path Length": len(cpm_result.critical_path),
        "Execution Time": f"{cpm_time:.3f} seconds",
        "Confidence Intervals": "N/A",
        "Risk Assessment": "Basic"
    })
    
    # PERT
    if pert_result:
        ci_90 = pert_result.confidence_intervals.get(0.9, (0, 0))
        comparison_data.append({
            "Model": "PERT",
            "Project Duration": f"{pert_result.project_expected_duration:.1f} days",
            "Critical Path Length": len(cpm_result.critical_path),
            "Execution Time": f"{pert_time:.3f} seconds",
            "Confidence Intervals": f"{ci_90[0]:.1f} - {ci_90[1]:.1f} days",
            "Risk Assessment": "Probabilistic"
        })
    
    # Monte Carlo
    if mc_result:
        ci_90 = mc_result.confidence_intervals.get(0.9, (0, 0))
        comparison_data.append({
            "Model": "Monte Carlo",
            "Project Duration": f"{mc_result.mean_duration:.1f} days",
            "Critical Path Length": len(cpm_result.critical_path),
            "Execution Time": f"{mc_time:.3f} seconds",
            "Confidence Intervals": f"{ci_90[0]:.1f} - {ci_90[1]:.1f} days",
            "Risk Assessment": "Comprehensive"
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Summary insights
    st.subheader("Key Insights")
    
    insights = []
    
    if cpm_result:
        insights.append(f"• **CPM** estimates project duration at **{cpm_result.duration:.1f} days**")
        insights.append(f"• Critical path consists of **{len(cpm_result.critical_path)} tasks**")
    
    if pert_result:
        insights.append(f"• **PERT** provides expected duration of **{pert_result.project_expected_duration:.1f} days** with uncertainty")
        insights.append(f"• **{pert_result.critical_path_probability:.1%}** probability of completing within critical path duration")
    
    if mc_result:
        insights.append(f"• **Monte Carlo** simulation shows mean duration of **{mc_result.mean_duration:.1f} days**")
        insights.append(f"• **{mc_result.std_duration:.1f} days** standard deviation indicates project risk level")
    
    for insight in insights:
        st.write(insight)
    
    # Recommendations
    st.subheader("Recommendations")
    
    st.info("""
    **Based on the analysis results:**
    
    1. **Use CPM** for deterministic project scheduling and critical path identification
    2. **Use PERT** for probabilistic duration estimates and risk assessment
    3. **Use Monte Carlo** for comprehensive uncertainty analysis and scenario planning
    4. **Combine all methods** for robust project planning and risk management
    
    **Remember**: These are estimates based on the provided data. Always validate with 
    experienced project managers and consider external factors not captured in the model.
    """)


if __name__ == "__main__":
    main()
