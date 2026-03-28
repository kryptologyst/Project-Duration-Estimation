# Project Duration Estimation

## DISCLAIMER

**IMPORTANT: This is an experimental research and educational tool. It is NOT intended for automated decision-making without human review. All project duration estimates should be validated by experienced project managers and stakeholders before making any business decisions.**

## Overview

This project provides advanced project duration estimation capabilities using multiple methodologies:

- **Critical Path Method (CPM)**: Identifies the longest path through dependent tasks
- **PERT Analysis**: Incorporates optimistic, most likely, and pessimistic time estimates
- **Monte Carlo Simulation**: Provides probabilistic duration estimates with confidence intervals
- **Risk Analysis**: Identifies critical tasks and potential bottlenecks

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from src.models.cpm import CriticalPathMethod
from src.data.synthetic import generate_sample_project

# Generate sample project data
project_data = generate_sample_project()

# Initialize CPM analyzer
cpm = CriticalPathMethod()
results = cpm.analyze(project_data)

print(f"Critical Path: {' -> '.join(results.critical_path)}")
print(f"Project Duration: {results.duration} days")
```

### Interactive Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
src/
├── data/           # Data loading and synthetic generation
├── features/       # Feature engineering
├── models/         # CPM, PERT, Monte Carlo models
├── evaluation/     # Metrics and evaluation
├── visualization/  # Plotting and visualization
└── utils/          # Utilities and helpers

configs/            # Configuration files
scripts/            # Training and evaluation scripts
tests/              # Unit tests
assets/             # Generated plots and reports
demo/               # Streamlit demo application
```

## Dataset Schema

### Project Tasks
- `task_id`: Unique task identifier
- `name`: Task description
- `duration`: Estimated duration (days)
- `dependencies`: List of prerequisite task IDs
- `resource_requirements`: Required resources
- `risk_level`: Low/Medium/High risk assessment

### PERT Estimates (optional)
- `optimistic`: Best-case scenario duration
- `most_likely`: Most probable duration
- `pessimistic`: Worst-case scenario duration

## Evaluation Metrics

### Business KPIs
- **Project Duration**: Total estimated time
- **Critical Path Length**: Longest dependency chain
- **Slack Time**: Available buffer for non-critical tasks
- **Risk Score**: Overall project risk assessment

### Technical Metrics
- **Schedule Accuracy**: MAE/RMSE vs actual durations
- **Confidence Intervals**: 90%/95% prediction intervals
- **Sensitivity Analysis**: Impact of duration changes

## Limitations

- Estimates are based on historical patterns and may not account for unforeseen circumstances
- Resource constraints and availability are simplified
- External dependencies and stakeholder delays are not modeled
- Results should be validated by experienced project managers

## Contributing

1. Install development dependencies: `pip install -e ".[dev]"`
2. Run pre-commit hooks: `pre-commit install`
3. Run tests: `pytest`
4. Format code: `black src/ tests/`

## License

MIT License - see LICENSE file for details.
# Project-Duration-Estimation
