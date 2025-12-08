# Customer Review AI Dashboard

This repository contains the code and resources for building an AI-powered dashboard to analyze and visualize customer reviews. The project leverages Python, Jupyter notebooks, and Streamlit for interactive dashboards, making it easy to experiment with and deploy analytical workflows.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Files and Structure](#files-and-structure)
- [Getting Started](#getting-started)
- [Features](#features)
- [Dependencies](#dependencies)
- [Development Environment](#development-environment)
- [Report](#report)
- [License](#license)

---

## Project Overview

The goal of this project is to automate the analysis of customer reviews using AI techniques and visualize results through a user-friendly dashboard. It is targeted at businesses and analysts seeking actionable insights from large volumes of review data.

---

## Files and Structure

- [`Task1.ipynb`](https://github.com/Vishnu4716/customer-review-ai-dashboard/blob/main/Task1.ipynb):  
  Jupyter notebook containing the workflow for processing customer reviews, feature engineering, modeling, and exploratory analysis.
- [`streamlit_app.py`](https://github.com/Vishnu4716/customer-review-ai-dashboard/blob/main/streamlit_app.py):  
  Streamlit application for interactive visualization and analysis of the results from the notebook. This script makes the data insights accessible in a dashboard format.
- [`requirements.txt`](https://github.com/Vishnu4716/customer-review-ai-dashboard/blob/main/requirements.txt):  
  Lists all Python dependencies required to run the notebook and dashboard.
- [`REPORT_TASK1_TASK2.pdf`](https://github.com/Vishnu4716/customer-review-ai-dashboard/blob/main/REPORT_TASK1_TASK2.pdf):  
  Comprehensive PDF report describing tasks performed, methodologies, results, and key findings.
- [`/.devcontainer/`](https://github.com/Vishnu4716/customer-review-ai-dashboard/tree/main/.devcontainer):  
  Configuration folder for development containers—helps ensure a consistent coding environment.
- [`.gitignore`](https://github.com/Vishnu4716/customer-review-ai-dashboard/blob/main/.gitignore):  
  Specifies files and folders to be ignored by git version control.

---

## Getting Started

**1. Clone the repository:**
```bash
git clone https://github.com/Vishnu4716/customer-review-ai-dashboard.git
cd customer-review-ai-dashboard
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit dashboard:**
```bash
streamlit run streamlit_app.py
```

**4. Explore the Jupyter notebook:**  
Open `Task1.ipynb` in JupyterLab, VS Code, or your preferred notebook environment to review the data preparation, analysis, and modeling steps.

---

## Features

- **Automated Data Preprocessing:**  
  Scripts for efficiently collecting, cleaning, and preparing customer review datasets.
- **AI-driven Insights:**  
  Machine learning and statistical analysis to uncover sentiment, trends, and significant factors in reviews.
- **Interactive Dashboard:**  
  Streamlit-based, supporting real-time visualizations, tables, and summary statistics for non-technical users.
- **Extensible Workflow:**  
  Modular notebook enables rapid experimentation with different models or techniques.

---

## Dependencies

Key packages (see `requirements.txt` for full list):

- pandas
- numpy
- scikit-learn
- streamlit

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Development Environment

To ensure consistency, especially across different developer machines, a devcontainer (`/.devcontainer/`) configuration is provided, which is ideal for VS Code or remote environments.

---

## Report

For detailed explanations of the data, tasks, results, and conclusions, see the included [`REPORT_TASK1_TASK2.pdf`](https://github.com/Vishnu4716/customer-review-ai-dashboard/blob/main/REPORT_TASK1_TASK2.pdf).

---

## License

For educational and demonstration purposes only. For commercial or production use, please refer to the repository license or contact the author.

---

**Author:** [Vishnu4716](https://github.com/Vishnu4716)
