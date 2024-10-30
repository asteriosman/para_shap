# SHAP Computation with Speedy

This project provides an efficient implementation for computing SHAP (SHapley Additive exPlanations) values using both baseline and parallel methods. The package leverages multi-threading to optimize SHAP computations and includes utilities for visualizing computation times.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Classes and Functions](#classes-and-functions)
  - [ShapleyConfig](#shapleyconfig)
  - [ShapleyComputer](#shapleycomputer)
  - [ShapPlotter](#shapplotter)
- [Example](#example)
- [Benchmarking](#benchmarking)

## Installation

To use this package, ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```
## Usage

1. Load your dataset and train a model.
2. Configure the `ShapleyConfig` for the number of samples and chunks.
3. Initialize the `ShapleyComputer` with your model and explainer class.
4. Compute SHAP values using either baseline or parallel methods.
5. (Optional) Use `ShapPlotter` to visualize computation times.

## Classes and Functions

### ShapleyConfig

A configuration class that defines the parameters for SHAP computation.

- **Attributes:**
  - `num_samples`: Number of samples for SHAP (must be greater than 0).
  - `num_chunks`: Number of chunks for parallel processing (must be greater than 0).
  - `plot_samples`: Optional list of sample sizes for plotting.

### ShapleyComputer

A class for computing SHAP values.

- **Methods:**
  - `__init__(model: Any, explainer_class: Type[shap.Explainer], config: ShapleyConfig)`: Initializes the SHAP computer.
  - `compute_baseline(X: np.ndarray, num_samples: Optional[int] = None)`: Computes SHAP values in a single thread.
  - `compute_parallel(X: np.ndarray, num_samples: Optional[int] = None)`: Computes SHAP values in parallel.
  - `_compute_shapley_chunk(X_chunk: np.ndarray)`: Helper method to compute SHAP values for a single chunk.

### ShapPlotter

A class for plotting SHAP computation times.

- **Methods:**
  - `plot_shap_computation_times(X: np.ndarray, shapley_computer: ShapleyComputer)`: Plots and saves the computation times for baseline and parallel methods.

### Example
Example usage on *example_usage.py*

### Benchmarks
Still running, but using the [ember_v2](https://www.kaggle.com/datasets/dhoogla/ember-2017-v2-features/data?select=test_ember_2017_v2_features.parquet) dataset which has 2381 features and 900k rows approximately. I have only tested it for 10 and 100 samples so far and this is the curve produced:
![output](https://github.com/user-attachments/assets/892508c0-5e46-4786-a4fc-87ccc78f9fcf)
