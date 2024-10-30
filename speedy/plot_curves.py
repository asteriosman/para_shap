import logging
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from speedy.shapley_computer import ShapleyComputer

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShapPlotter:
    def plot_shap_computation_times(
        self, X: np.ndarray, shapley_computer: ShapleyComputer
    ) -> None:
        max_samples = len(X)

        # Filter plot_samples to ensure sizes are within max_samples
        sample_sizes = [
            size for size in shapley_computer.plot_samples if size <= max_samples
        ]

        # Initialize lists to store computation times
        baseline_times = []
        parallel_times = []

        # Iterate through each sample size and compute SHAP values for baseline and parallel methods
        for size in sample_sizes:
            # Compute baseline SHAP values and measure time
            baseline_start_time = time.time()
            shapley_computer.compute_baseline(X, num_samples=size)
            baseline_times.append(time.time() - baseline_start_time)

            # Compute parallel SHAP values and measure time
            parallel_start_time = time.time()
            shapley_computer.compute_parallel(X, num_samples=size)
            parallel_times.append(time.time() - parallel_start_time)

        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, baseline_times, label="Baseline", marker="o")
        plt.plot(sample_sizes, parallel_times, label="Parallel", marker="o")
        plt.xscale("log")
        plt.xlabel("Number of Samples")
        plt.ylabel("Computation Time (seconds)")
        plt.title("SHAP Computation Time: Baseline vs Parallel")
        plt.legend()
        plt.grid()

        # Save the plot
        plot_filename = "shap_computation_time_plot.png"
        plt.savefig(plot_filename)
        logger.info(f"Plot saved as {plot_filename}")
        plt.show()
