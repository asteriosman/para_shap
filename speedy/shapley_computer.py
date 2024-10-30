import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any, List, Optional, Type

import numpy as np
import shap
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress SHAP library logging
shap_logger = logging.getLogger("shap")
shap_logger.setLevel(logging.WARNING)


class ShapleyConfig(BaseModel):
    num_samples: Annotated[int, Field(strict=True, gt=0)]  # Number of samples for SHAP
    num_chunks: Annotated[
        int, Field(strict=True, gt=0)
    ]  # Number of chunks for parallel processing
    plot_samples: Optional[List[int]] = None  # Sample sizes for plotting


class ShapleyComputer:
    def __init__(
        self, model: Any, explainer_class: Type[shap.Explainer], config: ShapleyConfig
    ):
        self.model = model
        self.explainer_class = explainer_class
        self.num_samples = config.num_samples
        self.num_chunks = config.num_chunks
        self.plot_samples = config.plot_samples

    def compute_baseline(
        self, X: np.ndarray, num_samples: Optional[int] = None
    ) -> List[np.ndarray]:
        """Compute SHAP values in a single thread."""
        start_time = time.time()
        num_samples = num_samples or self.num_samples
        X_samples = X[:num_samples]

        explainer = self.explainer_class(self.model, X_samples)
        shap_values = explainer.shap_values(X_samples)

        elapsed_time = time.time() - start_time
        logger.info(f"Baseline SHAP computation took {elapsed_time:.2f} seconds.")

        return shap_values

    def compute_parallel(
        self, X: np.ndarray, num_samples: Optional[int] = None
    ) -> np.ndarray:
        """Compute SHAP values in parallel using chunks of data."""
        # Determine the number of samples to use
        num_samples = num_samples or self.num_samples
        X_samples = X[:num_samples]
        chunk_size = max(1, len(X_samples) // self.num_chunks)
        chunks = [
            X_samples[i : i + chunk_size] for i in range(0, len(X_samples), chunk_size)
        ]

        shap_values_list = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.num_chunks) as executor:
            futures = {
                executor.submit(self._compute_shapley_chunk, chunk): chunk
                for chunk in chunks
            }

            for future in as_completed(futures):
                try:
                    shap_values = future.result()

                    if not shap_values_list:
                        shap_values_list.append(shap_values)
                    else:
                        if shap_values.shape[1] == shap_values_list[0].shape[1]:
                            shap_values_list.append(shap_values)
                        else:
                            logger.warning(
                                f"Inconsistent SHAP output dimensions: {shap_values.shape} (expected {shap_values_list[0].shape})"
                            )
                except Exception as e:
                    logger.error(f"Error computing SHAP values for chunk: {e}")

        elapsed_time = time.time() - start_time
        logger.info(f"Parallel SHAP computation took {elapsed_time:.2f} seconds.")

        if shap_values_list:
            return np.concatenate(shap_values_list, axis=0)
        else:
            raise ValueError("No SHAP values were computed.")

    def _compute_shapley_chunk(self, X_chunk: np.ndarray) -> np.ndarray:
        """Compute SHAP values for a single chunk."""
        explainer = self.explainer_class(self.model, X_chunk)
        return explainer.shap_values(X_chunk)
