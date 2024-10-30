# example usage of the speedy package

import shap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from speedy.plot_curves import ShapPlotter
from speedy.shapley_computer import ShapleyComputer, ShapleyConfig

# 1. Load your data
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(len(X_test))

# 2. Initialize and train a model on the training data OR load a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# model = joblib.load('xxx.pkl')

# 3. define the config for num_samples, num_chunks and plot_samples
config = ShapleyConfig(num_samples=100, num_chunks=4, plot_samples=[10, 20, 50, 100])

# 4. Initialize the SHAP computer with the trained model
shapley_computer = ShapleyComputer(
    model=model.predict_proba,  # Pass the model's probability prediction method
    explainer_class=shap.KernelExplainer,  # Pass the desired shap explainer
    config=config,
)
# # 5. Run SHAP computations
print("Running parallel SHAP computation...")
shap_values_parallel = shapley_computer.compute_parallel(X_test)
print("Parallel SHAP values calculated.")

print("Running baseline SHAP computation...")
shap_values_baseline = shapley_computer.compute_baseline(X_test)
print("Baseline SHAP values calculated.")

# 6. Initialize the plotter, if you want to produce computation plot
plotter = ShapPlotter()
plotter.plot_shap_computation_times(X_test, shapley_computer)
