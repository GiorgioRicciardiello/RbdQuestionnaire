import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config.config import config

sns.set(style="whitegrid")

# --- Stage simulation ---
def simulate_stage(true_labels, sensitivity, specificity):
    results = np.zeros_like(true_labels)
    for i, label in enumerate(true_labels):
        if label == 1:  # Case
            results[i] = np.random.rand() < sensitivity
        else:  # Control
            results[i] = np.random.rand() < specificity
    return results

# --- Metric calculation ---
def compute_metrics(predictions, true_labels):
    tp = np.sum((predictions == 1) & (true_labels == 1))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity

# --- Full simulation ---
def run_simulation(sens1, spec1, sens2, spec2, true_labels=None, n_samples=10000, case_ratio=0.5):
    if true_labels is None:
        true_labels = np.random.choice([0, 1], size=n_samples, p=[1 - case_ratio, case_ratio])
    stage1 = simulate_stage(true_labels, sens1, spec1)
    stage2 = simulate_stage(true_labels, sens2, spec2)
    combined = stage1 & stage2
    return compute_metrics(combined, true_labels)

# --- Heatmap plotting ---
def plot_dual_heatmap(sens_matrix, spec_matrix, x_vals, y_vals, xlabel, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(sens_matrix, xticklabels=np.round(x_vals, 2), yticklabels=np.round(y_vals, 2),
                cmap="viridis", annot=False, ax=axes[0])
    axes[0].set_title("Final Sensitivity")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    sns.heatmap(spec_matrix, xticklabels=np.round(x_vals, 2), yticklabels=np.round(y_vals, 2),
                cmap="plasma", annot=False, ax=axes[1])
    axes[1].set_title("Final Specificity")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()

# --- Main exploration ---
def explore_sens1_spec2_grid(fixed_spec1=0.9, fixed_sens2=0.9, true_labels=None):
    sens1_values = np.linspace(0.5, 1.0, 20)
    spec2_values = np.linspace(0.5, 1.0, 20)

    final_sens = np.zeros((len(spec2_values), len(sens1_values)))
    final_spec = np.zeros((len(spec2_values), len(sens1_values)))

    for i, spec2 in enumerate(spec2_values):
        for j, sens1 in enumerate(sens1_values):
            sens, spec = run_simulation(
                sens1=sens1,
                spec1=fixed_spec1,
                sens2=fixed_sens2,
                spec2=spec2,
                true_labels=true_labels
            )
            final_sens[i, j] = sens
            final_spec[i, j] = spec

    plot_dual_heatmap(final_sens, final_spec, sens1_values, spec2_values,
                      xlabel="Stage 1 Sensitivity", ylabel="Stage 2 Specificity")

# --- Example run ---
if __name__ == "__main__":
    # If you have real labels:
    df = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    df = df.loc[df['has_quest'] == 1, :]
    true_labels = df['diagnosis'].values
    # true_labels = None  # Simulate if None

    explore_sens1_spec2_grid(fixed_spec1=0.9, fixed_sens2=0.9, true_labels=true_labels)
