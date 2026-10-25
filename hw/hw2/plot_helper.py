# === plot_helper.py ===
# A universal plotting helper for HFD, CWT, and DWT clustering visualization.
# DO NOT MODIFY THIS FILE.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_clusters_auto(X, labels, times=None, title="Cluster Visualization", y_label="Feature value"):
    """
    Generic cluster visualization for HFD-only, CWT-only, and DWT-only pipelines.

    Behavior:
    - If X has ≥ 2 features: plot 2D PCA scatter colored by cluster labels.
    - If X has exactly 1 feature: plot that feature versus time (or window index) colored by cluster labels.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The standardized feature matrix (e.g., from StandardScaler).
        - Rows correspond to sliding windows.
        - Columns correspond to extracted features (HFD, CWT, or DWT).

    labels : np.ndarray, shape (n_samples,)
        The cluster labels (e.g., from sklearn KMeans).
        - Each label represents the cluster assignment for one window.
        - Typically 0 = Normal, 1 = Asystole (interpretation left to analysis).

    times : np.ndarray or None, optional
        The starting time (in seconds) of each sliding window.
        - Used for x-axis in 1D feature plots.
        - If None, the x-axis will simply be the window index.

    title : str, optional
        Title prefix for the plot (e.g., "HFD Features — K-Means (k=2)").

    y_label : str, optional
        Label for the y-axis in 1D feature plots.
        - Defaults to "Feature value", but can be customized.
        - Example: "HFD value", "CWT energy", "DWT coefficient energy", etc.

    Returns
    -------
    None
        Displays the plot directly.
    """

    n_samples, n_features = X.shape

    if n_features >= 2:
        # --- PCA 2D scatter ---
        n_comp = min(2, n_samples, n_features)
        X_2d = PCA(n_components=n_comp, random_state=42).fit_transform(X)
        plt.figure(figsize=(6.8, 5.2))
        for lab in np.unique(labels):
            idx = (labels == lab)
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=22, alpha=0.85, label=f"Cluster {lab}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{title} (PCA 2D)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        # --- Single feature: plot feature value over time or index ---
        x1 = X[:, 0]
        x_axis = times if times is not None else np.arange(n_samples)
        xlabel = "Time (s)" if times is not None else "Window index"

        plt.figure(figsize=(8.2, 3.6))
        for lab in np.unique(labels):
            idx = (labels == lab)
            plt.scatter(x_axis[idx], x1[idx], s=24, alpha=0.85, label=f"Cluster {lab}")
        plt.xlabel(xlabel)
        plt.ylabel(y_label)
        plt.title(f"{title} (1D feature over time)")
        plt.legend()
        plt.tight_layout()
        plt.show()
# === end of plot_helper.py ===