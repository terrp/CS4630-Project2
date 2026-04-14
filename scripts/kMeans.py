import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compactness(X_data, labels, centers):
    distances = np.linalg.norm(X_data - centers[labels], axis=1)
    return float(np.mean(distances))


def separation(centers):
    dists = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dists.append(np.linalg.norm(centers[i] - centers[j]))
    return float(np.mean(dists)) if dists else 0.0


def cluster_size_stats(labels, k):
    counts = np.bincount(labels, minlength=k)
    return {
        "Cluster Sizes": counts,
        "Min Cluster Size": int(np.min(counts)),
        "Max Cluster Size": int(np.max(counts)),
        "Cluster Size Std": float(np.std(counts)),
    }


def compute_stability(X_data, k, n_runs=5):
    all_labels = []
    seeds = [42 + i for i in range(n_runs)]

    for seed in seeds:
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(X_data)
        all_labels.append(labels)

    pairwise_ari = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            pairwise_ari.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    return float(np.mean(pairwise_ari)) if pairwise_ari else np.nan


def evaluate(X_data, labels, model, true_labels=None, explained_variance=None, k=2):
    log("  Computing evaluation metrics...")

    metrics = {
        "Dataset": "",
        "Silhouette": silhouette_score(X_data, labels, sample_size=10000, random_state=42),
        "Davies-Bouldin": davies_bouldin_score(X_data, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X_data, labels),
        "Compactness": compactness(X_data, labels, model.cluster_centers_),
        "Separation": separation(model.cluster_centers_),
        "Inertia": float(model.inertia_),
        "Explained Variance": np.nan if explained_variance is None else float(explained_variance),
        "Stability ARI": compute_stability(X_data, k=k, n_runs=5),
    }

    size_info = cluster_size_stats(labels, k)
    metrics["Min Cluster Size"] = size_info["Min Cluster Size"]
    metrics["Max Cluster Size"] = size_info["Max Cluster Size"]
    metrics["Cluster Size Std"] = size_info["Cluster Size Std"]

    metrics["ARI vs Truth"] = np.nan
    metrics["NMI vs Truth"] = np.nan
    metrics["Homogeneity"] = np.nan
    metrics["Completeness"] = np.nan
    metrics["V-Measure"] = np.nan

    if true_labels is not None and len(true_labels) == len(labels):
        metrics["ARI vs Truth"] = adjusted_rand_score(true_labels, labels)
        metrics["NMI vs Truth"] = normalized_mutual_info_score(true_labels, labels)
        metrics["Homogeneity"] = homogeneity_score(true_labels, labels)
        metrics["Completeness"] = completeness_score(true_labels, labels)
        metrics["V-Measure"] = v_measure_score(true_labels, labels)
    elif true_labels is not None:
        log(
            f"  Warning: skipping truth-based metrics because "
            f"{len(true_labels)=} and {len(labels)=} do not match"
        )

    return metrics, size_info["Cluster Sizes"]


def run_kmeans(X_data, name, k=2, true_labels=None, explained_variance=None):
    if not isinstance(X_data, np.ndarray):
        raise TypeError(f"{name}: expected numpy array before KMeans, got {type(X_data)}")

    log(f"Running K-Means on {name}...")
    log(f"  dtype={X_data.dtype}, shape={X_data.shape}")

    model = KMeans(n_clusters=k, random_state=42, n_init=10)

    start = time.time()
    labels = model.fit_predict(X_data)
    log(f"K-Means complete ({name}) in {time.time() - start:.2f}s")

    metrics, cluster_sizes = evaluate(
        X_data,
        labels,
        model,
        true_labels=true_labels,
        explained_variance=explained_variance,
        k=k,
    )
    metrics["Dataset"] = name
    return labels, model, metrics, cluster_sizes


def save_metric_bar_chart(results_df, metric_name, filename, higher_is_better=True):
    log(f"Creating graph for {metric_name}...")

    plot_df = results_df[["Dataset", metric_name]].dropna().copy()
    if plot_df.empty:
        log(f"  Skipping {metric_name}: no valid values")
        return

    plot_df = plot_df.sort_values(metric_name, ascending=not higher_is_better)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_df["Dataset"], plot_df[metric_name])
    plt.title(f"{metric_name} by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel(metric_name)
    plt.xticks(rotation=20)
    plt.tight_layout()

    y_min, y_max = plt.ylim()
    offset = (y_max - y_min) * 0.01 if y_max > y_min else 0.01

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.savefig(filename)
    plt.close()
    log(f"Saved {filename}")


def save_cluster_size_chart(cluster_sizes_by_dataset, filename="cluster_size_distribution.png"):
    log("Creating cluster size distribution graph...")

    datasets = list(cluster_sizes_by_dataset.keys())
    size_arrays = [cluster_sizes_by_dataset[name] for name in datasets]
    k = len(size_arrays[0])

    x = np.arange(len(datasets))
    width = 0.8 / k

    plt.figure(figsize=(10, 6))
    for cluster_idx in range(k):
        values = [sizes[cluster_idx] for sizes in size_arrays]
        plt.bar(x + cluster_idx * width, values, width=width, label=f"Cluster {cluster_idx}")

    plt.xticks(x + width * (k - 1) / 2, datasets, rotation=20)
    plt.xlabel("Dataset")
    plt.ylabel("Cluster Size")
    plt.title("Cluster Size Distribution by Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    log(f"Saved {filename}")


def save_all_metric_graphs(results_df):
    metric_configs = [
        ("Silhouette", True, "metric_silhouette.png"),
        ("Davies-Bouldin", False, "metric_davies_bouldin.png"),
        ("Calinski-Harabasz", True, "metric_calinski_harabasz.png"),
        ("Compactness", False, "metric_compactness.png"),
        ("Separation", True, "metric_separation.png"),
        ("Inertia", False, "metric_inertia.png"),
        ("Explained Variance", True, "metric_explained_variance.png"),
        ("ARI vs Truth", True, "metric_ari_vs_truth.png"),
        ("NMI vs Truth", True, "metric_nmi_vs_truth.png"),
        ("Homogeneity", True, "metric_homogeneity.png"),
        ("Completeness", True, "metric_completeness.png"),
        ("V-Measure", True, "metric_v_measure.png"),
        ("Stability ARI", True, "metric_stability_ari.png"),
        ("Min Cluster Size", True, "metric_min_cluster_size.png"),
        ("Max Cluster Size", True, "metric_max_cluster_size.png"),
        ("Cluster Size Std", False, "metric_cluster_size_std.png"),
    ]

    for metric_name, higher_is_better, filename in metric_configs:
        if metric_name in results_df.columns:
            save_metric_bar_chart(results_df, metric_name, filename, higher_is_better=higher_is_better)

    log("Creating combined internal metrics graph...")
    internal_metrics = [
        "Silhouette",
        "Davies-Bouldin",
        "Calinski-Harabasz",
        "Compactness",
        "Separation",
        "Inertia",
    ]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(results_df["Dataset"]))
    for metric in internal_metrics:
        if metric in results_df.columns:
            plt.plot(x, results_df[metric], marker="o", label=metric)

    plt.xticks(x, results_df["Dataset"], rotation=20)
    plt.xlabel("Dataset")
    plt.ylabel("Metric Value")
    plt.title("Internal Clustering Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("all_internal_metrics_combined.png")
    plt.close()
    log("Saved all_internal_metrics_combined.png")

    log("Creating combined label-based metrics graph...")
    label_metrics = ["ARI vs Truth", "NMI vs Truth", "Homogeneity", "Completeness", "V-Measure"]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(results_df["Dataset"]))
    plotted_any = False
    for metric in label_metrics:
        if metric in results_df.columns and results_df[metric].notna().any():
            plt.plot(x, results_df[metric], marker="o", label=metric)
            plotted_any = True

    plt.xticks(x, results_df["Dataset"], rotation=20)
    plt.xlabel("Dataset")
    plt.ylabel("Metric Value")
    plt.title("Label-Based Clustering Metrics")
    if plotted_any:
        plt.legend()
    plt.tight_layout()
    plt.savefig("all_label_metrics_combined.png")
    plt.close()
    log("Saved all_label_metrics_combined.png")


def build_and_save_pca_results(X_scaled, dims=(2, 5, 10), output_path="pca_results.pkl"):
    log("Regenerating PCA results from cleaned dataset...")
    pca_results = {}

    for dim in dims:
        log(f"  Fitting PCA for {dim} components...")
        pca = PCA(n_components=dim, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        pca_results[dim] = {
            "data": X_pca,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }

        log(
            f"  PCA {dim}D complete: shape={X_pca.shape}, "
            f"explained_variance_sum={np.sum(pca.explained_variance_ratio_):.6f}"
        )

    with open(output_path, "wb") as f:
        pickle.dump(pca_results, f)

    log(f"Saved regenerated PCA results to {output_path}")
    return pca_results


# Load cleaned dataset

log("Loading dataset...")
df = pd.read_csv("data/processed/higgs_200k.csv")
log(f"Dataset loaded with shape: {df.shape}")

feature_cols = [f"feature_{i}" for i in range(1, 29)]
X = df[feature_cols].copy()
true_labels = df["label"].to_numpy()

log("Scaling full 28D dataset...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
log("Scaling complete.")

k = 2

# Regenerate PCA results from cleaned data
pca_results = build_and_save_pca_results(
    X_scaled,
    dims=(2, 5, 10),
    output_path="pca_results.pkl"
)

# Run KMeans on full data
results = []
pca_outputs = {}
cluster_sizes_by_dataset = {}

log("=== FULL 28D ===")
full_labels, full_model, full_metrics, full_cluster_sizes = run_kmeans(
    X_scaled,
    "Full 28D",
    k=k,
    true_labels=true_labels,
    explained_variance=1.0,
)
results.append(full_metrics)
cluster_sizes_by_dataset["Full 28D"] = full_cluster_sizes

# Run KMeans on regenerated PCA data
log("=== REGENERATED PCA DATA ===")
for dim in [2, 5, 10]:
    log(f"Processing PCA {dim}D...")

    entry = pca_results[dim]
    X_pca = np.asarray(entry["data"], dtype=float)
    explained_variance = float(np.sum(entry["explained_variance_ratio"]))

    log(f"  Extracted shape: {X_pca.shape}")
    log(f"  Explained variance: {explained_variance:.6f}")
    log(f"  PCA row count = {X_pca.shape[0]}")
    log(f"  Current dataset row count = {len(true_labels)}")

    labels, model, metrics, cluster_sizes = run_kmeans(
        X_pca,
        f"PCA {dim}D",
        k=k,
        true_labels=true_labels,
        explained_variance=explained_variance,
    )
    results.append(metrics)
    pca_outputs[dim] = {"data": X_pca, "labels": labels}
    cluster_sizes_by_dataset[f"PCA {dim}D"] = cluster_sizes

# Save comparison table
log("Saving clustering comparison...")
results_df = pd.DataFrame(results)
print("\nClustering Comparison:\n")
print(results_df.to_string(index=False))
results_df.to_csv("clustering_comparison.csv", index=False)
log("Saved clustering_comparison.csv")

# Save cluster size details
log("Saving cluster size details...")
cluster_size_rows = []
for dataset_name, sizes in cluster_sizes_by_dataset.items():
    row = {"Dataset": dataset_name}
    for idx, value in enumerate(sizes):
        row[f"Cluster_{idx}_Size"] = int(value)
    cluster_size_rows.append(row)

cluster_sizes_df = pd.DataFrame(cluster_size_rows)
cluster_sizes_df.to_csv("cluster_size_details.csv", index=False)
log("Saved cluster_size_details.csv")

# Metric plots
log("Generating metric plots...")
save_all_metric_graphs(results_df)
save_cluster_size_chart(cluster_sizes_by_dataset)

# Scatter visualizations
log("Generating scatter visualizations...")

log("Creating full 28D -> 2D PCA projection...")
pca_vis = PCA(n_components=2, random_state=42)
X_vis = pca_vis.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=full_labels, s=8, alpha=0.6)
plt.title("Full 28D (visualized in 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("full_28d.png")
plt.close()
log("Saved full_28d.png")

for dim in [2, 5, 10]:
    if dim not in pca_outputs:
        continue

    log(f"Plotting PCA {dim}D...")
    X_data = pca_outputs[dim]["data"]
    labels = pca_outputs[dim]["labels"]

    if dim == 2:
        X_plot = X_data
    else:
        reducer = PCA(n_components=2, random_state=42)
        X_plot = reducer.fit_transform(X_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, s=8, alpha=0.6)
    plt.title(f"PCA {dim}D Clustering")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(f"pca_{dim}d.png")
    plt.close()
    log(f"Saved pca_{dim}d.png")

log("Done.")