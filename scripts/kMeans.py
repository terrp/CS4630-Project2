import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


def evaluate(X_data, labels, model):
    log("  Computing evaluation metrics...")
    sil = silhouette_score(X_data, labels, sample_size=10000, random_state=42)
    db = davies_bouldin_score(X_data, labels)
    comp = compactness(X_data, labels, model.cluster_centers_)
    sep = separation(model.cluster_centers_)
    return {
        "Dataset": "",
        "Silhouette": sil,
        "Davies-Bouldin": db,
        "Compactness": comp,
        "Separation": sep,
    }


def run_kmeans(X_data, name, k=2):
    if not isinstance(X_data, np.ndarray):
        raise TypeError(f"{name}: expected numpy array before KMeans, got {type(X_data)}")

    log(f"Running K-Means on {name}...")
    log(f"  dtype={X_data.dtype}, shape={X_data.shape}")

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    start = time.time()
    labels = model.fit_predict(X_data)
    log(f"K-Means complete ({name}) in {time.time() - start:.2f}s")

    metrics = evaluate(X_data, labels, model)
    metrics["Dataset"] = name
    return labels, model, metrics


def describe_object(obj, prefix="root", depth=0, max_depth=3):
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{prefix}: <max depth reached>")
        return

    if isinstance(obj, dict):
        print(f"{indent}{prefix}: dict with keys={list(obj.keys())}")
        for key, value in obj.items():
            describe_object(value, f"{prefix}[{repr(key)}]", depth + 1, max_depth)
    elif isinstance(obj, np.ndarray):
        print(f"{indent}{prefix}: ndarray shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, pd.DataFrame):
        print(f"{indent}{prefix}: DataFrame shape={obj.shape}")
    elif isinstance(obj, pd.Series):
        print(f"{indent}{prefix}: Series shape={obj.shape}")
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{prefix}: {type(obj).__name__} len={len(obj)}")
        for i, value in enumerate(obj[:3]):
            describe_object(value, f"{prefix}[{i}]", depth + 1, max_depth)
    else:
        print(f"{indent}{prefix}: {type(obj)}")


def recursive_find_array(obj, path="root"):
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
            return obj, path
        return None, None

    if isinstance(obj, pd.DataFrame):
        arr = obj.to_numpy()
        if arr.ndim == 2:
            return arr, path

    if isinstance(obj, list):
        try:
            arr = np.asarray(obj, dtype=float)
            if arr.ndim == 2:
                return arr, path
        except Exception:
            pass

        for i, value in enumerate(obj):
            found, found_path = recursive_find_array(value, f"{path}[{i}]")
            if found is not None:
                return found, found_path

    if isinstance(obj, tuple):
        try:
            arr = np.asarray(obj, dtype=float)
            if arr.ndim == 2:
                return arr, path
        except Exception:
            pass

        for i, value in enumerate(obj):
            found, found_path = recursive_find_array(value, f"{path}[{i}]")
            if found is not None:
                return found, found_path

    if isinstance(obj, dict):
        preferred_keys = [
            "data", "X_pca", "reduced_data", "transformed_data",
            "X_reduced", "components", "embedding", "scores"
        ]

        for key in preferred_keys:
            if key in obj:
                found, found_path = recursive_find_array(obj[key], f"{path}[{repr(key)}]")
                if found is not None:
                    return found, found_path

        for key, value in obj.items():
            found, found_path = recursive_find_array(value, f"{path}[{repr(key)}]")
            if found is not None:
                return found, found_path

    return None, None


def get_pca_entry(pca_results, dim):
    candidate_keys = [dim, str(dim), f"{dim}D", f"pca_{dim}", f"PCA_{dim}", f"pca_{dim}d", f"PCA {dim}D"]
    for key in candidate_keys:
        if key in pca_results:
            return pca_results[key], key
    return None, None


log("Loading dataset...")
df = pd.read_csv("data/processed/higgs_200k.csv")
log(f"Dataset loaded with shape: {df.shape}")

feature_cols = [f"feature_{i}" for i in range(1, 29)]
X = df[feature_cols].copy()

log("Scaling full 28D dataset...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
log("Scaling complete.")

k = 2

log("Loading PCA results from pca_results.pkl...")
with open("pca_results.pkl", "rb") as f:
    pca_results = pickle.load(f)

log(f"Loaded PCA object type: {type(pca_results)}")

if not isinstance(pca_results, dict):
    raise TypeError(f"Expected pca_results to be a dict, got {type(pca_results)}")

log("Dumping PCA structure:")
describe_object(pca_results, max_depth=2)

results = []
pca_outputs = {}

log("=== FULL 28D ===")
full_labels, full_model, full_metrics = run_kmeans(X_scaled, "Full 28D", k=k)
results.append(full_metrics)

log("=== SAVED PCA DATA ===")
for dim in [2, 5, 10]:
    log(f"Processing PCA {dim}D...")

    entry, actual_key = get_pca_entry(pca_results, dim)
    if entry is None:
        log(f"  No matching key found for PCA {dim}D")
        continue

    log(f"  Found top-level key: {repr(actual_key)}")
    log(f"  Entry type: {type(entry)}")

    found_array, found_path = recursive_find_array(entry, path=f"pca_results[{repr(actual_key)}]")
    if found_array is None:
        log(f"  Could not locate a numeric 2D array for PCA {dim}D")
        describe_object(entry, prefix=f"pca_results[{repr(actual_key)}]", max_depth=3)
        continue

    X_pca = np.asarray(found_array, dtype=float)
    log(f"  Using array from: {found_path}")
    log(f"  Extracted shape: {X_pca.shape}")

    if X_pca.ndim != 2:
        raise ValueError(f"PCA {dim}D array is not 2D: shape={X_pca.shape}")

    if X_pca.shape[1] != dim:
        log(f"  Warning: expected {dim} columns, found {X_pca.shape[1]}")

    labels, model, metrics = run_kmeans(X_pca, f"PCA {dim}D", k=k)
    results.append(metrics)
    pca_outputs[dim] = {"data": X_pca, "labels": labels}

log("Saving clustering comparison...")
results_df = pd.DataFrame(results)
print("\nClustering Comparison:\n")
print(results_df.to_string(index=False))
results_df.to_csv("clustering_comparison.csv", index=False)
log("Saved clustering_comparison.csv")

log("Generating visualizations...")

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