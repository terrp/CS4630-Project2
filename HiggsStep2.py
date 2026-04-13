import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_theme(style='whitegrid', palette='muted')
RANDOM_STATE = 42

# ── 0. Load Subsample & Scale (Acting as Member 1 temporarily) ──────────────
print("Loading 200k subsample from HIGGS.csv.gz...")
# HIGGS has no headers. Col 0 is the label, Cols 1-28 are features.
df = pd.read_csv('HIGGS.csv.gz', header=None, nrows=200000)

y = df.iloc[:, 0].values
X_raw = df.iloc[:, 1:].values

print(f"Applying StandardScaler to X shape {X_raw.shape}...")
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── 1a. Fit PCA on full 28 dimensions ───────────────────────────────────────
print("Running full 28-dimensional PCA...")

t0 = time.perf_counter()  # <--- Updated to high-resolution timer
pca_full = PCA(n_components=28, random_state=RANDOM_STATE)
pca_full.fit(X)
elapsed_full = time.perf_counter() - t0  # <--- Updated calculation

print(f"Full PCA fit in {elapsed_full:.3f}s")  # <--- Updated to 3 decimal places

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cumvar, 0.95) + 1
print(f"** Components required to explain 95% variance: {n_95} **")

# ── 1b. Scree plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(1, 29), pca_full.explained_variance_ratio_, alpha=0.6, label='Individual')
ax.plot(range(1, 29), cumvar, marker='o', ms=4, color='tomato', label='Cumulative')
ax.axhline(0.95, color='grey', linestyle='--', linewidth=0.8, label='95% threshold')
ax.axvline(n_95, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel('Principal component')
ax.set_ylabel('Explained variance ratio')
ax.set_title('Scree plot — HIGGS 28 features')
ax.legend()
plt.tight_layout()
plt.savefig('pca_scree.png', dpi=150)
print("Saved Scree plot to 'pca_scree.png'")

# ── 1c. Produce reduced datasets at 2, 5, 10 components ─────────────────────
pca_results = {}
for n in [2, 5, 10]:
    t0 = time.perf_counter()  # <--- Updated to high-resolution timer
    pca_n = PCA(n_components=n, random_state=RANDOM_STATE)
    X_pca = pca_n.fit_transform(X)
    elapsed = time.perf_counter() - t0  # <--- Updated calculation

    # Store the transformed data, the PCA object, and the runtime
    pca_results[n] = {'X': X_pca, 'pca': pca_n, 'time': elapsed}

    var_explained = pca_n.explained_variance_ratio_.sum()
    print(
        f"PCA-{n:>2}:  {var_explained * 100:.1f}% variance  |  runtime: {elapsed:.3f}s")  # <--- Updated to 3 decimal places

# ── 1d. Save Deliverable for Member 3 ───────────────────────────────────────
with open('pca_results.pkl', 'wb') as f:
    pickle.dump(pca_results, f)
print("Exported reduced datasets to 'pca_results.pkl' for Step 3.")