"""
Gaussian Mixture Model clustering on FBLN expression.
Compares natural data clusters to rule-based categories via Adjusted Rand Index.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from data_loader import load_metabric, TARGET_GENES, CATEGORY_COLOURS, CATEGORY_ORDER

warnings.filterwarnings('ignore')


df = load_metabric()
cats = pd.read_parquet('data/processed/metabric_categories.parquet')
df = df.merge(cats, on='PATIENT_ID', how='left')
X  = df[TARGET_GENES].values


# Find the best number of components via BIC and AIC (Bayesian and Akaike Information Criterion) 
# the number of components where BIC/ AIC stops falling sharply is
# the natural cluster count the data prefers.

print("Fitting GMMs for n_components 2–8 (BIC selection):")
aic_scores = []
bic_scores = []
n_range    = range(2, 9)

for n in n_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full',
                           random_state=42, n_init=5)
    gmm.fit(X)
    aic_scores.append(gmm.aic(X))
    bic_scores.append(gmm.bic(X))
    print(f"  n={n}  BIC={gmm.bic(X):.1f}")
    print(f"  n={n}  AIC={gmm.aic(X):.1f}")

best_n_bic = n_range[int(np.argmin(bic_scores))]
best_n_aic = n_range[int(np.argmin(aic_scores))]
print(f"\nBIC selects n={best_n_bic}, AIC selects n={best_n_aic}")

best_n = best_n_aic
print(f"Using n={best_n} (AIC) for final model")


# Fit final GMM

gmm_final = GaussianMixture(n_components=best_n, covariance_type='full',
                              random_state=42, n_init=10)
gmm_final.fit(X)
df['gmm_label'] = gmm_final.predict(X)


#ARI: compare GMM clusters to rule-based categories

le = LabelEncoder()
rule_encoded = le.fit_transform(df['category'])
ari = adjusted_rand_score(rule_encoded, df['gmm_label'])
print(f"\nAdjusted Rand Index (rule-based vs GMM): {ari:.3f}")
print("  >0.6  = strong agreement — rules reflect real data structure")
print("  0.3–0.6 = moderate agreement")
print("  <0.3  = weak agreement — rules may be imposing arbitrary boundaries")


# Cross-tabulation: which GMM cluster maps to which category ─

print("\nGMM cluster vs rule-based category cross-tabulation:")
cross = pd.crosstab(df['gmm_label'], df['category'],
                    margins=True, margins_name='Total')
print(cross.to_string())


# Cluster-level survival check
# If GMM clusters are clinically meaningful they should show
# different survival distributions independent of our rules.

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

print("\nGMM cluster survival check (pairwise log-rank vs cluster 0):")
ref = df[df['gmm_label'] == 0]
for cluster in range(1, best_n):
    grp = df[df['gmm_label'] == cluster]
    if len(grp) < 10:
        continue
    result = logrank_test(
        ref['os_months'], grp['os_months'],
        event_observed_A=ref['os_event'],
        event_observed_B=grp['os_event']
    )
    sig = '✓' if result.p_value < 0.05 else '✗'
    print(f"  Cluster 0 vs {cluster}: p={result.p_value:.4f} {sig}")


# Figure: BIC curve
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('GMM Clustering — FBLN Expression Space', fontsize=13, fontweight='bold')

axes[0].plot(list(n_range), bic_scores, marker='o', color='#2c3e50', linewidth=2)
axes[0].axvline(best_n, color='#e74c3c', linestyle='--', linewidth=1.5,
                label=f'Best n={best_n}')
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('BIC')
axes[0].set_title('BIC by Component Count')
axes[0].legend()

# Scatter: FBLN1 vs FBLN2, coloured by GMM cluster
scatter_colors = plt.cm.tab10(np.linspace(0, 0.8, best_n))
for cluster in range(best_n):
    mask = df['gmm_label'] == cluster
    axes[1].scatter(df.loc[mask, 'FBLN1'], df.loc[mask, 'FBLN2'],
                    c=[scatter_colors[cluster]], alpha=0.3, s=10,
                    label=f'Cluster {cluster}')

axes[1].set_xlabel('FBLN1 Expression')
axes[1].set_ylabel('FBLN2 Expression')
axes[1].set_title(f'GMM Clusters (n={best_n}) — FBLN1 vs FBLN2')
axes[1].legend(markerscale=2, fontsize=8)

plt.tight_layout()
plt.savefig('outputs/gmm_clusters.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nGMM figures saved to outputs/gmm_clusters.png")

# Save cluster assignments for use in survival_tree.py
df[['PATIENT_ID', 'category', 'gmm_label']].to_parquet(
    'data/processed/gmm_labels.parquet', index=False
)
print("GMM labels saved to data/processed/gmm_labels.parquet")

# GMM cluster characterisation

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("\nMean expression per GMM cluster:")
print(df.groupby('gmm_label')[TARGET_GENES].mean().round(3))

print("\nSubtype distribution per GMM cluster (row %)")
print(pd.crosstab(df['gmm_label'], df['subtype'], normalize='index').round(3))

print("\nCategory distribution per GMM cluster (row %):")
print(pd.crosstab(df['gmm_label'], df['category'], normalize='index').round(3))

print("\nSurvival summary per GMM cluster:")
for cluster in sorted(df['gmm_label'].unique()):
    sub             = df[df['gmm_label'] == cluster]
    event_rate      = sub['os_event'].mean()
    median_survival = sub.loc[sub['os_event'] == 1, 'os_months'].median()
    print(f"  Cluster {cluster} (n={len(sub)}): "
          f"event rate={event_rate:.1%}, "
          f"median survival={median_survival:.0f} months")