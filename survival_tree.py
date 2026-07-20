"""
Survival-supervised tree derived from outcome data.
Includes subtype as a feature to capture context-dependent survival signal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from data_loader import load_metabric, TARGET_GENES, CATEGORY_ORDER
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings('ignore')

from sksurv.tree import SurvivalTree
from sksurv.util import Surv

# Load data and category labels
df   = load_metabric()
cats = pd.read_parquet('data/processed/metabric_categories.parquet')
df   = df.merge(cats, on='PATIENT_ID', how='left')

# Encode subtype
le_sub = LabelEncoder()
df['subtype_encoded'] = le_sub.fit_transform(df['subtype'].fillna('Unknown'))

FEATURES = TARGET_GENES + ['subtype_encoded']
X      = df[FEATURES].to_numpy().astype(np.float32)
y_surv = Surv.from_dataframe('os_event', 'os_months', df)

tree = SurvivalTree(
    max_leaf_nodes=5,
    min_samples_leaf=50,
    random_state=42
)
tree.fit(X, y_surv)
df['tree_leaf'] = tree.apply(X)

leaves = sorted(df['tree_leaf'].unique())
print(f"SurvivalTree found {len(leaves)} terminal leaves")
print(df['tree_leaf'].value_counts().sort_index().to_string())

# Pairwise log-rank vs first leaf
print("\nLeaf survival check:")
ref_leaf = leaves[0]
ref = df[df['tree_leaf'] == ref_leaf]
for leaf in leaves[1:]:
    grp    = df[df['tree_leaf'] == leaf]
    result = logrank_test(
        ref['os_months'], grp['os_months'],
        event_observed_A=ref['os_event'],
        event_observed_B=grp['os_event']
    )
    sig = 'significant' if result.p_value < 0.05 else 'insignificant'
    print(f"  Leaf {ref_leaf} vs {leaf} (n={len(grp)}): p={result.p_value:.4f} {sig}")

# Cross-tabulation vs rule-based categories
print("\nTree leaf vs rule-based category:")
cross = pd.crosstab(df['tree_leaf'], df['category'],
                    margins=True, margins_name='Total')
print(cross.to_string())

# Subtype distribution per leaf — key diagnostic given subtype is now a feature
print("\nSubtype distribution per leaf:")
print(pd.crosstab(df['tree_leaf'], df['subtype'], normalize='index').round(3).to_string())

# ARI
le           = LabelEncoder()
rule_encoded = le.fit_transform(df['category'])
ari          = adjusted_rand_score(rule_encoded, df['tree_leaf'])
print(f"\nAdjusted Rand Index (rule-based vs SurvivalTree): {ari:.3f}")

# KM figure
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Kaplan-Meier — SurvivalTree Leaves (expression + subtype)',
             fontsize=13, fontweight='bold')

leaf_colors = plt.cm.tab10(np.linspace(0, 0.8, len(leaves)))
kmf = KaplanMeierFitter()
for leaf, color in zip(leaves, leaf_colors):
    subset = df[df['tree_leaf'] == leaf]
    kmf.fit(subset['os_months'], subset['os_event'],
            label=f'Leaf {leaf} (n={len(subset)})')
    kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=2)

ax.set_xlabel('Time (months)')
ax.set_ylabel('Survival Probability')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_xlim(0, 200)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('outputs/survival_tree_km.png', dpi=150, bbox_inches='tight')
plt.close()

# Save
df[['PATIENT_ID', 'category', 'tree_leaf']].to_parquet(
    'data/processed/survival_tree_labels.parquet', index=False
)