"""
fbln_pipeline.py
Primary METABRIC analysis: uncertainty categorisation, Random Forest,
SHAP explainability, calibration, and survival validation of categories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from data_loader import (
    load_metabric, TARGET_GENES,
    CATEGORY_COLOURS, CATEGORY_ORDER, BOUNDARY_MULTIPLIER
)

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})


# Load metabric
df = load_metabric()
N  = len(df)

# Compute per-gene statistics used by categorisation
stats = {
    gene: {'med': df[gene].median(), 'std': df[gene].std()}
    for gene in TARGET_GENES
}


# Categorisation id by uncertainty group using statistics

def assign_category(row):
    f1 = row['FBLN1']
    f2 = row['FBLN2']
    f5 = row['FBLN5']
    subtype = row.get('subtype', None)

    med1, std1 = stats['FBLN1']['med'], stats['FBLN1']['std']
    med2, std2 = stats['FBLN2']['med'], stats['FBLN2']['std']
    med5, std5 = stats['FBLN5']['med'], stats['FBLN5']['std']

    boundary = BOUNDARY_MULTIPLIER

    # Out of scope if 3.5 standard deviations away from the median
    if abs(f1 - med1) > 3.5 * std1 or abs(f5 - med5) > 3.5 * std5:
        return 'OUT_OF_SCOPE'

    # Insufficient data if in the 0.05th quantile
    if f1 < df['FBLN1'].quantile(0.05) and f5 < df['FBLN5'].quantile(0.05):
        return 'DATA_INSUFFICIENCY'

    # Ambiguous if 0.5 away stds away from the median
    near_median = (
        abs(f1 - med1) < boundary * std1 and
        abs(f2 - med2) < boundary * std2 and
        abs(f5 - med5) < boundary * std5
    )
    if near_median:
        return 'AMBIGUITY'

    # Controversy if ambiguous signal from fbln 2 and poor signal from fbln1 and fbln 5
    poor_signals = sum([f1 > med1, f5 > med5])
    if f2 > med2 and poor_signals >= 1:
        return 'CONTROVERSY'
    aggressive = {'Basal', 'Her2', 'LumB'}
    if subtype in aggressive:
        return 'HIGH_CONFIDENCE_UNFAVOURABLE'
    return 'HIGH_CONFIDENCE_FAVOURABLE'


df['category'] = df.apply(assign_category, axis=1)
df[['PATIENT_ID', 'category']].to_parquet(
    'data/processed/metabric_categories.parquet', index=False
)

cat_counts = df['category'].value_counts()
print("Category distribution:")
for cat in CATEGORY_ORDER:
    n = cat_counts.get(cat, 0)
    print(f"  {cat:<22} {n:>4}  ({n/N*100:.1f}%)")

non_hc = N - cat_counts.get('HIGH_CONFIDENCE_FAVOURABLE', 0) - cat_counts.get('HIGH_CONFIDENCE_UNFAVOURABLE', 0)
print(f"\n  {non_hc} patients ({non_hc/N*100:.1f}%) outside HIGH_CONFIDENCE")

# Triangulation statistics - checks how many patients have a single gene or all three genes near the
# median (indicator that the signal isn't strong enough to diagnose prognosis)
near_any = sum([
    (abs(df[g] - stats[g]['med']) < BOUNDARY_MULTIPLIER * stats[g]['std']).any()
    for g in TARGET_GENES
])
near_any_pct = (
    (abs(df['FBLN1'] - stats['FBLN1']['med']) < BOUNDARY_MULTIPLIER * stats['FBLN1']['std']) |
    (abs(df['FBLN2'] - stats['FBLN2']['med']) < BOUNDARY_MULTIPLIER * stats['FBLN2']['std']) |
    (abs(df['FBLN5'] - stats['FBLN5']['med']) < BOUNDARY_MULTIPLIER * stats['FBLN5']['std'])
).mean() * 100

near_all_pct = (
    (abs(df['FBLN1'] - stats['FBLN1']['med']) < BOUNDARY_MULTIPLIER * stats['FBLN1']['std']) &
    (abs(df['FBLN2'] - stats['FBLN2']['med']) < BOUNDARY_MULTIPLIER * stats['FBLN2']['std']) &
    (abs(df['FBLN5'] - stats['FBLN5']['med']) < BOUNDARY_MULTIPLIER * stats['FBLN5']['std'])
).mean() * 100

print(f"\nTriangulation:")
print(f"  Near median (≥1 gene): {near_any_pct:.1f}%")
print(f"  Near median (all 3):   {near_all_pct:.1f}%")


# Figure 1: Category distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Uncertainty Category Distribution — METABRIC (FBLN1, FBLN2, FBLN5)',
             fontsize=13, fontweight='bold')

cats   = [c for c in CATEGORY_ORDER if c in cat_counts.index]
counts = [cat_counts[c] for c in cats]
colors = [CATEGORY_COLOURS[c] for c in cats]

bars = axes[0].bar(cats, counts, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
axes[0].set_xlabel('Uncertainty Category')
axes[0].set_ylabel('Number of Patients')
axes[0].set_title('Patient Count per Category')
axes[0].tick_params(axis='x', rotation=25)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 12,
                 f'{count}\n({count/N*100:.1f}%)', ha='center', fontsize=9, fontweight='bold')

axes[1].pie(counts, labels=cats, colors=colors, autopct='%1.1f%%',
            startangle=140, pctdistance=0.75,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Proportion per Category')

plt.tight_layout()
plt.savefig('outputs/fig1_category_distribution.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 2: Expression profiles by category

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Biomarker Expression Profiles by Uncertainty Category', fontsize=13, fontweight='bold')

for ax, gene in zip(axes, TARGET_GENES):
    sns.boxplot(data=df, x='category', y=gene, order=CATEGORY_ORDER,
                palette=CATEGORY_COLOURS, ax=ax, width=0.55, linewidth=1.2,
                flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})
    ax.set_title(f'{gene} Expression')
    ax.set_xlabel('')
    ax.set_ylabel('Expression (log2)')
    ax.tick_params(axis='x', rotation=30)
    ax.axhline(df[gene].median(), color='navy', linestyle='--', alpha=0.4, linewidth=1, label='Median')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('outputs/fig2_expression_by_category.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 3: Kaplan-Meier + between-category log-rank

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Kaplan-Meier Survival by Uncertainty Category', fontsize=13, fontweight='bold')

kmf = KaplanMeierFitter()
for cat in CATEGORY_ORDER:
    subset = df[df['category'] == cat]
    if len(subset) < 10:
        continue
    kmf.fit(subset['os_months'], subset['os_event'], label=cat)
    kmf.plot_survival_function(ax=ax, ci_show=True,
                               color=CATEGORY_COLOURS[cat], linewidth=2.5, ci_alpha=0.08)

ax.set_xlabel('Time (months)')
ax.set_ylabel('Survival Probability')
ax.legend(title='Category', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_xlim(0, 200)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('outputs/fig3_kaplan_meier.png', dpi=150, bbox_inches='tight')
plt.close()

# Pairwise log-rank tests between categories
print("\nBetween-category log-rank tests (vs HIGH_CONFIDENCE_FAVOURABLE):")
ref = df[df['category'] == 'HIGH_CONFIDENCE_FAVOURABLE']
for cat in ['HIGH_CONFIDENCE_UNFAVOURABLE', 'AMBIGUITY', 'CONTROVERSY', 
            'DATA_INSUFFICIENCY', 'OUT_OF_SCOPE']:
    grp = df[df['category'] == cat]
    if len(grp) < 10:
        continue
    result = logrank_test(
        ref['os_months'], grp['os_months'],
        event_observed_A=ref['os_event'],
        event_observed_B=grp['os_event']
    )
    sig = 'significant' if result.p_value < 0.05 else 'insignificant'
    print(f"  HIGH_CONFIDENCE vs {cat:<22} p={result.p_value:.4f} {sig}")


# Figure 4: Correlation heatmap + controversy scatter

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('FBLN Gene Inter-Activity', fontsize=13, fontweight='bold')

corr = df[TARGET_GENES].corr()
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            ax=axes[0], vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_title('Pearson Correlation')

for cat in CATEGORY_ORDER:
    mask = df['category'] == cat
    axes[1].scatter(df.loc[mask, 'FBLN1'], df.loc[mask, 'FBLN2'],
                    c=CATEGORY_COLOURS[cat], alpha=0.35, s=12, label=cat)

axes[1].axvline(stats['FBLN1']['med'], color='navy', linestyle='--', alpha=0.5, linewidth=1)
axes[1].axhline(stats['FBLN2']['med'], color='darkred', linestyle='--', alpha=0.5, linewidth=1)
axes[1].set_xlabel('FBLN1 Expression')
axes[1].set_ylabel('FBLN2 Expression')
axes[1].set_title('FBLN1 vs FBLN2 — Controversy Space')
axes[1].legend(markerscale=2, fontsize=8, title='Category')

plt.tight_layout()
plt.savefig('outputs/fig4_gene_interactivity.png', dpi=150, bbox_inches='tight')
plt.close()


# Proof of Concept Random Forest 

le = LabelEncoder()
X  = df[TARGET_GENES].values
y  = le.fit_transform(df['category'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

cv_scores = cross_val_score(rf, X, y, cv=5, scoring='f1_macro')
y_pred    = rf.predict(X_test)

print(f"\nRandom Forest — 5-fold CV Macro F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))


# Figure 5: Confusion matrix

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix — Random Forest (rule-based labels)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('outputs/fig5_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 6: SHAP 

explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test[:200])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('SHAP Feature Attributions', fontsize=13, fontweight='bold')

mean_shap = np.array([np.abs(shap_values[:, :, i]).mean(axis=0)
                      for i in range(len(le.classes_))])
shap_df = pd.DataFrame(mean_shap, index=le.classes_, columns=TARGET_GENES)
shap_df.T.plot(kind='bar', ax=axes[0],
               color=[CATEGORY_COLOURS[c] for c in le.classes_],
               edgecolor='white', width=0.7)
axes[0].set_xlabel('Gene')
axes[0].set_ylabel('Mean |SHAP|')
axes[0].set_title('Mean Feature Importance per Category')
axes[0].legend(title='Category', fontsize=8, bbox_to_anchor=(1, 1))
axes[0].tick_params(axis='x', rotation=0)

example_cats = ['HIGH_CONFIDENCE_FAVOURABLE', 'HIGH_CONFIDENCE_UNFAVOURABLE',
                  'CONTROVERSY', 'AMBIGUITY']
y_test_labels  = le.inverse_transform(y_test)
example_colors = ['#2ecc71', '#e67e22', '#e74c3c', '#f39c12']
gene_colors    = ['#3498db', '#e67e22', '#1abc9c']
bar_height     = 0.2
y_positions    = [0.78, 0.54, 0.30, 0.06]

ax2 = axes[1]
ax2.set_title('SHAP Contributions — Example Patients')

for i, (cat, ypos, col) in enumerate(zip(example_cats, y_positions, example_colors)):
    idx_list = np.where(y_test_labels == cat)[0]
    if len(idx_list) == 0:
        continue
    idx     = idx_list[0]
    cat_idx = list(le.classes_).index(cat)
    sv      = shap_values[idx, :, cat_idx]
    left    = 0
    for gene, val, gcol in zip(TARGET_GENES, sv, gene_colors):
        ax2.barh(ypos, val, height=bar_height, left=left, color=gcol,
                 edgecolor='white', linewidth=0.8,
                 label=gene if i == 0 else '')
        if abs(val) > 0.01:
            ax2.text(left + val/2, ypos, f'{gene}\n{val:+.2f}',
                     ha='center', va='center', fontsize=7,
                     fontweight='bold', color='white')
        left += val
    display_name = cat.replace('HIGH_CONFIDENCE_', 'HC_')
    ax2.text(-0.35, ypos, display_name, ha='right', va='center',
             fontsize=8, fontweight='bold', color=col)

ax2.axvline(0, color='black', linewidth=1)
ax2.set_xlabel('SHAP Contribution')
ax2.set_xlim(-0.5, 0.5)
ax2.set_yticks([])
ax2.legend(title='Gene', loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/fig6_shap.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 7: Calibration

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5)

proba = rf.predict_proba(X_test)
for i, cat in enumerate(le.classes_):
    binary_y = (y_test == i).astype(int)
    if binary_y.sum() < 5:
        continue
    frac_pos, mean_pred = calibration_curve(binary_y, proba[:, i], n_bins=8)
    ax.plot(mean_pred, frac_pos, marker='o', linewidth=2,
            color=CATEGORY_COLOURS[cat], label=cat, markersize=5)

ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves per Category')
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('outputs/fig7_calibration.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll figures saved to outputs/")