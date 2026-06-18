"""
Multi-Class Uncertainty Categorisation for Breast Cancer Prognosis
===================================================================
Exploratory Analysis: FBLN1, FBLN2, FBLN5 Biomarker Signals
Dataset: METABRIC

Author: Joel Allen-Caliste
MSc Artificial Intelligence, University of Surrey
April 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
import os
import shap
import warnings

warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# COLOUR PALETTE & STYLE
CATEGORY_COLOURS = {
    'HIGH_CONFIDENCE': '#2ecc71',
    'AMBIGUITY':       '#f39c12',
    'CONTROVERSY':     '#e74c3c',
    'DATA_INSUFFICIENCY': '#9b59b6',
    'OUT_OF_SCOPE':    '#95a5a6',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

# Data Loading

TARGET_GENES = ['FBLN1', 'FBLN2', 'FBLN5']

# Load mRNA expression matrix

mrna = pd.read_csv('brca_metabric/data_mrna_illumina_microarray.txt', sep='\t', index_col=0, low_memory=False)
# Drop the Entrez ID column if present 
if 'Entrez_Gene_Id' in mrna.columns:
    mrna = mrna.drop(columns=['Entrez_Gene_Id'])

# Filter to FBLN genes, then transpose: rows=patients, cols=genes
fbln_expr = mrna.loc[mrna.index.isin(TARGET_GENES)].T
fbln_expr.index.name = 'PATIENT_ID'
fbln_expr = fbln_expr.reset_index()
fbln_expr = fbln_expr.rename(columns={
    'FBLN1': 'FBLN1',
    'FBLN2': 'FBLN2',
    'FBLN5': 'FBLN5'
})

# Confirm all three genes loaded
missing = [g for g in TARGET_GENES if g not in fbln_expr.columns]
if missing:
    raise ValueError(f"Genes not found in expression matrix: {missing}.")

# Load clinical data
clin = pd.read_csv('brca_metabric/data_clinical_patient.txt', sep='\t', comment='#', low_memory=False)
clin.columns = clin.columns.str.upper()

# Identify the correct survival columns 
OS_MONTHS_COL = 'OS_MONTHS'     
OS_STATUS_COL = 'OS_STATUS'      
SUBTYPE_COL   = 'CLAUDIN_SUBTYPE'

# Convert status to binary int
clin['death_event'] = clin[OS_STATUS_COL].astype(str).str.startswith('1').astype(int)
clin = clin.rename(columns={
    'PATIENT_ID': 'PATIENT_ID',
    OS_MONTHS_COL: 'survival_months',
    SUBTYPE_COL: 'subtype'
})

# Merge expression + clinical
df = fbln_expr.reset_index().merge(
    clin[['PATIENT_ID', 'survival_months', 'death_event', 'subtype']],
    on='PATIENT_ID',
    how='inner'
)

# Drop any patients with missing values in key columns
df = df.dropna(subset=['FBLN1', 'FBLN2', 'FBLN5', 'survival_months', 'death_event'])
df = df.reset_index(drop=True)
N = len(df)

print(f"Dataset loaded: {N} patients with  FBLN1/2/4 expression and survival data")
print(f"Columns available: {list(df.columns)}")

# Seperate patients by uncertainty

FBLN1_MED = df['FBLN1'].median()
FBLN2_MED = df['FBLN2'].median()
FBLN5_MED = df['FBLN5'].median()

FBLN1_STD = df['FBLN1'].std()
FBLN2_STD = df['FBLN2'].std()
FBLN5_STD = df['FBLN5'].std()

# Finding boundary - within half a standard deviation of the median
BOUNDARY_FBLN1 = 0.5 * FBLN1_STD
BOUNDARY_FBLN2 = 0.5 * FBLN2_STD
BOUNDARY_FBLN5 = 0.5 * FBLN5_STD

def assign_category(row):
    f1, f2, f5 = row['FBLN1'], row['FBLN2'], row['FBLN5']

    near_boundary = (
        abs(f1 - FBLN1_MED) < BOUNDARY_FBLN1 and
        abs(f2 - FBLN2_MED) < BOUNDARY_FBLN2 and
        abs(f5 - FBLN5_MED) < BOUNDARY_FBLN5
    )

    f1_poor = f1 > FBLN1_MED
    f2_good = f2 > FBLN2_MED
    f5_poor = f5 > FBLN5_MED

    poor_signals = sum([f1_poor, f5_poor])

    if (abs(f1 - FBLN1_MED) > 3.5 * FBLN1_STD or 
        abs(f5 - FBLN5_MED) > 3.5 * FBLN5_STD):
        return 'OUT_OF_SCOPE'

    if f1 < df['FBLN1'].quantile(0.05) and f5 < df['FBLN5'].quantile(0.05):
        return 'DATA_INSUFFICIENCY'

    if near_boundary:
        return 'AMBIGUITY'

    if f2_good and poor_signals >= 1:
        return 'CONTROVERSY'

    return 'HIGH_CONFIDENCE'

df['category'] = df.apply(assign_category, axis=1)

# Split patients by category based on uncertainty boundaries
print("  UNCERTAINTY CATEGORY DISTRIBUTION")
cat_counts = df['category'].value_counts()
for cat, count in cat_counts.items():
    pct = count / N * 100
    print(f"  {cat:<22} {count:>4}  ({pct:.1f}%)")
print(f"\n  Total patients: {N}")
non_binary = N - cat_counts.get('HIGH_CONFIDENCE', 0)
print(f"  NOTE: {non_binary} patients ({non_binary/N*100:.1f}%) cannot be handled by binary classification")

# FIGURE 1: Category Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 1: Uncertainty Category Distribution\nMETABRIC Dataset — FBLN1, FBLN2, FBLN5', 
             fontsize=14, fontweight='bold', y=1.02)

# Bar chart
cats   = list(cat_counts.index)
counts = list(cat_counts.values)
colors = [CATEGORY_COLOURS[c] for c in cats]
bars = axes[0].bar(cats, counts, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
axes[0].set_xlabel('Uncertainty Category')
axes[0].set_ylabel('Number of Patients')
axes[0].set_title('Patient Count per Category')
axes[0].tick_params(axis='x', rotation=25)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 f'{count}\n({count/N*100:.1f}%)', ha='center', fontsize=9, fontweight='bold')

# Pie chart
axes[1].pie(counts, labels=cats, colors=colors, autopct='%1.1f%%',
            startangle=140, pctdistance=0.75,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Proportion of Patients per Category')

plt.tight_layout()
plt.savefig('outputs/fig1_category_distribution.png', dpi=150, bbox_inches='tight')
plt.close()


# FIGURE 2: FBLN2 Expression by Category (the clinical insight)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Figure 2: Biomarker Expression Profiles by Uncertainty Category\n(Key: FBLN2 protective signal creates CONTROVERSY)',
             fontsize=13, fontweight='bold', y=1.02)

cat_order = ['HIGH_CONFIDENCE', 'AMBIGUITY', 'CONTROVERSY', 'DATA_INSUFFICIENCY', 'OUT_OF_SCOPE']
pal = [CATEGORY_COLOURS[c] for c in cat_order]

for ax, gene in zip(axes, ['FBLN1', 'FBLN2', 'FBLN5']):
    sns.boxplot(data=df, x='category', y=gene, order=cat_order,
                palette=pal, ax=ax, width=0.55, linewidth=1.2,
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


# FIGURE 3: Survival by Category 
from lifelines import KaplanMeierFitter

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Figure 3: Kaplan-Meier Survival Curves by Uncertainty Category\n(Validates prognostic separation between categories)',
             fontsize=13, fontweight='bold')

kmf = KaplanMeierFitter()
for cat in cat_order:
    subset = df[df['category'] == cat]
    if len(subset) < 10:
        continue
    kmf.fit(subset['survival_months'], subset['death_event'], label=cat)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=CATEGORY_COLOURS[cat],
                               linewidth=2.5, ci_alpha=0.08)

ax.set_xlabel('Time (months)', fontsize=11)
ax.set_ylabel('Survival Probability', fontsize=11)
ax.set_title('')
ax.legend(title='Category', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_xlim(0, 200)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('outputs/fig3_kaplan_meier.png', dpi=150, bbox_inches='tight')
plt.close()


# FIGURE 4: Correlation heatmap — FBLN2 inter-activity
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Figure 4: FBLN2 Gene Inter-Activity\n(Motivates using all three together vs. any single gene)',
             fontsize=13, fontweight='bold', y=1.02)

corr = df[['FBLN1', 'FBLN2', 'FBLN5']].corr()
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            ax=axes[0], vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_title('Pearson Correlation Between FBLN2 Genes')

# Scatter: FBLN1 vs FBLN2 coloured by category — shows the controversy space
controversy_mask = df['category'] == 'CONTROVERSY'
for cat in cat_order:
    mask = df['category'] == cat
    axes[1].scatter(df.loc[mask, 'FBLN1'], df.loc[mask, 'FBLN2'],
                    c=CATEGORY_COLOURS[cat], alpha=0.35, s=12, label=cat)

axes[1].axvline(FBLN1_MED, color='navy', linestyle='--', alpha=0.5, linewidth=1)
axes[1].axhline(FBLN2_MED, color='darkred', linestyle='--', alpha=0.5, linewidth=1)
axes[1].set_xlabel('FBLN1 Expression')
axes[1].set_ylabel('FBLN2 Expression (tumour-protective)')
axes[1].set_title('FBLN1 vs FBLN2 — Controversy Space\n(Top-right quadrant: high FBLN1 poor signal + high FBLN2 protection)')
axes[1].legend(markerscale=2, fontsize=8, title='Category')

plt.tight_layout()
plt.savefig('outputs/fig4_fbln_interactivity.png', dpi=150, bbox_inches='tight')
plt.close()


# Naive Random Forest Model
print("  RANDOM FOREST CLASSIFIER — INITIAL RESULTS")

le = LabelEncoder()
X = df[['FBLN1', 'FBLN2', 'FBLN5']].values
y = le.fit_transform(df['category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

cv_scores = cross_val_score(rf, X, y, cv=5, scoring='f1_macro')
print(f"\n  5-Fold CV Macro F1:  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"\n  Classification Report (Test Set):\n")
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))


# FIGURE 5: Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, ax=ax)
ax.set_xlabel('Predicted Category', fontsize=11)
ax.set_ylabel('True Category', fontsize=11)
ax.set_title('Figure 5: Confusion Matrix — Random Forest\n(Rule-based labels as ground truth, to be replaced by oncologist labels)',
             fontsize=12, fontweight='bold')
plt.xticks(rotation=30)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/fig5_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()


# FIGURE 6: SHAP

explainer   = shap.TreeExplainer(rf)
# shap_values shape: (n_samples, n_features, n_classes)
shap_values = explainer.shap_values(X_test[:200])

gene_names = ['FBLN1', 'FBLN2', 'FBLN5']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 6: SHAP Explanations',
             fontsize=13, fontweight='bold', y=1.02)

# Mean absolute SHAP per gene per class
mean_shap = np.array([np.abs(shap_values[:, :, i]).mean(axis=0) for i in range(len(le.classes_))])
# mean_shap shape: (n_classes, n_features) = (5, 3)
shap_df = pd.DataFrame(mean_shap, index=le.classes_, columns=gene_names)

shap_df.T.plot(kind='bar', ax=axes[0], color=[CATEGORY_COLOURS[c] for c in le.classes_],
               edgecolor='white', width=0.7)
axes[0].set_xlabel('Gene')
axes[0].set_ylabel('Mean |SHAP value|')
axes[0].set_title('Mean Feature Importance per Category')
axes[0].legend(title='Category', fontsize=8, bbox_to_anchor=(1, 1))
axes[0].tick_params(axis='x', rotation=0)

example_cats = ['HIGH_CONFIDENCE', 'CONTROVERSY', 'AMBIGUITY']
y_test_labels = le.inverse_transform(y_test)
colours_eg = ['#2ecc71', '#e74c3c', '#f39c12']

ax2 = axes[1]
ax2.set_title('SHAP Contributions — 3 Example Patients\n(Shows which gene dominates each categorisation)')

bar_height = 0.25
y_positions = [0.7, 0.4, 0.1]

for i, (cat, ypos, col) in enumerate(zip(example_cats, y_positions, colours_eg)):
    idx_list = np.where(y_test_labels == cat)[0]
    if len(idx_list) == 0:
        continue
    idx     = idx_list[0]
    cat_idx = list(le.classes_).index(cat)
    sv      = shap_values[idx, :, cat_idx]   # shape: (n_features,) = (3,)

    left = 0
    for gene, val, gcol in zip(gene_names, sv, ['#3498db', '#e67e22', '#1abc9c']):
        ax2.barh(ypos, val, height=bar_height, left=left, color=gcol,
                 edgecolor='white', linewidth=0.8, label=gene if i == 0 else "")
        if abs(val) > 0.01:
            ax2.text(left + val/2, ypos, f'{gene}\n{val:+.2f}',
                     ha='center', va='center', fontsize=7.5, fontweight='bold', color='white')
        left += val

    ax2.text(-0.35, ypos, cat, ha='right', va='center', fontsize=9,
             fontweight='bold', color=col)

ax2.axvline(0, color='black', linewidth=1)
ax2.set_xlabel('SHAP Contribution (→ poor prognosis)')
ax2.set_xlim(-0.5, 0.5)
ax2.set_yticks([])
ax2.legend(title='Gene', loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/fig6_shap_explanations.png', dpi=150, bbox_inches='tight')
plt.close()


# FIGURE 7: Calibration — Trustworthy Uncertainty Estimates
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

ax.set_xlabel('Mean Predicted Probability', fontsize=11)
ax.set_ylabel('Fraction of Positives (Actual)', fontsize=11)
ax.set_title('Figure 7: Calibration Curves per Category\n(Closer to diagonal = more trustworthy uncertainty estimates)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('outputs/fig7_calibration.png', dpi=150, bbox_inches='tight')
plt.close()


# Log-rank tests + Cox hazard ratios per gene
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

print("  INDIVIDUAL BIOMARKER SURVIVAL ANALYSIS")

for gene in ['FBLN1', 'FBLN2', 'FBLN5']:
    median_val = df[gene].median()
    high = df[df[gene] >  median_val]
    low  = df[df[gene] <= median_val]

    result = logrank_test(
        high['survival_months'], low['survival_months'],
        event_observed_A=high['death_event'],
        event_observed_B=low['death_event']
    )

    # Cox model for hazard ratio
    cox_df = df[['survival_months', 'death_event', gene]].copy()
    cox_df['gene_high'] = (cox_df[gene] > median_val).astype(int)
    cph = CoxPHFitter()
    cph.fit(cox_df[['survival_months', 'death_event', 'gene_high']],
            duration_col='survival_months', event_col='death_event')

    summary = cph.summary
    hr       = summary.loc['gene_high', 'exp(coef)']
    ci_lower = summary.loc['gene_high', 'exp(coef) lower 95%']
    ci_upper = summary.loc['gene_high', 'exp(coef) upper 95%']
    p_cox    = summary.loc['gene_high', 'p']

    print(f"\n  {gene}")
    print(f"    Log-rank p-value : {result.p_value:.4f}  {'✓ SIGNIFICANT' if result.p_value < 0.05 else '✗ NOT SIGNIFICANT'}")
    print(f"    Hazard Ratio     : {hr:.3f}  (95% CI: {ci_lower:.3f}–{ci_upper:.3f})")
    print(f"    Cox p-value      : {p_cox:.4f}")
    direction = "high expression → WORSE survival" if hr > 1 else "high expression → BETTER survival"
    print(f"    Direction        : {direction}")

print(df['subtype'].value_counts())
print(f"\nExpression ranges:")
for gene in ['FBLN1', 'FBLN2', 'FBLN5']:
    print(f"  {gene}: min={df[gene].min():.2f}, median={df[gene].median():.2f}, max={df[gene].max():.2f}")