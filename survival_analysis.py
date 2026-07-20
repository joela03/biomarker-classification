"""
Cross-dataset, cross-endpoint survival analysis for FBLN1, FBLN2, FBLN5.
Includes subtype-stratified Cox models and the triangulation checks.
Results saved to outputs/survival_endpoint_comparison.csv
"""

import pandas as pd
from data_loader import load_metabric, load_tcga, run_cox, TARGET_GENES

df_mb   = load_metabric()
df_tcga = load_tcga()

# Full endpoint comparison

configs = [
    ('METABRIC', 'OS',  df_mb,    'os_months',  'os_event'),
    ('METABRIC', 'DSS', df_mb,    'os_months',  'dss_event'),
    ('METABRIC', 'RFS', df_mb,    'rfs_months', 'rfs_event'),
    ('TCGA',     'OS',  df_tcga,  'os_months',  'os_event'),
    ('TCGA',     'DFS', df_tcga,  'dfs_months', 'dfs_event'),
]

rows = []
for dataset, endpoint, df_use, time_col, event_col in configs:
    if time_col not in df_use.columns or event_col not in df_use.columns:
        continue
    for gene in TARGET_GENES:
        for split in ['median', 'quartile']:
            result = run_cox(df_use, gene, time_col, event_col, split=split)
            if result is None:
                continue
            rows.append({'Dataset': dataset, 'Endpoint': endpoint,
                         'Gene': gene, 'Split': split, **result})

results_df = pd.DataFrame(rows)

print("Survival endpoint comparison:\n")
for (dataset, endpoint), group in results_df.groupby(['Dataset', 'Endpoint']):
    print(f"{dataset} | {endpoint}")
    print(f"  {'Gene':<8} {'Split':<10} {'HR':<8} {'95% CI':<18} "
          f"{'p(cox)':<10} {'n_events':<10} Dir   Sig")
    for _, row in group.iterrows():
        ci = f"{row['CI_lower']}–{row['CI_upper']}"
        print(f"  {row['Gene']:<8} {row['Split']:<10} {row['HR']:<8} "
              f"{ci:<18} {row['p_cox']:<10} {row['n_events']:<10} "
              f"{row['direction']:<6} {row['sig']}")
    print()

results_df.to_csv('outputs/survival_endpoint_comparison.csv', index=False)


# Subtype-stratified analysis (METABRIC OS)

print("\nSubtype-stratified Cox — OS (all-cause):\n")
subtypes = ['LumA', 'LumB', 'Her2', 'claudin-low', 'Basal', 'Normal', 'NC']

for subtype in subtypes:
    sub = df_mb[df_mb['subtype'] == subtype]
    print(f"{subtype} (n={len(sub)})")
    for gene in TARGET_GENES:
        result = run_cox(sub, gene, 'os_months', 'os_event', split='median')
        if result:
            sig = '*' if result['p_cox'] < 0.05 else ''
            print(f"  {gene}: HR={result['HR']:.3f}  p={result['p_cox']:.4f}  "
                  f"{result['direction']} {sig}")
        else:
            print(f"  {gene}: insufficient events")
    print()


# DSS subtype stratification

if 'dss_event' in df_mb.columns:
    print("Subtype-stratified Cox — DSS (disease-specific):\n")
    for subtype in ['LumA', 'LumB', 'Her2', 'Basal']:
        sub = df_mb[df_mb['subtype'] == subtype]
        print(f"{subtype} (n={len(sub)}, DSS events={int(sub['dss_event'].sum())})")
        for gene in TARGET_GENES:
            result = run_cox(sub, gene, 'os_months', 'dss_event', split='median')
            if result:
                sig = '*' if result['p_cox'] < 0.05 else ''
                print(f"  {gene}: HR={result['HR']:.3f}  p={result['p_cox']:.4f}  "
                      f"{result['direction']} {sig}")
            else:
                print(f"  {gene}: insufficient events")
        print()


# Triangulation statistics

import numpy as np

near_any = (
    (abs(df_mb['FBLN1'] - df_mb['FBLN1'].median()) < 0.5 * df_mb['FBLN1'].std()) |
    (abs(df_mb['FBLN2'] - df_mb['FBLN2'].median()) < 0.5 * df_mb['FBLN2'].std()) |
    (abs(df_mb['FBLN5'] - df_mb['FBLN5'].median()) < 0.5 * df_mb['FBLN5'].std())
)
near_all = (
    (abs(df_mb['FBLN1'] - df_mb['FBLN1'].median()) < 0.5 * df_mb['FBLN1'].std()) &
    (abs(df_mb['FBLN2'] - df_mb['FBLN2'].median()) < 0.5 * df_mb['FBLN2'].std()) &
    (abs(df_mb['FBLN5'] - df_mb['FBLN5'].median()) < 0.5 * df_mb['FBLN5'].std())
)

print(f"Near median (≥1 gene): {near_any.mean()*100:.1f}%")
print(f"Near median (all 3):   {near_all.mean()*100:.1f}%")
