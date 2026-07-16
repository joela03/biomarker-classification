"""
Shared loading, caching, and Cox helper for the project.
"""

import os
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

TARGET_GENES = ['FBLN1', 'FBLN2', 'FBLN5']

METABRIC_MRNA     = 'brca_metabric/data_mrna_illumina_microarray.txt'
METABRIC_CLINICAL = 'brca_metabric/data_clinical_patient.txt'
TCGA_MRNA         = 'brca_tcga/data_mrna_agilent_microarray.txt'
TCGA_CLINICAL     = 'brca_tcga/data_clinical_patient.txt'

METABRIC_CACHE = 'data/processed/metabric_fbln.parquet'
TCGA_CACHE     = 'data/processed/tcga_fbln.parquet'

CATEGORY_COLOURS = {
    'HIGH_CONFIDENCE_FAVOURABLE' : '#2ecc71',
    'HIGH_CONFIDENCE_UNFAVOURABLE': '#e67e22',
    'AMBIGUITY'                  : '#f39c12',
    'CONTROVERSY'                : '#e74c3c',
    'DATA_INSUFFICIENCY'         : '#9b59b6',
    'OUT_OF_SCOPE'               : '#95a5a6',
}

CATEGORY_ORDER = [
    'HIGH_CONFIDENCE_FAVOURABLE',
    'HIGH_CONFIDENCE_UNFAVOURABLE',
    'AMBIGUITY',
    'CONTROVERSY',
    'DATA_INSUFFICIENCY',
    'OUT_OF_SCOPE'
]

BOUNDARY_MULTIPLIER = 0.5

os.makedirs('data/processed', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


def _load_expression(path, strip_suffix=False):
    mrna = pd.read_csv(path, sep='\t', index_col=0, low_memory=False)
    # Remove entrez gene_id column
    if 'Entrez_Gene_Id' in mrna.columns:
        mrna = mrna.drop(columns=['Entrez_Gene_Id'])

    # Check that defined target genes exist in the dataset
    missing = [g for g in TARGET_GENES if g not in mrna.index]
    if missing:
        raise ValueError(f"Genes not found in {path}: {missing}")

    # Locate and transpose rows that have target genes
    expr = mrna.loc[mrna.index.isin(TARGET_GENES)].T
    expr.index.name = 'PATIENT_ID'
    expr = expr.reset_index()

    # Clean ids (remove duplicates and strip hyphens)
    if strip_suffix:
        expr['PATIENT_ID'] = expr['PATIENT_ID'].str.rsplit('-', n=1).str[0]
        dupes = expr['PATIENT_ID'].duplicated().sum()
        if dupes > 0:
            print(f"  {dupes} duplicate patient IDs after suffix strip — keeping first")
            expr = expr.drop_duplicates(subset='PATIENT_ID', keep='first')

    return expr


def load_metabric(force_reload=False):
    if not force_reload and os.path.exists(METABRIC_CACHE):
        print(f"Loading METABRIC from cache")
        return pd.read_parquet(METABRIC_CACHE)

    print("Building METABRIC from raw files...")
    expr = _load_expression(METABRIC_MRNA)

    clin = pd.read_csv(METABRIC_CLINICAL, sep='\t', comment='#', low_memory=False)
    clin.columns = clin.columns.str.upper()

    # Rename os status and event columns
    clin['os_event'] = clin['OS_STATUS'].astype(str).str.startswith('1').astype(int)
    clin = clin.rename(columns={'OS_MONTHS': 'os_months'})

    # Rename Risk Free Survival Columns
    if 'RFS_MONTHS' in clin.columns and 'RFS_STATUS' in clin.columns:
        clin['rfs_event'] = clin['RFS_STATUS'].astype(str).str.startswith('1').astype(int)
        clin = clin.rename(columns={'RFS_MONTHS': 'rfs_months'})

    if 'VITAL_STATUS' in clin.columns:
        clin['dss_event'] = (
            clin['VITAL_STATUS'].astype(str)
            .str.contains('Died of Disease', case=False, na=False)
            .astype(int)
        )

    # Values to add to our final df
    keep = ['PATIENT_ID', 'os_months', 'os_event', 'CLAUDIN_SUBTYPE']
    keep += [c for c in ['rfs_months', 'rfs_event', 'dss_event'] if c in clin.columns]

    # Perform an inner merge on the ID of our selected columns
    df = expr.merge(clin[keep], on='PATIENT_ID', how='inner')
    df = df.rename(columns={'CLAUDIN_SUBTYPE': 'subtype'})
    df = df.dropna(subset=['FBLN1', 'FBLN2', 'FBLN5', 'os_months', 'os_event'])
    df = df.reset_index(drop=True)

    print(f"  {len(df)} patients | OS events: {int(df['os_event'].sum())}", end='')
    if 'rfs_event' in df.columns:
        print(f" | RFS: {int(df['rfs_event'].sum())}", end='')
    if 'dss_event' in df.columns:
        print(f" | DSS: {int(df['dss_event'].sum())}", end='')
    print()

    df.to_parquet(METABRIC_CACHE, index=False)
    print(f"  Cached to {METABRIC_CACHE}")
    return df


def load_tcga(force_reload=False):
    if not force_reload and os.path.exists(TCGA_CACHE):
        print(f"Loading TCGA from cache")
        return pd.read_parquet(TCGA_CACHE)

    print("Building TCGA from raw files...")
    expr = _load_expression(TCGA_MRNA, strip_suffix=True)

    clin = pd.read_csv(TCGA_CLINICAL, sep='\t', comment='#', low_memory=False)
    clin.columns = clin.columns.str.upper()

    clin['os_event'] = clin['OS_STATUS'].astype(str).str.startswith('1').astype(int)
    clin = clin.rename(columns={'OS_MONTHS': 'os_months'})

    if 'DFS_MONTHS' in clin.columns and 'DFS_STATUS' in clin.columns:
        clin['dfs_event'] = clin['DFS_STATUS'].astype(str).str.startswith('1').astype(int)
        clin = clin.rename(columns={'DFS_MONTHS': 'dfs_months'})

    keep = ['PATIENT_ID', 'os_months', 'os_event']
    keep += [c for c in ['dfs_months', 'dfs_event'] if c in clin.columns]

    df = expr.merge(clin[keep], on='PATIENT_ID', how='inner')
    df['os_months'] = pd.to_numeric(df['os_months'], errors='coerce')
    df['os_event']  = pd.to_numeric(df['os_event'],  errors='coerce')
    df = df.dropna(subset=['FBLN1', 'FBLN2', 'FBLN5', 'os_months', 'os_event'])
    df = df.reset_index(drop=True)

    print(f"  {len(df)} patients | OS events: {int(df['os_event'].sum())} ({df['os_event'].mean()*100:.1f}%)")
    df.to_parquet(TCGA_CACHE, index=False)
    print(f"  Cached to {TCGA_CACHE}")
    return df


def run_cox(df, gene, time_col, event_col, split='median'):
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test

    subset = df[[time_col, event_col, gene]].copy()
    subset[time_col]  = pd.to_numeric(subset[time_col],  errors='coerce')
    subset[event_col] = pd.to_numeric(subset[event_col], errors='coerce')
    subset = subset.dropna()

    if subset[event_col].sum() < 10:
        return None

    if split == 'median':
        cut = subset[gene].median()
        subset['gene_high'] = (subset[gene] > cut).astype(int)
    elif split == 'quartile':
        q1 = subset[gene].quantile(0.25)
        q3 = subset[gene].quantile(0.75)
        subset = subset[(subset[gene] < q1) | (subset[gene] > q3)].copy()
        subset['gene_high'] = (subset[gene] > q3).astype(int)

    if subset[event_col].sum() < 10:
        return None

    high = subset[subset['gene_high'] == 1]
    low  = subset[subset['gene_high'] == 0]
    lr   = logrank_test(
        high[time_col], low[time_col],
        event_observed_A=high[event_col],
        event_observed_B=low[event_col]
    )

    cph = CoxPHFitter()
    cph.fit(subset[[time_col, event_col, 'gene_high']],
            duration_col=time_col, event_col=event_col)

    hr       = cph.summary.loc['gene_high', 'exp(coef)']
    ci_lower = cph.summary.loc['gene_high', 'exp(coef) lower 95%']
    ci_upper = cph.summary.loc['gene_high', 'exp(coef) upper 95%']
    p_cox    = cph.summary.loc['gene_high', 'p']

    return {
        'HR':         round(hr, 3),
        'CI_lower':   round(ci_lower, 3),
        'CI_upper':   round(ci_upper, 3),
        'p_logrank':  round(lr.p_value, 4),
        'p_cox':      round(p_cox, 4),
        'n_patients': len(subset),
        'n_events':   int(subset[event_col].sum()),
        'direction':  'invasive' if hr > 1 else 'protective',
        'sig':        '1' if p_cox < 0.05 else '0'
    }


if __name__ == '__main__':
    load_metabric(force_reload=True)
    load_tcga(force_reload=True)