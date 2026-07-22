"""
Data-driven rule induction for a protein family.
"""

import numpy as np
import pandas as pd
from data_loader import run_cox
from scipy.optimize import brentq
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from lifelines.statistics import logrank_test

def infer_gene_roles(df, genes, time_col='os_months', event_col='os_event'):
    """
    Run Cox models per gene to infer protective vs invasive direction.
    """
    roles = {}
    for gene in genes:
        result = run_cox(df, gene, time_col, event_col, split='median')
        if result is None:
            roles[gene] = 'unclear'
        elif result['HR'] < 1 and result['p_cox'] < 0.05:
            roles[gene] = 'protective'
        elif result['HR'] > 1 and result['p_cox'] < 0.05:
            roles[gene] = 'invasive'
        else:
            roles[gene] = 'unclear'
    return roles


def infer_controversy_pairs(df, genes, roles,
                            time_col='os_months',
                            event_col='os_event',
                            min_events=30):
    """
    Identify gene pairs where one protective and one invasive gene
    show positive expression correlation - creating genuine controversy.
    """
    subtypes = df['subtype'].dropna().unique()
    gene_pairs = [(g1, g2) for i, g1 in enumerate(genes)
                  for g2 in genes[i+1:]]

    controversy_evidence = {}

    for g1, g2 in gene_pairs:
        pair_key = (g1, g2)
        controversy_evidence[pair_key] = {
            'opposing_subtypes': [],
            'correlation': round(df[g1].corr(df[g2]), 3),
            'is_controversy_pair': False
        }

        for subtype in subtypes:
            sub = df[df['subtype'] == subtype]
            if int(sub[event_col].sum()) < min_events:
                continue

            r1 = run_cox(sub, g1, time_col, event_col, split='median')
            r2 = run_cox(sub, g2, time_col, event_col, split='median')

            if r1 is None or r2 is None:
                continue

            # Opposing directions in this subtype
            g1_protective = r1['HR'] < 1
            g2_protective = r2['HR'] < 1

            if g1_protective != g2_protective:
                controversy_evidence[pair_key]['opposing_subtypes'].append({
                    'subtype':      subtype,
                    'g1_HR':        r1['HR'],
                    'g1_p':         r1['p_cox'],
                    'g2_HR':        r2['HR'],
                    'g2_p':         r2['p_cox'],
                    'g1_direction': 'protective' if g1_protective else 'invasive',
                    'g2_direction': 'protective' if g2_protective else 'invasive',
                })

        # A pair is a controversy pair if:
        # They oppose in at least one powered subtype
        # They are positively correlated (so co-elevation is possible)
        opp = controversy_evidence[pair_key]['opposing_subtypes']
        corr = controversy_evidence[pair_key]['correlation']

        if len(opp) > 0 and corr > 0.2:
            controversy_evidence[pair_key]['is_controversy_pair'] = True

    # Report
    print("\nSubtype-stratified controversy detection:")
    controversy_pairs = []

    for (g1, g2), evidence in controversy_evidence.items():
        corr = evidence['correlation']
        opp  = evidence['opposing_subtypes']
        is_c = evidence['is_controversy_pair']
        flag = '✓ CONTROVERSY PAIR' if is_c else '✗'

        print(f"\n  {g1} vs {g2}  (correlation r={corr})  {flag}")
        if opp:
            for sub_ev in opp:
                print(f"    {sub_ev['subtype']:<15} "
                      f"{g1}={sub_ev['g1_direction']} "
                      f"(HR={sub_ev['g1_HR']:.3f}, p={sub_ev['g1_p']:.3f})  "
                      f"{g2}={sub_ev['g2_direction']} "
                      f"(HR={sub_ev['g2_HR']:.3f}, p={sub_ev['g2_p']:.3f})")
        else:
            print(f"    No powered subtypes show opposing directions")

        if is_c:
            controversy_pairs.append((g1, g2, corr))

    return controversy_pairs


def calibrate_boundary(df, genes, target_ambiguity_pct=0.10):
    """
    Find the SD multiplier that produces approximately target_ambiguity_pct
    """
    
    def ambiguity_pct(multiplier):
        near = pd.Series([True] * len(df))
        for gene in genes:
            med = df[gene].median()
            std = df[gene].std()
            near = near & (abs(df[gene] - med) < multiplier * std)
        return near.mean() - target_ambiguity_pct
    
    try:
        optimal = brentq(ambiguity_pct, 0.1, 1.5)
    except ValueError:
        optimal = 0.5  # fallback if no solution in range
    
    return round(optimal, 3)


def infer_data_insufficiency(df, genes, roles):
    """
    DATA_INSUFFICIENCY = simultaneously extreme low expression on
    invasive genes
    """
    invasive_genes = [g for g, r in roles.items() if r == 'invasive']
    if not invasive_genes:
        invasive_genes = genes  # fallback if roles unclear
    
    thresholds = {g: df[g].quantile(0.05) for g in invasive_genes}
    return thresholds


def build_framework_config(df, genes,
                            time_col='os_months',
                            event_col='os_event',
                            target_ambiguity_pct=0.10):
    """
    Master function. Given a dataframe with any protein family,
    returns a config dict that assign_category can consume.
    """
    print(f"Inferring rules for gene family: {genes}")
    
    # Gene roles from survival
    roles = infer_gene_roles(df, genes, time_col, event_col)
    print(f"  Inferred roles: {roles}")
    
    # Controversy pairs
    controversy_pairs = infer_controversy_pairs(df, genes, roles)
    print(f"  Controversy pairs: {controversy_pairs}")
    
    # Boundary calibration
    boundary_multiplier = calibrate_boundary(df, genes, target_ambiguity_pct)
    print(f"  Boundary multiplier: {boundary_multiplier} SD")
    
    # Data insufficiency thresholds
    di_thresholds = infer_data_insufficiency(df, genes, roles)
    print(f"  DATA_INSUFFICIENCY thresholds: {di_thresholds}")
    
    # Out of scope — always 3.5 SD, gene-agnostic
    oos_multiplier = 3.5
    
    config = {
        'genes':                genes,
        'roles':                roles,
        'controversy_pairs':    controversy_pairs,
        'boundary_multiplier':  boundary_multiplier,
        'di_thresholds':        di_thresholds,
        'oos_multiplier':       oos_multiplier,
        'time_col':             time_col,
        'event_col':            event_col,
    }
    
    return config

def compare_rule_sets(df, genes, config, original_categories,
                      time_col='os_months', event_col='os_event'):
    """
    Compare original hand-written rule categories against
    data-driven inferred categories from induction config.
    """

    # Build inferred category assignments using config
    assign_fn = make_assign_category(df, config)
    df = df.copy()
    df['category_inferred'] = df.apply(assign_fn, axis=1)
    df['category_original'] = original_categories

    # Category distribution comparison
    print("Category distribution comparison:")
    print(f"\n  {'Category':<35} {'Original':>10} {'Inferred':>10}")
    print("  " + "-" * 57)

    all_cats = sorted(set(df['category_original'].unique()) |
                      set(df['category_inferred'].unique()))
    orig_counts = df['category_original'].value_counts()
    inf_counts  = df['category_inferred'].value_counts()
    N = len(df)

    for cat in all_cats:
        orig_pct = orig_counts.get(cat, 0) / N * 100
        inf_pct  = inf_counts.get(cat, 0) / N * 100
        diff     = inf_pct - orig_pct
        diff_str = f"({diff:+.1f}%)" if abs(diff) > 0.5 else ""
        print(f"  {cat:<35} {orig_pct:>9.1f}%  {inf_pct:>9.1f}%  {diff_str}")

    # Agreement between rule sets
    le = LabelEncoder()
    orig_encoded = le.fit_transform(df['category_original'])
    inf_encoded  = LabelEncoder().fit_transform(df['category_inferred'])
    ari   = adjusted_rand_score(orig_encoded, inf_encoded)

    from sklearn.metrics import cohen_kappa_score
    # Align encodings
    all_labels = sorted(set(df['category_original']) |
                        set(df['category_inferred']))
    label_map  = {l: i for i, l in enumerate(all_labels)}
    orig_mapped = df['category_original'].map(label_map)
    inf_mapped  = df['category_inferred'].map(label_map)
    kappa = cohen_kappa_score(orig_mapped, inf_mapped)

    print(f"\n  Agreement between original and inferred rules:")
    print(f"    Adjusted Rand Index : {ari:.3f}")
    print(f"    Cohen's Kappa       : {kappa:.3f}")
    print(f"    Interpretation: ", end="")
    if kappa > 0.6:
        print("strong agreement — inferred rules replicate original reasoning")
    elif kappa > 0.4:
        print("moderate agreement — broadly consistent with some boundary differences")
    elif kappa > 0.2:
        print("fair agreement — inferred rules capture similar but not identical structure")
    else:
        print("weak agreement — inferred rules diverge from original reasoning")

    # Cross-tabulation (checking disagreement)
    print("\n  Cross-tabulation (original rows vs inferred columns):")
    cross = pd.crosstab(df['category_original'],
                        df['category_inferred'],
                        margins=True, margins_name='Total')
    print(cross.to_string())

    # Survival separation
    print("\n  Survival separation comparison:")
    print(f"  {'Category':<35} {'Original HR':>12} {'Inferred HR':>12}")
    print("  " + "-" * 62)

    for cat in all_cats:
        if cat in ['OUT_OF_SCOPE', 'DATA_INSUFFICIENCY']:
            continue

        orig_sub = df[df['category_original'] == cat]
        inf_sub  = df[df['category_inferred'] == cat]

        if len(orig_sub) < 10 or len(inf_sub) < 10:
            continue

        # Event rate as a simple survival proxy
        orig_rate = orig_sub[event_col].mean()
        inf_rate  = inf_sub[event_col].mean()

        print(f"  {cat:<35} "
              f"event={orig_rate:.1%}    "
              f"event={inf_rate:.1%}")

    # Pairwise log-rank
    print("\n  Log-rank: HC_FAVOURABLE vs HC_UNFAVOURABLE")
    for label_col, name in [('category_original', 'Original'),
                             ('category_inferred', 'Inferred')]:
        fav   = df[df[label_col] == 'HIGH_CONFIDENCE_FAVOURABLE']
        unfav = df[df[label_col] == 'HIGH_CONFIDENCE_UNFAVOURABLE']
        if len(fav) < 10 or len(unfav) < 10:
            print(f"    {name}: insufficient patients")
            continue
        result = logrank_test(
            fav[time_col], unfav[time_col],
            event_observed_A=fav[event_col],
            event_observed_B=unfav[event_col]
        )
        sig = 'significiant' if result.p_value < 0.05 else 'insignificant'
        print(f"    {name}: p={result.p_value:.4f} {sig}")

    return df

def make_assign_category(df, config):
    """
    Returns an assign_category function configured for any gene family.
    Replaces the hardcoded FBLN-specific version.
    """
    genes    = config['genes']
    roles    = config['roles']
    boundary = config['boundary_multiplier']
    oos_mult = config['oos_multiplier']
    di_thresholds  = config['di_thresholds']
    controversy_pairs = config['controversy_pairs']
    
    # Precompute medians and stds
    stats = {g: {'med': df[g].median(), 'std': df[g].std()} for g in genes}
    
    def assign_category(row):
        
        # OUT_OF_SCOPE - extreme outlier on any gene
        for gene in genes:
            med = stats[gene]['med']
            std = stats[gene]['std']
            if abs(row[gene] - med) > oos_mult * std:
                return 'OUT_OF_SCOPE'
        
        # DATA_INSUFFICIENCY - extreme low on invasive genes simultaneously
        if all(row[g] < di_thresholds[g]
               for g in di_thresholds):
            return 'DATA_INSUFFICIENCY'
        
        # AMBIGUITY - all genes near median
        near_median = all(
            abs(row[gene] - stats[gene]['med']) < boundary * stats[gene]['std']
            for gene in genes
        )
        if near_median:
            return 'AMBIGUITY'
        
        # CONTROVERSY - protective gene elevated alongside invasive gene
        for protective_gene, invasive_gene, _ in controversy_pairs:
            p_above = row[protective_gene] > stats[protective_gene]['med']
            i_above = row[invasive_gene]   > stats[invasive_gene]['med']
            if p_above and i_above:
                return 'CONTROVERSY'
        
        # HIGH_CONFIDENCE - subtype-aware split
        aggressive = {'Basal', 'Her2', 'LumB'}
        subtype = row.get('subtype', None)
        if subtype in aggressive:
            return 'HIGH_CONFIDENCE_UNFAVOURABLE'
        return 'HIGH_CONFIDENCE_FAVOURABLE'
    
    return assign_category

if __name__ == '__main__':
    from data_loader import load_metabric, TARGET_GENES
    import pandas as pd

    df   = load_metabric()
    cats = pd.read_parquet('data/processed/metabric_categories.parquet')
    df   = df.merge(cats, on='PATIENT_ID', how='left')

    # Build config from survival data
    config = build_framework_config(df, TARGET_GENES)

    # Compare original vs inferred
    df_compared = compare_rule_sets(
        df=df,
        genes=TARGET_GENES,
        config=config,
        original_categories=df['category']
    )