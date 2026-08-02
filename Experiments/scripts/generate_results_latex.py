import pandas as pd
import numpy as np

# Define file paths
csv_path = "results/aggregate/KNN/01_master_all_models_all_datasets.csv"
output_difficulty = "results/aggregate/KNN/latex_difficulty_spectrum.tex"
output_model_comparison = "results/aggregate/KNN/latex_model_comparison.tex"
output_correlations = "results/aggregate/KNN/latex_correlations.tex"

def main():
    print("Loading master results...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Extract Top1 columns
    top1_cols = [c for c in df.columns if c.endswith('_Top1')]
    dataset_cols = [c.replace('_Top1', '') for c in top1_cols]
    
    # Rename columns to cleaner names for printing
    col_mapping = {c: c.replace('_Top1', '') for c in top1_cols}
    df_top1 = df[['Model'] + top1_cols].rename(columns=col_mapping)
    
    # -------------------------------------------------------------
    # 1. Dataset Difficulty Spectrum Table
    # -------------------------------------------------------------
    stats = []
    for col in dataset_cols:
        series = df_top1[col].dropna()
        if len(series) == 0:
            continue
        mean_val = series.mean() * 100
        min_val = series.min() * 100
        max_val = series.max() * 100
        best_idx = series.idxmax()
        best_model = df_top1.loc[best_idx, 'Model']
        stats.append({
            'dataset': col,
            'mean': mean_val,
            'min': min_val,
            'max': max_val,
            'best_model': best_model
        })
    df_stats = pd.DataFrame(stats).sort_values(by='mean', ascending=False)
    
    # Generate Difficulty LaTeX Table
    diff_latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{Mean Acc. (\\%)} & \\textbf{Min Acc. (\\%)} & \\textbf{Max Acc. (\\%)} & \\textbf{Best Model} \\\\\n"
        "\\midrule\n"
    )
    for _, r in df_stats.iterrows():
        clean_name = r['dataset'].replace('_', '\\_')
        best_model_clean = r['best_model'].replace('_', '\\_')
        diff_latex += f"{clean_name} & {r['mean']:.2f}\\% & {r['min']:.2f}\\% & {r['max']:.2f}\\% & {best_model_clean} \\\\\n"
    diff_latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Robustness benchmarks ordered by difficulty (mean Top-1 accuracy across all 28 models). Our synthetic datasets rank among the most challenging benchmarks alongside ObjectNet.}\n"
        "\\label{tab:dataset_difficulty_spectrum}\n"
        "\\end{table}\n"
    )
    with open(output_difficulty, 'w') as f:
        f.write(diff_latex)
    print(f"Saved difficulty spectrum table to {output_difficulty}")

    # -------------------------------------------------------------
    # 2. Key Models Comparison Table
    # -------------------------------------------------------------
    selected_models = [
        'ResNet-50-IN1K-KNN',
        'ViT-L-IN1K-KNN',
        'Swin-L-IN1K-KNN',
        'CLIP-B-IN1K-KNN',
        'CLIP-L-IN1K-KNN',
        'DINOv2-B-KNN',
        'DINOv2-L-KNN',
        'DINOv3-B-KNN',
        'DINOv3-L-KNN'
    ]
    
    selected_datasets = [
        'imagenet-1k',
        'imagenet-v2',
        'imagenet-r',
        'imagenet-sketch',
        'imagenet-a',
        'objectnet',
        'PUG_ImageNet',
        'normal_dataset',
        'plastic_dataset'
    ]
    
    # Filter datasets that actually exist
    selected_datasets = [d for d in selected_datasets if d in df_top1.columns]
    # Filter models that actually exist
    available_models = [m for m in selected_models if m in df_top1['Model'].values]
    
    df_comp = df_top1[df_top1['Model'].isin(available_models)].copy()
    
    # Generate Model Comparison LaTeX Table
    col_headers = " & ".join([f"\\textbf{{{d.replace('_', '\\_')}}}" for d in selected_datasets])
    comp_latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\begin{tabular}{l" + "c" * len(selected_datasets) + "}\n"
        "\\toprule\n"
        "\\textbf{Model} & " + col_headers + " \\\\\n"
        "\\midrule\n"
    )
    for m in selected_models:
        if m not in available_models:
            continue
        row_data = df_comp[df_comp['Model'] == m].iloc[0]
        acc_str = []
        for d in selected_datasets:
            val = row_data[d]
            if pd.isna(val):
                acc_str.append("-")
            else:
                acc_str.append(f"{val*100:.1f}\\%")
        comp_latex += f"{m.replace('_', '\\_')} & " + " & ".join(acc_str) + " \\\\\n"
    comp_latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Comparison of top-performing and baseline vision models across standard robustness datasets and our synthetic normal/plastic benchmarks. Accuracies are reported as Top-1 accuracy (\\%).}\n"
        "\\label{tab:model_robustness_comparison}\n"
        "\\end{table}\n"
    )
    with open(output_model_comparison, 'w') as f:
        f.write(comp_latex)
    print(f"Saved model comparison table to {output_model_comparison}")

    # -------------------------------------------------------------
    # 3. Spearman Rank Correlation Table
    # -------------------------------------------------------------
    corrs = []
    for col in dataset_cols:
        if col in ['normal_dataset', 'plastic_dataset']:
            continue
        corr_normal = df_top1['normal_dataset'].corr(df_top1[col], method='spearman')
        corr_plastic = df_top1['plastic_dataset'].corr(df_top1[col], method='spearman')
        corrs.append({
            'dataset': col,
            'corr_normal': corr_normal,
            'corr_plastic': corr_plastic
        })
    df_corrs = pd.DataFrame(corrs).sort_values(by='corr_normal', ascending=False)
    
    corr_latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{Spearman $\\rho$ (Normal)} & \\textbf{Spearman $\\rho$ (Plastic)} \\\\\n"
        "\\midrule\n"
    )
    for _, r in df_corrs.iterrows():
        clean_name = r['dataset'].replace('_', '\\_')
        corr_latex += f"{clean_name} & {r['corr_normal']:.3f} & {r['corr_plastic']:.3f} \\\\\n"
    corr_latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Spearman rank correlation coefficients ($\\rho$) between vision model performance on our synthetic datasets and established out-of-distribution benchmarks. High positive values indicate model performance rankings remain highly consistent.}\n"
        "\\label{tab:spearman_rank_correlations}\n"
        "\\end{table}\n"
    )
    with open(output_correlations, 'w') as f:
        f.write(corr_latex)
    print(f"Saved correlations table to {output_correlations}")

if __name__ == "__main__":
    main()
