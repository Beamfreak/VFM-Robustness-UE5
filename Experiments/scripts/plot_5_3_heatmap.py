import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def plot_heatmap(dataset_name='Multi-Factor Dataset'):
    base_dir = Path(__file__).resolve().parent.parent
    if dataset_name == 'Multi-Factor Dataset':
        results_dir = base_dir / 'results' / 'normal_dataset'
    else:
        results_dir = base_dir / 'results' / 'plastic_dataset'

    if not results_dir.exists():
        print(f"Warning: {results_dir} does not exist. Skipping {dataset_name}.")
        return

    # We want to map model directories to their display names
    comp_summary_path = results_dir / 'comparative' / 'comparative_summary.json'
    
    model_names = {}
    if comp_summary_path.exists():
        with open(comp_summary_path, 'r') as f:
            data = json.load(f)
            for m_key, m_data in data.get('models', {}).items():
                model_names[m_key] = m_data.get('name', m_key)
    
    heatmap_data = {}
    
    # Iterate over model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == 'comparative':
            continue
            
        csv_path = model_dir / 'stratified_true_shapenet_superclass.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=';')
            # Filter unmapped
            df = df[df['value'] != '<unmapped>']
            
            if 'value' in df.columns and 'shapenet_top1_acc' in df.columns:
                # Convert the column to numeric just in case there are parsing issues
                df['shapenet_top1_acc'] = pd.to_numeric(df['shapenet_top1_acc'], errors='coerce')
                
                acc_dict = dict(zip(df['value'], df['shapenet_top1_acc']))
                
                # Use a nice name if available, else the directory name
                m_name = model_names.get(model_dir.name, model_dir.name)
                heatmap_data[m_name] = acc_dict
                
    if not heatmap_data:
        print(f"No stratified_true_shapenet_superclass.csv files found for {dataset_name}.")
        return

    # Convert to DataFrame: rows are classes, cols are models
    df_heat = pd.DataFrame(heatmap_data)
    
    # Drop rows that are all NaN
    df_heat = df_heat.dropna(how='all')
    
    # Sort models (columns) by their average performance across these categories
    df_heat = df_heat.reindex(df_heat.mean().sort_values(ascending=False).index, axis=1)
    
    # Sort categories (rows) by average performance across models
    df_heat = df_heat.loc[df_heat.mean(axis=1).sort_values(ascending=False).index]
    
    # Plotting
    plt.figure(figsize=(18, 14))
    # We use a visually distinct colormap.
    sns.heatmap(df_heat, cmap='YlGnBu', annot=False, cbar_kws={'label': 'ShapeNet Top-1 Accuracy'})
    
    plt.title(f'ShapeNet Superclass Accuracy by Model ({dataset_name})', fontsize=16)
    plt.xlabel('Models (sorted by average accuracy)', fontsize=12)
    plt.ylabel('ShapeNet Superclasses (sorted by average accuracy)', fontsize=12)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    out_dir = base_dir / 'results' / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'5_3_heatmap_{dataset_name}.png'
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.savefig(out_file.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved heatmap to {out_file}")

if __name__ == '__main__':
    plot_heatmap('Multi-Factor Dataset')
    plot_heatmap('Multi-Color Dataset')
