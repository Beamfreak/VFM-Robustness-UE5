import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def generate_average_accuracy_table(dataset_name='normal_dataset'):
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / 'results' / dataset_name
    
    if not results_dir.exists():
        print(f"Warning: {results_dir} does not exist. Skipping {dataset_name}.")
        return None

    all_data = []
    
    # Iterate over model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == 'comparative':
            continue
            
        csv_path = model_dir / 'stratified_true_shapenet_superclass.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=';')
            df = df[df['value'] != '<unmapped>']
            
            if 'value' in df.columns and 'imagenet_top1_acc' in df.columns and 'shapenet_top1_acc' in df.columns:
                df['imagenet_top1_acc'] = pd.to_numeric(df['imagenet_top1_acc'], errors='coerce')
                df['shapenet_top1_acc'] = pd.to_numeric(df['shapenet_top1_acc'], errors='coerce')
                
                for _, row in df.iterrows():
                    all_data.append({
                        'model': model_dir.name,
                        'class': row['value'],
                        'imagenet_acc': row['imagenet_top1_acc'],
                        'shapenet_acc': row['shapenet_top1_acc']
                    })

    if not all_data:
        print(f"No data found for {dataset_name}.")
        return None

    # Create a master dataframe
    df_all = pd.DataFrame(all_data)
    
    # Group by class and calculate means
    grouped = df_all.groupby('class').agg({
        'imagenet_acc': 'mean',
        'shapenet_acc': 'mean'
    }).reset_index()
    
    # Calculate Ratio
    grouped['ratio'] = grouped['shapenet_acc'] / grouped['imagenet_acc'].replace(0, np.nan)
    
    # Sort by ShapeNet accuracy descending
    grouped = grouped.sort_values(by='shapenet_acc', ascending=False)
    
    # Format the numbers for the table
    grouped['imagenet_acc'] = grouped['imagenet_acc'].apply(lambda x: f"{x:.1%}")
    grouped['shapenet_acc'] = grouped['shapenet_acc'].apply(lambda x: f"{x:.1%}")
    grouped['ratio'] = grouped['ratio'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
    
    # Rename columns for display
    grouped.columns = ['ShapeNet Superclass', 'Avg ImageNet Top-1', 'Avg ShapeNet Top-1', 'Ratio (ShapeNet/ImageNet)']
    return grouped

def plot_tables():
    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / 'results' / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_normal = generate_average_accuracy_table('normal_dataset')
    df_plastic = generate_average_accuracy_table('plastic_dataset')
    
    if df_normal is not None:
        print("\n" + "="*60)
        print("MULTI-FACTOR DATASET - AVERAGE ACCURACY TABLE")
        print("="*60)
        print(df_normal.to_string(index=False))
        
    if df_plastic is not None:
        print("\n" + "="*60)
        print("MULTI-COLOR DATASET - AVERAGE ACCURACY TABLE")
        print("="*60)
        print(df_plastic.to_string(index=False))
        print("\n")
    
    # We will plot both tables side-by-side or stacked in one figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 14))
    fig.suptitle('Average ShapeNet vs ImageNet Accuracy per Class (Across All Models)', fontsize=16, y=0.98)
    
    datasets = [
        ('Multi-Factor Dataset', df_normal, axes[0]),
        ('Multi-Color Dataset', df_plastic, axes[1])
    ]
    
    for title, df, ax in datasets:
        ax.axis('off')
        if df is None:
            ax.text(0.5, 0.5, f"No data for {title}", ha='center', va='center')
            continue
            
        ax.set_title(title, fontsize=14, pad=10)
        
        # Create table
        table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         loc='center',
                         cellLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Make headers bold and add background color
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')
            elif col == 0:
                # Class names left aligned
                cell.set_text_props(ha='left')
                cell.set_edgecolor('black')
                
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_file = out_dir / '5_3_accuracy_tables.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.savefig(out_file.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved accuracy tables to {out_file}")

if __name__ == '__main__':
    plot_tables()
