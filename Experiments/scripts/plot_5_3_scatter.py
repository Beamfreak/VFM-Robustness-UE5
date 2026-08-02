import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import scipy.stats as stats

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    x = []
    y = []
    names = []
    
    for model_key, model_data in data['models'].items():
        if 'imagenet' in model_data and 'shapenet' in model_data:
            in_acc = model_data['imagenet']['top1_accuracy']
            sn_acc = model_data['shapenet']['top1_accuracy']
            if in_acc is not None and sn_acc is not None:
                x.append(in_acc)
                y.append(sn_acc)
                names.append(model_data.get('name', model_key))
                
    return np.array(x), np.array(y), names

def plot_scatter():
    base_dir = Path(__file__).resolve().parent.parent
    normal_path = base_dir / 'results' / 'normal_dataset' / 'comparative' / 'comparative_summary.json'
    plastic_path = base_dir / 'results' / 'plastic_dataset' / 'comparative' / 'comparative_summary.json'
    
    out_dir = base_dir / 'results' / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    datasets = [
        ('Multi-Factor Dataset', normal_path, axes[0]),
        ('Multi-Color Dataset', plastic_path, axes[1])
    ]
    
    for title, path, ax in datasets:
        if not path.exists():
            print(f"Warning: {path} does not exist.")
            continue
            
        x, y, names = load_data(path)
        
        sns.regplot(x=x, y=y, ax=ax, scatter=True, ci=95, 
                    scatter_kws={'alpha':0.6}, line_kws={'color': 'red', 'label': 'Regression Line'})
        
        # Calculate correlation
        r, p = stats.pearsonr(x, y)
        
        anchors = ['DINOv2-L-KNN', 'ResNet-50-KNN', 'CLIP-L-KNN', 'DINOv3-L-KNN', 'Swin-L-KNN', 'ViT-B-KNN']
        for xi, yi, name in zip(x, y, names):
            if name in anchors:
                ax.annotate(name, (xi, yi), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_title(f'{title}\nr = {r:.3f}')
        ax.set_xlabel('ImageNet Top-1 Accuracy')
        ax.set_ylabel('ShapeNet Top-1 Accuracy')
        
        # Add y=x line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='y=x (Equal Acc)')
        ax.legend()

    plt.tight_layout()
    out_file = out_dir / '5_3_scatter_imagenet_vs_shapenet.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.savefig(out_file.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Plot saved to {out_file}")

if __name__ == '__main__':
    plot_scatter()
