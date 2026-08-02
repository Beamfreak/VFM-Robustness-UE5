# Evaluating the Robustness of Vision Foundation Models (VFMs) via UE5 Synthesis

This repository contains the evaluation framework, dataset metadata, and Unreal Engine 5 pipeline used to analyze the out-of-distribution (OOD) robustness of Vision Foundation Models (VFMs) as detailed in the thesis:

**"Evaluating the Robustness of Vision Foundation Models on an ImageNet Class-Subset Using Photorealistic Images Synthesized in Unreal Engine."**

## Overview

Modern vision models often fail under domain shift despite high in-distribution accuracy. This project provides a procedural synthetic rendering pipeline in Unreal Engine 5 to stress-test models like CLIP and DINOv2/v3 under controlled physical transformations.

By mapping ShapeNet 3D models to the ImageNet-1K taxonomy, we evaluate model performance at both fine-grained (leaf class) and semantic (superclass) levels.

## Key Datasets

We introduce two novel synthetic benchmarks with perfect causal granularity:

- **Multi-Factor Dataset:** Systematically varies 5 nuisance factors: Camera Viewpoint, Object Material, Illumination, Background Scene, and Atmospheric Fog.
- **Multi-Color Dataset:** Isolates texture bias by overriding object materials with uniform colored plastic across the primary spectrum.

## Features

- **UE5 Rendering Pipeline:** Built on an adapted Dataset Renderer plugin utilizing Lumen and Nanite for photorealism.
- **Triple Evaluation Strategy:** Standardized comparison across supervised and self-supervised architectures using Logits, k-NN, and Linear Probing.
- **Hierarchy-Aware Metrics:** Analysis of semantic vs. fine-grained robustness to distinguish between label confusion and total semantic collapse.
- **Advanced Diagnostics:** Includes Equivariance Analysis via transition vectors and Spatial Attribution using Grad-CAM and automated segmentation masks.

## Key Findings

- DINOv2/v3 models exhibit superior OOD robustness compared to language-supervised (CLIP) or standard supervised (ResNet) models.
- Top-down viewpoints represent a universal failure mode across all evaluated backbones.
- Supervised fine-tuning often destroys the latent invariance of pre-trained backbones, making linear probing a more reliable metric for OOD robustness.


## Citation

If you use this pipeline or dataset in your research, please cite:

```bibtex
@masterthesis{leinberger2026robustness,
  title={Evaluating the Robustness of Vision Foundation Models on an ImageNet Class-Subset Using Photorealistic Images Synthesized in Unreal Engine},
  author={Markus Leinberger},
  year={2026},
  school={University of Bamberg}
}
