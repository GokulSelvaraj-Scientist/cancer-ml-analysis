# Cancer Diagnosis: Machine Learning Classification and Clustering

## Overview
This project applies supervised and unsupervised machine learning to classify breast cancer diagnoses and discover patient subgroups using the Wisconsin Breast Cancer Diagnostic Dataset. It demonstrates end-to-end ML workflow including data preprocessing, model training, cross-validation, performance evaluation, and visualization.

## Background
Early and accurate diagnosis of breast cancer is critical for patient outcomes. Machine learning models trained on quantitative cell nucleus features from fine needle aspirate (FNA) images can assist pathologists in distinguishing malignant from benign tumors. This project demonstrates how ML can be applied to clinical diagnostic data in an oncology context.

## Dataset
- **Source:** Wisconsin Breast Cancer Dataset (`mlbench` R package) — publicly available
- **Samples:** 683 patients (after removing missing values)
- **Features:** 9 quantitative features from digitized cell nucleus images (clump thickness, cell size uniformity, cell shape uniformity, marginal adhesion, single epithelial cell size, bare nuclei, bland chromatin, normal nucleoli, mitoses)
- **Target:** Diagnosis — Benign vs Malignant
- **Class distribution:** ~65% Benign, ~35% Malignant

## Methods

### Supervised Classification
- 80/20 train/test split with stratified sampling
- 5-fold cross-validation for model tuning
- **Random Forest** — ensemble method with variable importance
- **Logistic Regression** — interpretable baseline model
- Performance metrics: Accuracy, Sensitivity, Specificity, AUC-ROC

### Unsupervised Clustering
- Feature scaling (z-score normalization)
- Optimal cluster selection using within-sum-of-squares (elbow method)
- **K-means clustering** (k=2, matching known biology)
- PCA visualization of cluster structure vs true labels

## Results

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Random Forest | ~0.97 | ~0.96 | ~0.98 | ~0.99 |
| Logistic Regression | ~0.96 | ~0.95 | ~0.97 | ~0.99 |

- K-means clustering with k=2 recovers the benign/malignant structure with high agreement
- Cell size and shape uniformity are the most predictive features

## Outputs
| File | Description |
|---|---|
| `feature_distributions.png` | Density plots of all features by diagnosis |
| `correlation_heatmap.png` | Feature correlation matrix |
| `roc_curves.png` | ROC curves for both models with AUC |
| `feature_importance.png` | Top 10 most important features (Random Forest) |
| `confusion_matrix.png` | Confusion matrix heatmap (Random Forest) |
| `optimal_clusters.png` | Elbow plot for optimal k selection |
| `kmeans_clustering.png` | PCA plot of k-means clusters vs true diagnosis |
| `model_performance_summary.csv` | Full performance metrics for both models |

## How to Run
1. Install R and RStudio
2. Install required packages:
```r
install.packages(c("tidyverse", "caret", "randomForest", "ggplot2",
                   "pheatmap", "RColorBrewer", "pROC", "cluster",
                   "factoextra", "corrplot", "mlbench"))
```
3. Run `cancer_ml_analysis.R` in RStudio

## Requirements
- R >= 4.0
- tidyverse, caret, randomForest, ggplot2, pROC, cluster, factoextra, corrplot, mlbench

## Author
**Gokul Selvaraj, PhD**
GitHub: [GokulSelvaraj-Scientist](https://github.com/GokulSelvaraj-Scientist)
