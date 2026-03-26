# Cancer Diagnosis: Machine Learning Classification and Clustering

## Overview
This project applies supervised and unsupervised machine learning to classify breast cancer diagnoses and discover patient subgroups using the Wisconsin Breast Cancer Diagnostic Dataset. It demonstrates an end-to-end ML workflow including data preprocessing, model training, cross-validation, performance evaluation, and visualization — with full biological interpretation at each step.

📊 **[View Full Interactive Report](cancer_ml_report.html)** — rendered R Markdown with narrative, code, and plots

## Why This Matters
Machine learning is transforming oncology drug development at every stage of the pipeline. The ability to build, validate, and interpret predictive models from clinical and molecular data is a core skill for life science data scientists. This project demonstrates directly applicable capabilities:

- **Diagnostic support tools** — ML models trained on quantitative biomarkers can assist pathologists in making faster, more reproducible diagnoses, reducing inter-observer variability in clinical trials
- **Patient stratification** — classification models identify patient subgroups with distinct biology, enabling enrichment strategies that increase trial power and reduce sample size requirements
- **Biomarker validation** — feature importance analysis identifies which molecular features are most predictive, prioritizing candidates for prospective biomarker validation studies
- **Regulatory context** — FDA has issued guidance on AI/ML-based software as a medical device (SaMD); understanding model performance metrics (sensitivity, specificity, AUC) in a clinical context is essential for regulatory submissions
- **Companion diagnostics** — the framework demonstrated here — train a classifier on molecular features, validate on held-out data, assess clinical performance — mirrors the development pathway for companion diagnostic tests that support precision medicine approvals

The unsupervised clustering component is equally relevant — in drug development, clustering patient data without using outcome labels is used to discover novel patient subpopulations that may respond differently to treatment, informing adaptive trial designs.

## Dataset
- **Source:** Wisconsin Breast Cancer Dataset (`mlbench` R package)
- **Samples:** 683 patients after removing missing values
- **Features:** 9 quantitative features from digitized FNA cell nucleus images
- **Target:** Diagnosis — Benign vs Malignant

## Methods

### Supervised Classification
- 80/20 train/test split with stratified sampling
- 5-fold cross-validation
- Random Forest and Logistic Regression
- Performance: Accuracy, Sensitivity, Specificity, AUC-ROC

### Unsupervised Clustering
- Feature scaling (z-score normalization)
- Elbow method for optimal k selection
- K-means clustering with PCA visualization

## Results

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Random Forest | 0.978 | 0.979 | 0.977 | 0.998 |
| Logistic Regression | 0.978 | 0.979 | 0.977 | 0.999 |

## Outputs
| File | Description |
|---|---|
| `cancer_ml_report.html` | Full R Markdown report with narrative and biological interpretation |
| `feature_distributions.png` | Density plots of all features by diagnosis |
| `correlation_heatmap.png` | Feature correlation matrix |
| `roc_curves.png` | ROC curves for both models with AUC |
| `feature_importance.png` | Top 10 most important features |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `optimal_clusters.png` | Elbow plot for k selection |
| `kmeans_clustering.png` | PCA plot of clusters vs true diagnosis |
| `model_performance_summary.csv` | Full performance metrics |

## Key Findings
- Both models achieve 97.8% accuracy and AUC > 0.998 on held-out test data
- Cell size and shape uniformity are the strongest predictors — consistent with established pathological criteria for malignancy
- K-means clustering independently recovers the benign/malignant distinction without using diagnosis labels — validating that the feature space captures genuine biological signal
- Only 3 misclassifications out of 135 test samples — clinically meaningful performance

## Biological and Clinical Interpretation
The dominance of cell size and shape uniformity in feature importance directly reflects the biology of cancer — malignant cells lose the tight size and shape regulation that characterizes normal tissue homeostasis. The fact that mitotic rate (Mitoses) contributes least suggests that proliferation rate alone is not the key distinguishing feature in FNA samples, consistent with pathological guidelines that weight nuclear morphology more heavily than mitotic figures in fine needle aspiration cytology. The near-identical performance of Random Forest and Logistic Regression is clinically important — it suggests a linear decision boundary exists in feature space, which favors the simpler logistic regression model for regulatory and interpretability reasons when deploying diagnostic tools.

## How to Run
```r
install.packages(c("tidyverse", "caret", "randomForest", "ggplot2",
                   "pheatmap", "RColorBrewer", "pROC", "cluster",
                   "factoextra", "corrplot", "mlbench"))
```
Run `cancer_ml_analysis.R` or knit `cancer_ml_report.Rmd` in RStudio.

## Author
**Gokul Selvaraj, PhD**
GitHub: [GokulSelvaraj-Scientist](https://github.com/GokulSelvaraj-Scientist)

