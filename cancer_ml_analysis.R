# ============================================================
# Cancer ML Analysis: Classification and Clustering
# Author: Gokul Selvaraj
# GitHub: GokulSelvaraj-Scientist
# Description: Machine learning analysis of breast cancer data
#              including classification (Random Forest, Logistic
#              Regression) and unsupervised clustering (k-means,
#              hierarchical) using the Wisconsin Breast Cancer dataset
# ============================================================

# --- Load Libraries ---
library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)
library(pROC)
library(cluster)
library(factoextra)
library(corrplot)

# --- Install if needed ---
# install.packages(c("tidyverse", "caret", "randomForest", "ggplot2",
#                    "pheatmap", "RColorBrewer", "pROC", "cluster",
#                    "factoextra", "corrplot"))

# ============================================================
# Dataset: Wisconsin Breast Cancer Diagnostic Dataset
# Built into R as mlbench::BreastCancer or available via UCI
# Features: 30 numeric features from digitized cell nucleus images
# Target: Diagnosis (Malignant vs Benign)
# ============================================================

# Load dataset
data(BreastCancer, package = "mlbench")

# --- Data Cleaning ---
bc_clean <- BreastCancer %>%
  select(-Id) %>%
  mutate(across(-Class, ~ as.numeric(as.character(.)))) %>%
  filter(complete.cases(.)) %>%
  mutate(Class = factor(Class, levels = c("benign", "malignant")))

cat("Dataset dimensions:", nrow(bc_clean), "samples x", ncol(bc_clean), "features\n")
cat("Class distribution:\n")
print(table(bc_clean$Class))

# --- Plot 1: Feature Distributions by Class ---
bc_long <- bc_clean %>%
  pivot_longer(cols = -Class, names_to = "Feature", values_to = "Value")

feature_dist <- ggplot(bc_long, aes(x = Value, fill = Class)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ Feature, scales = "free", ncol = 3) +
  scale_fill_manual(values = c("benign" = "#2A9D8F", "malignant" = "#E76F51")) +
  labs(
    title    = "Feature Distributions by Diagnosis",
    subtitle = "Wisconsin Breast Cancer Dataset",
    x        = "Feature Value",
    y        = "Density",
    fill     = "Diagnosis"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold"),
    strip.text    = element_text(size = 8),
    legend.position = "top"
  )

ggsave("feature_distributions.png", feature_dist, width = 12, height = 10, dpi = 300)
cat("Saved: feature_distributions.png\n")

# --- Plot 2: Correlation Heatmap ---
cor_matrix <- cor(bc_clean %>% select(-Class), use = "complete.obs")

png("correlation_heatmap.png", width = 1000, height = 900, res = 150)
corrplot(
  cor_matrix,
  method  = "color",
  type    = "upper",
  tl.cex  = 0.7,
  tl.col  = "black",
  col     = colorRampPalette(c("#457B9D", "white", "#E63946"))(100),
  title   = "Feature Correlation Matrix",
  mar     = c(0, 0, 2, 0)
)
dev.off()
cat("Saved: correlation_heatmap.png\n")

# ============================================================
# PART 1: SUPERVISED CLASSIFICATION
# ============================================================

# --- Train/Test Split (80/20) ---
set.seed(42)
train_idx  <- createDataPartition(bc_clean$Class, p = 0.8, list = FALSE)
train_data <- bc_clean[train_idx, ]
test_data  <- bc_clean[-train_idx, ]

cat("\nTraining set:", nrow(train_data), "samples\n")
cat("Test set:", nrow(test_data), "samples\n")

# --- Cross-validation setup ---
ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# --- Model 1: Random Forest ---
cat("\nTraining Random Forest...\n")
set.seed(42)
rf_model <- train(
  Class ~ .,
  data      = train_data,
  method    = "rf",
  trControl = ctrl,
  metric    = "ROC",
  tuneLength = 5
)

rf_pred  <- predict(rf_model, test_data)
rf_probs <- predict(rf_model, test_data, type = "prob")
rf_cm    <- confusionMatrix(rf_pred, test_data$Class, positive = "malignant")

cat("\nRandom Forest Results:\n")
cat("Accuracy:", round(rf_cm$overall["Accuracy"], 3), "\n")
cat("Sensitivity:", round(rf_cm$byClass["Sensitivity"], 3), "\n")
cat("Specificity:", round(rf_cm$byClass["Specificity"], 3), "\n")

# --- Model 2: Logistic Regression ---
cat("\nTraining Logistic Regression...\n")
set.seed(42)
lr_model <- train(
  Class ~ .,
  data      = train_data,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl,
  metric    = "ROC"
)

lr_pred  <- predict(lr_model, test_data)
lr_probs <- predict(lr_model, test_data, type = "prob")
lr_cm    <- confusionMatrix(lr_pred, test_data$Class, positive = "malignant")

cat("\nLogistic Regression Results:\n")
cat("Accuracy:", round(lr_cm$overall["Accuracy"], 3), "\n")
cat("Sensitivity:", round(lr_cm$byClass["Sensitivity"], 3), "\n")
cat("Specificity:", round(lr_cm$byClass["Specificity"], 3), "\n")

# --- Plot 3: ROC Curves ---
rf_roc <- roc(test_data$Class, rf_probs$malignant, levels = c("benign", "malignant"))
lr_roc <- roc(test_data$Class, lr_probs$malignant, levels = c("benign", "malignant"))

roc_df <- rbind(
  data.frame(
    FPR   = 1 - rf_roc$specificities,
    TPR   = rf_roc$sensitivities,
    Model = paste0("Random Forest (AUC = ", round(auc(rf_roc), 3), ")")
  ),
  data.frame(
    FPR   = 1 - lr_roc$specificities,
    TPR   = lr_roc$sensitivities,
    Model = paste0("Logistic Regression (AUC = ", round(auc(lr_roc), 3), ")")
  )
)

roc_plot <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.2) +
  geom_abline(linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("#E76F51", "#457B9D")) +
  labs(
    title    = "ROC Curves: Cancer Diagnosis Classification",
    subtitle = "Wisconsin Breast Cancer Dataset — Test Set Performance",
    x        = "False Positive Rate (1 - Specificity)",
    y        = "True Positive Rate (Sensitivity)",
    color    = "Model"
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title      = element_text(face = "bold"),
    legend.position = c(0.65, 0.2),
    legend.background = element_rect(fill = "white", color = "grey80")
  )

ggsave("roc_curves.png", roc_plot, width = 8, height = 6, dpi = 300)
cat("Saved: roc_curves.png\n")

# --- Plot 4: Feature Importance (Random Forest) ---
importance_df <- varImp(rf_model)$importance %>%
  tibble::rownames_to_column("Feature") %>%
  arrange(desc(Overall)) %>%
  head(10)

importance_plot <- ggplot(importance_df, aes(x = reorder(Feature, Overall), y = Overall, fill = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient(low = "#A8DADC", high = "#E63946") +
  labs(
    title    = "Top 10 Most Important Features",
    subtitle = "Random Forest Variable Importance",
    x        = "Feature",
    y        = "Importance Score",
    fill     = "Importance"
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold"),
    legend.position = "none"
  )

ggsave("feature_importance.png", importance_plot, width = 8, height = 6, dpi = 300)
cat("Saved: feature_importance.png\n")

# --- Plot 5: Confusion Matrix Heatmap ---
cm_df <- as.data.frame(rf_cm$table) %>%
  rename(Predicted = Prediction, Actual = Reference)

cm_plot <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 8, fontface = "bold") +
  scale_fill_gradient(low = "#F1FAEE", high = "#E63946") +
  labs(
    title    = "Confusion Matrix: Random Forest",
    subtitle = paste0("Accuracy: ", round(rf_cm$overall["Accuracy"] * 100, 1), "%"),
    x        = "Actual Diagnosis",
    y        = "Predicted Diagnosis",
    fill     = "Count"
  ) +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("confusion_matrix.png", cm_plot, width = 7, height = 6, dpi = 300)
cat("Saved: confusion_matrix.png\n")

# ============================================================
# PART 2: UNSUPERVISED CLUSTERING
# ============================================================

# Scale features for clustering
bc_scaled <- bc_clean %>%
  select(-Class) %>%
  scale() %>%
  as.data.frame()

# --- Optimal number of clusters ---
set.seed(42)
fviz_nbclust_data <- fviz_nbclust(bc_scaled, kmeans, method = "wss", k.max = 8)
ggsave("optimal_clusters.png", fviz_nbclust_data, width = 8, height = 6, dpi = 300)
cat("Saved: optimal_clusters.png\n")

# --- K-means Clustering (k=2, matching known biology) ---
set.seed(42)
km_result <- kmeans(bc_scaled, centers = 2, nstart = 25)

# PCA for visualization
pca_result <- prcomp(bc_scaled, scale. = FALSE)
pca_df <- data.frame(
  PC1      = pca_result$x[, 1],
  PC2      = pca_result$x[, 2],
  Cluster  = factor(km_result$cluster),
  TrueClass = bc_clean$Class
)

var_exp <- round(summary(pca_result)$importance[2, 1:2] * 100, 1)

# --- Plot 6: K-means Clustering vs True Labels ---
cluster_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster, shape = TrueClass)) +
  geom_point(size = 2.5, alpha = 0.7) +
  scale_color_manual(values = c("1" = "#2A9D8F", "2" = "#E76F51"),
                     labels = c("Cluster 1", "Cluster 2")) +
  scale_shape_manual(values = c("benign" = 16, "malignant" = 17),
                     labels = c("Benign", "Malignant")) +
  labs(
    title    = "K-means Clustering vs True Diagnosis",
    subtitle = "PCA visualization — Wisconsin Breast Cancer Dataset",
    x        = paste0("PC1 (", var_exp[1], "% variance)"),
    y        = paste0("PC2 (", var_exp[2], "% variance)"),
    color    = "K-means Cluster",
    shape    = "True Diagnosis"
  ) +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("kmeans_clustering.png", cluster_plot, width = 9, height = 6, dpi = 300)
cat("Saved: kmeans_clustering.png\n")

# --- Cluster vs True Label Agreement ---
cluster_table <- table(km_result$cluster, bc_clean$Class)
cat("\nCluster vs True Label Agreement:\n")
print(cluster_table)

# --- Save Model Performance Summary ---
results_summary <- data.frame(
  Model       = c("Random Forest", "Logistic Regression"),
  Accuracy    = c(round(rf_cm$overall["Accuracy"], 3),
                  round(lr_cm$overall["Accuracy"], 3)),
  Sensitivity = c(round(rf_cm$byClass["Sensitivity"], 3),
                  round(lr_cm$byClass["Sensitivity"], 3)),
  Specificity = c(round(rf_cm$byClass["Specificity"], 3),
                  round(lr_cm$byClass["Specificity"], 3)),
  AUC         = c(round(auc(rf_roc), 3),
                  round(auc(lr_roc), 3))
)

write.csv(results_summary, "model_performance_summary.csv", row.names = FALSE)
cat("\nModel Performance Summary:\n")
print(results_summary)
cat("\nSaved: model_performance_summary.csv\n")
cat("\nAnalysis complete. All outputs saved.\n")
