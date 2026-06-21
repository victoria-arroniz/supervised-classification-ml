# ==============================================================================
# ==============================================================================
# ===                                                                        ===
# ===                     Statistical Machine Learning                       ===
# ===                                                                        ===
# ===                   Victoria Arróniz Soriano 25221241                    ===
# ===                                                                        ===
# ==============================================================================
# ==============================================================================

# Packages
library(caret)
library(randomForest)
library(kernlab)
library(doParallel)
library(ROCR)

# ==============================================================================
# Inspection of the data
# ==============================================================================

load("data_assignment.RData")

# Missing values 
colSums(is.na(data))

# Initial inspection od the data
str(data)
dim(data)       # number of rows and columns
dim(test)

# Class distribution 
table(data$y)
prop.table(table(data$y))
# If the classes are highly imbalanced we should use AU-ROC (not accuracy)
# as the main metric, as accuracy can be misleading.


# Stablish y as a factor with pos as the positive class 
# caret's twoClassSummary treats the FIRST level as the positive class.
# We set pos as the first level so AU-ROC is computed correctly.
data$y <- factor(data$y, levels = c("pos", "neg"))
table(data$y)


# ==============================================================================
# Cross-validation
# ==============================================================================
# We use repeated k-fold cross-validation (5 folds, 10 repeats = 50 resamples)
# to tune and compare models.
# Performance metric: AU-ROC (area under the ROC curve)..

train_ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,       # 5 folds
  repeats = 10,     # 10 repetitions -> 50 resamples total
  summaryFunction = twoClassSummary,
  classProbs = TRUE,         # needed for AU-ROC
  savePredictions = "final"  # store fold-level predictions
)

# Enable parallel computing to speed up tuning
cl <- makeCluster(6) # My computer has 8 cores
registerDoParallel(cl)


# ==============================================================================
# Models
# ==============================================================================


# --- Model 1 – Logistic Regression with stepwise AIC selection ----------------

set.seed(123)
fit_lr <- train( y ~ ., data = data,
  method = "glmStepAIC",   # logistic regression + both AIC selection
  family = "binomial",
  direction = "both",
  trControl = train_ctrl,
  preProcess = c("center", "scale"),
  metric = "ROC",
  trace = FALSE           # suppress AIC step output
)

# Selected variables
print(summary(fit_lr$finalModel))

# --- Model 2 – SVM with RBF kernel --------------------------------------------

tune_grid_svm <- expand.grid(
  C = c(0.1,0.5, 1, 5, 10, 50),
  sigma = c(0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75)
)

set.seed(123) # same seed

fit_svm <- train( y ~ ., data = data,
  method = "svmRadial",
  trControl = train_ctrl,
  tuneGrid = tune_grid_svm,
  preProcess = c("center", "scale"),
  metric = "ROC"
)

# SVM RBF: best hyperparameters
best_C <- fit_svm$bestTune$C
best_sigma <- fit_svm$bestTune$sigma
print(fit_svm$bestTune)

# Best AU-ROC
best_roc <- max(fit_svm$results$ROC)
print(best_roc)

# Plot 
plot(fit_svm,
     main = "SVM RBF: tuning grid",
     xlab = "Cost (C)",
     ylab = "Mean CV AU-ROC",
     lwd = 2, pch = 16, cex = 1.2)


# --- Model 3 – Random Forest --------------------------------------------------

ntree_grid <- c(100, 200, 500)

rf_results <- vector("list", length(ntree_grid))


for (j in seq_along(ntree_grid)) {
  set.seed(123)   # same seed
  rf_results[[j]] <- train( y ~ ., data = data,
                            method = "rf",
                            trControl = train_ctrl,
                            tuneGrid = expand.grid(
                              mtry = c(2, 4, 5, 6, 8, 10, 12)),
                            metric = "ROC",
                            ntree = ntree_grid[j])
}
stopCluster(cl)

#  Best model
rf_mean_roc <- sapply(rf_results, function(x) max(x$results$ROC))
best_ntree_idx <- which.max(rf_mean_roc)
fit_rf <- rf_results[[best_ntree_idx]]

# Best ntree 
print(ntree_grid[best_ntree_idx])
# Best mtry
print(fit_rf$bestTune)

#  AU-ROC table
roc <- sapply(rf_results, function(x) x$results$ROC)
colnames(roc) <- paste0("ntree=", ntree_grid)
mtry_grid <- rf_results[[1]]$results$mtry
roc <- cbind(mtry = mtry_grid, as.data.frame(roc))
print(round(roc, 4))

# Plot
colors <- c("purple2", "forestgreen", "darkorange3")  # one per ntree value
matplot(roc$mtry, roc[, -1],
        type = "b", pch = 19, lwd = 2, lty = 1,
        col = colors,
        xlab = "mtry",
        ylab = "Mean CV AU-ROC",
        main = "Random Forest: tuning grid")
grid()
legend("bottomright",
       legend = paste("ntree =", ntree_grid),
       col = colors,
       lwd = 2,
       bty = "n")

# Variable importance plot
varImpPlot(fit_rf$finalModel,
           main = "Variable Importance (Random Forest)",
           n.var = 26,
           cex = 1.8,
           col = "forestgreen",
           pch = 19)



# ==============================================================================
#  Model comparison
# ==============================================================================

model_comparison <- resamples(list(
  LogReg = fit_lr,
  SVM = fit_svm,
  RF = fit_rf
))

# CV AU-ROC table 
cv_summary <- summary(model_comparison)$statistics$ROC
print(round(cv_summary, 4))

# Model comparison summary (AU-ROC)
print(summary(model_comparison))

cols <- c("darkorange3", "purple3", "forestgreen")
roc_values <- model_comparison$values[, c(2, 5, 8)]
sens_values <- model_comparison$values[, c(3, 6, 9)]
spec_values <- model_comparison$values[, c(4, 7, 10)]

par(mfrow = c(1, 3), mar = c(6, 6, 5, 2))
# Boxplot AU-ROC
boxplot(roc_values,
        col = adjustcolor(cols, 0.4),
        border = cols,
        names  = c("Log. Reg.", "SVM-RBF", "RF"),
        ylab = "AU-ROC",
        main = "Cross-validated AU-ROC") 
points(1:3, colMeans(roc_values), pch = 15, cex = 1.5, col = cols)
# Boxplot Sensitivity
boxplot(sens_values,
        col = adjustcolor(cols, 0.4),
        border = cols,
        names  = c("Log. Reg.", "SVM-RBF", "RF"),
        ylab = "Sensitivity",
        main = "Cross-validated Sensitivity") 
points(1:3, colMeans(sens_values), pch = 15, cex = 1.5, col = cols)
# Boxplot Specificity
boxplot(spec_values,
        col = adjustcolor(cols, 0.4),
        border = cols,
        names  = c("Log. Reg.", "SVM-RBF", "RF"),
        ylab = "Specificity",
        main = "Cross-validated Specificity") 
points(1:3, colMeans(spec_values), pch = 15, cex = 1.5, col = cols)

par(mfrow = c(1, 1), mar = c(5, 4, 4, 2)) 

# Final model: Random Forest (best CV AU-ROC)
final_fit <- fit_rf
print(final_fit$bestTune)

# ==============================================================================
# Predict on test data
# ==============================================================================

#   y_hat -> predicted class labels (pos / neg), length = nrow(test)
#   p_hat -> predicted probabilities of class "pos", length = nrow(test)

# Predicted class labels
y_hat <- predict(final_fit, newdata = test, type = "raw")

# Predicted probabilities for class "pos"
p_hat_matrix <- predict(final_fit, newdata = test, type = "prob")
p_hat <- p_hat_matrix[, "pos"]
summary(p_hat)
table(y_hat)

prop.table(table(y_hat))
prop.table(table(data$y))

# Save student id
save(y_hat, p_hat, file = "25221241.RData")

