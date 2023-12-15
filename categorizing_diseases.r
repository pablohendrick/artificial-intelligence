library(readr)
library(dplyr)
library(tidyr)
library(cluster)
library(randomForest)
library(glmnet)
library(caret)

data <- read.csv('diseases_list.csv', header = TRUE) 

head(data)
dim(data)
summary(data)
str(data)
colSums(is.na(data))

encoded_data <- data %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, ~as.numeric(.))

causes <- encoded_data[, c("Cause_Column")] 
causes_clusters <- kmeans(encoded_data[, -c("Cause_Column")], centers = 6)

symptoms <- encoded_data[, -(c("Cause_Column", "Disease_Column"))]
symptoms_clusters <- kmeans(symptoms, centers = 5)

print(causes_clusters)
print(symptoms_clusters)

rf_model <- randomForest(Mortality_Rate_Column ~ ., data = data, ntree = 100)

logit_model <- glm(Mortality_Rate_Column ~ ., data = data, family = "binomial")

set.seed(123) 
train_indices <- createDataPartition(data$Mortality_Rate_Column, p = 0.7, list = FALSE)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

trained_rf_model <- randomForest(Mortality_Rate_Column ~ ., data = train_data, ntree = 100)

rf_predictions <- predict(trained_rf_model, newdata = test_data)

accuracy <- confusionMatrix(rf_predictions, test_data$Mortality_Rate_Column)$overall['Accuracy']

high_mortality <- ifelse(test_data$Mortality_Rate_Column > threshold, 1, 0)
recall <- confusionMatrix(rf_predictions, high_mortality)$byClass['Recall']

print(paste("Accuracy:", accuracy))
print(paste("Recall:", recall))

set.seed(123)
control <- trainControl(method = "cv", number = 5)
cv_model <- train(Mortality_Rate_Column ~ ., data = data, method = "rf", trControl = control)

print(cv_model)

metrics <- cv_model$results[, c("Accuracy", "Recall")]
mean_accuracy <- mean(metrics$Accuracy)
mean_recall <- mean(metrics$Recall)

print(paste("Mean Accuracy in Cross Validation:", mean_accuracy))
print(paste("Mean Recall in Cross Validation:", mean_recall))

for (i in unique(data$cluster_column)) {
  cluster_data <- subset(data, cluster_column == i)
  ggplot(cluster_data, aes(x = Attribute1, y = Attribute2, color = as.factor(cluster_column))) +
    geom_point() +
    labs(title = paste("Cluster", i), x = "Attribute1", y = "Attribute2") +
    theme_minimal()
}
