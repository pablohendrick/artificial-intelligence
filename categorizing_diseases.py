import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the CSV file
data = pd.read_csv('diseases_list.csv')

# Displaying the first few rows of the DataFrame
print(data.head())

# Displaying the dimensions of the DataFrame
print(data.shape)

# Displaying summary statistics
print(data.describe())

# Displaying information about the DataFrame
print(data.info())

# Checking for missing values
print(data.isnull().sum())

# Encoding categorical variables and converting factors to numeric
encoded_data = data.copy()
encoded_data = encoded_data.apply(lambda x: pd.factorize(x)[0] if pd.api.types.is_categorical_dtype(x) else x)

# Clustering causes using k-means
causes = encoded_data["Cause_Column"]
causes_clusters = KMeans(n_clusters=6).fit(encoded_data.drop(["Cause_Column"], axis=1))

# Clustering symptoms using k-means
symptoms = encoded_data.drop(["Cause_Column", "Disease_Column"], axis=1)
symptoms_clusters = KMeans(n_clusters=5).fit(symptoms)

# Displaying cluster information
print(causes_clusters.labels_)
print(symptoms_clusters.labels_)

# Building a random forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(encoded_data.drop(["Mortality_Rate_Column"], axis=1), encoded_data["Mortality_Rate_Column"])

# Building a logistic regression model
logit_model = LogisticRegression()
logit_model.fit(encoded_data.drop(["Mortality_Rate_Column"], axis=1), encoded_data["Mortality_Rate_Column"])

# Splitting the data into training and testing sets
np.random.seed(123)
train_data, test_data = train_test_split(data, test_size=0.3)

# Training a random forest model on the training data
trained_rf_model = RandomForestClassifier(n_estimators=100)
trained_rf_model.fit(train_data.drop(["Mortality_Rate_Column"], axis=1), train_data["Mortality_Rate_Column"])

# Making predictions on the test data
rf_predictions = trained_rf_model.predict(test_data.drop(["Mortality_Rate_Column"], axis=1))

# Calculating accuracy and recall
accuracy = accuracy_score(test_data["Mortality_Rate_Column"], rf_predictions)
threshold = np.mean(test_data["Mortality_Rate_Column"])
high_mortality = (test_data["Mortality_Rate_Column"] > threshold).astype(int)
recall = recall_score(high_mortality, rf_predictions)

print("Accuracy:", accuracy)
print("Recall:", recall)

# Cross-validation using random forest
cv_model = RandomForestClassifier(n_estimators=100)
cv_scores = cross_val_score(cv_model, data.drop(["Mortality_Rate_Column"], axis=1), data["Mortality_Rate_Column"], cv=5, scoring="accuracy")

print("Cross-Validation Accuracy:", np.mean(cv_scores))

# Clustering visualization
unique_clusters = data["cluster_column"].unique()

for i in unique_clusters:
    cluster_data = data[data["cluster_column"] == i]
    plt.figure()
    sns.scatterplot(data=cluster_data, x="Attribute1", y="Attribute2", hue="cluster_column")
    plt.title(f"Cluster {i}")
    plt.xlabel("Attribute1")
    plt.ylabel("Attribute2")
    plt.show()

