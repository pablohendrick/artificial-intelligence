import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load datasets for each season (assuming files are named 'spring_fruits.csv', 'summer_fruits.csv', 'autumn_fruits.csv', and 'winter_fruits.csv')
spring_fruits = pd.read_csv('spring_fruits.csv')
summer_fruits = pd.read_csv('summer_fruits.csv')
autumn_fruits = pd.read_csv('autumn_fruits.csv')
winter_fruits = pd.read_csv('winter_fruits.csv')

spring_fruits['Season'] = 'Spring'
summer_fruits['Season'] = 'Summer'
autumn_fruits['Season'] = 'Autumn'
winter_fruits['Season'] = 'Winter'

fruits = pd.concat([spring_fruits, summer_fruits, autumn_fruits, winter_fruits], ignore_index=True)

print(fruits.head())

X = fruits['Fruit']
y = fruits['Season']

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

print(classification_report(y_test, y_pred))

# Assuming you have data about the planting periods of each fruit for each season in an appropriate format

# Here's a simple example for demonstration purposes
planting_period_data = {
    'Fruit': ['Apple', 'Orange', 'Banana', 'Strawberry'],
    'Spring': [30, 60, 45, 50],
    'Summer': [20, 40, 35, 45],
    'Autumn': [35, 55, 40, 60],
    'Winter': [25, 50, 30, 55]
}

df_planting_period = pd.DataFrame(planting_period_data)

seasons = ['Spring', 'Summer', 'Autumn', 'Winter']

for season in seasons:
    plt.figure(figsize=(8, 6))
    plt.bar(df_planting_period['Fruit'], df_planting_period[season], color='skyblue')
    plt.title(f'Planting periods of fruits in {season}')
    plt.xlabel('Fruit')
    plt.ylabel('Planting period (days)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
