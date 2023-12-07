import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.activations import linear, sigmoid, softmax, relu
from keras.optimizers import SGD
import matplotlib.pyplot as plt

# Data
file_name = 'path/to/data/file.xlsx'
data = pd.read_excel(file_name)

# Check for missing values
print("Missing values before preprocessing:")
print(data.isnull().sum())

# Remove rows with missing values
data_without_nulls = data.dropna()

# Detecting and removing outliers
outlier_detector = IsolationForest(contamination=0.1)
outliers = outlier_detector.fit_predict(data_without_nulls)
data_without_outliers = data_without_nulls[outliers != -1]

# Standardizing values
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data_without_outliers)

# Convert back to Pandas DataFrame
standardized_data = pd.DataFrame(standardized_data, columns=data_without_outliers.columns)

# Check for missing values after preprocessing
print("\nMissing values after preprocessing:")
print(standardized_data.isnull().sum())

# Splitting data into feature set (X) and target (y)
X = standardized_data.drop('target_column', axis=1)
y = standardized_data['target_column']

# Splitting data into training (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Size of sets:")
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# Model definition using Sequential
model = Sequential()

# Adding recurrent RNN layers
timesteps =  0
features =  0
for i in range(8):
    if i == 0:
        model.add(SimpleRNN(units=24, activation=linear, return_sequences=True, input_shape=(timesteps, features)))
    elif i == 7:
        model.add(SimpleRNN(units=24, activation=relu))
    else:
        model.add(SimpleRNN(units=24, activation=sigmoid, return_sequences=True))

# Output layer with softmax activation
model.add(Dense(units=1, activation=softmax))

# Compiling the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Adjust loss function and metrics as needed

# Display model summary
model.summary()

# Compiling the model with SGD and metrics
sgd = SGD(lr=0.01, momentum=0.9) 
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae', 'mse'])

# Model training
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluating the model on the test set
test_results = model.evaluate(X_test, y_test)
print("Loss on test set:", test_results[0])
print("MAE on test set:", test_results[1])
print("MSE on test set:", test_results[2])

# Plotting training and validation metrics across epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
