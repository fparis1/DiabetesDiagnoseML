import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the column names
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Read the CSV file into a Pandas DataFrame
dataset = pd.read_csv('diabetes.csv', header=0, names=names)

# Split the dataset into features (X) and target (y)
X = dataset.iloc[:, :-1].values  # Features (all columns except 'Outcome')
y = dataset['Outcome'].values    # Target (the 'Outcome' column)

# Split the data into a training set (70%) and a testing set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a neural network model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),  # Input layer with 8 features
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 units and ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 unit and sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model on the testing data
y_pred = (model.predict(X_test) > 0.5).astype(int)  # Convert probabilities to 0 or 1
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
