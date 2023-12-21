import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Define the column names
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
         'Age', 'Outcome']

# Read the CSV file into a Pandas DataFrame
dataset = pd.read_csv('diabetes.csv', header=0, names=names)

# Split the dataset into features (X) and target (y)
X = dataset.iloc[:, :-1].values  # Features (all columns except 'Outcome')
y = dataset['Outcome'].values  # Target (the 'Outcome' column)

# Impute missing values with mean
imputer = SimpleImputer(missing_values=0, strategy='mean')
X = imputer.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize StratifiedKFold with 10 splits
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []  # List to store accuracies of each fold

# Iterate through each fold
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]  # Features split
    y_train, y_test = y[train_index], y[test_index]  # Target split

    # Create a neural network model using TensorFlow
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),  # Input layer with 8 features
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        tf.keras.layers.Dropout(0.4),  # Dropout layer to prevent overfitting
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 unit and sigmoid activation
    ])

    # Implement learning rate scheduling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile the model with weighted binary cross-entropy
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model on the training data for 20 epochs
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)

    # Evaluate the model on the testing data
    _, accuracy = model.evaluate(X_test, y_test)
    accuracies.append(accuracy)

# Calculate and print the average accuracy across all folds
avg_accuracy = np.mean(accuracies)
print(f'Average Accuracy: {avg_accuracy * 100:.2f}%')

# Create a Tkinter window
root = tk.Tk()
root.title("Diabetes Dataset Visualization")

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size to fill the entire screen
root.geometry(f"{screen_width}x{screen_height}")

# Placeholder for current displayed graph
current_plot = None

# Frame to contain the buttons for graph selection
button_frame = tk.Frame(root)
button_frame.pack()

# Frame to contain other types of graphs
other_graphs_frame = tk.Frame(root)
other_graphs_frame.pack()


# Function to clear the current plot
def clear_plot():
    global current_plot
    if current_plot:
        current_plot.get_tk_widget().pack_forget()


# Placeholder for current displayed graph
current_plot = None


# Function to clear the current plot
def clear_plot():
    global current_plot
    if current_plot:
        current_plot.get_tk_widget().pack_forget()


# Function to display Pairplot
def show_pairplot():
    global current_plot
    clear_plot()
    pairplot = sns.pairplot(dataset, hue='Outcome')
    pairplot.fig.suptitle('Pairplot of Features')
    pairplot.fig.tight_layout()
    pairplot.fig.subplots_adjust(top=0.95)
    pairplot_canvas = FigureCanvasTkAgg(pairplot.fig, master=other_graphs_frame)
    pairplot_canvas.draw()
    pairplot_canvas.get_tk_widget().pack()
    current_plot = pairplot_canvas


# Function to display Correlation Heatmap
def show_heatmap():
    global current_plot
    clear_plot()
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    heatmap_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
    heatmap_canvas.draw()
    heatmap_canvas.get_tk_widget().pack()
    current_plot = heatmap_canvas


# Function to display Histograms/Density Plots
def show_histograms():
    global current_plot
    clear_plot()
    plt.figure(figsize=(10, 8))
    for i, column in enumerate(dataset.columns[:-1]):
        plt.subplot(3, 3, i + 1)
        sns.histplot(dataset[column], kde=True)
        plt.title(f'{column} Distribution')
    plt.tight_layout()
    histograms_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
    histograms_canvas.draw()
    histograms_canvas.get_tk_widget().pack()
    current_plot = histograms_canvas


# Function to display Count Plot for Target Variable
def show_countplot():
    global current_plot
    clear_plot()
    plt.figure()
    countplot = sns.countplot(x='Outcome', data=dataset)
    plt.title('Distribution of Outcome')
    countplot_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
    countplot_canvas.draw()
    countplot_canvas.get_tk_widget().pack()
    current_plot = countplot_canvas


# Function to display the Accuracy Bar Plot
def show_accuracy_bar_plot():
    global current_plot
    clear_plot()

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(accuracies)), accuracies, color='skyblue', edgecolor='black')

    # Adding text labels with accuracies
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')

    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for each Fold')
    plt.ylim(0, 1)  # Set y-axis limit between 0 and 1 for accuracy
    plt.xticks(range(len(accuracies)), [f'Fold {i + 1}' for i in range(len(accuracies))])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Embed the plot in a Tkinter frame
    accuracy_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
    accuracy_canvas.draw()
    accuracy_canvas.get_tk_widget().pack()
    current_plot = accuracy_canvas


# Buttons to trigger graph displays
pairplot_button = tk.Button(button_frame, text="Pairplot", command=show_pairplot)
pairplot_button.pack(side=tk.LEFT, padx=5, pady=5)

heatmap_button = tk.Button(button_frame, text="Heatmap", command=show_heatmap)
heatmap_button.pack(side=tk.LEFT, padx=5, pady=5)

histograms_button = tk.Button(button_frame, text="Histograms", command=show_histograms)
histograms_button.pack(side=tk.LEFT, padx=5, pady=5)

countplot_button = tk.Button(button_frame, text="Count Plot", command=show_countplot)
countplot_button.pack(side=tk.LEFT, padx=5, pady=5)

accuracy_bar_plot_button = tk.Button(button_frame, text="Accuracy Bar Plot", command=show_accuracy_bar_plot)
accuracy_bar_plot_button.pack(side=tk.LEFT, padx=5, pady=5)

# Run the Tkinter main loop
root.mainloop()
