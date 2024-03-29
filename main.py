import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from graph_create import graph_create

names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
         'Age', 'Outcome']

dataset = pd.read_csv('diabetes.csv', header=0, names=names)

X = dataset.iloc[:, :-1].values
y = dataset['Outcome'].values

imputer = SimpleImputer(missing_values=0, strategy='mean')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
conf_matrices = []
roc_auc_scores = []
y_test_list = []
y_pred_prob_list = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.7)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=150, batch_size=50, verbose=0)

    _, accuracy = model.evaluate(X_test, y_test)
    accuracy_scores.append(accuracy)

    y_pred_prob = model.predict(X_test)
    # y_pred = (y_pred_prob > 0.5).astype("int32")

    y_pred = [y_element > 0.4 for y_element in y_pred_prob]

    y_test_list.append(y_test)
    y_pred_prob_list.append(y_pred_prob)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    roc_auc_scores.append(roc_auc)

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    conf_matrices.append(conf_matrix)

avg_accuracy = np.mean(accuracy_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)
avg_roc_auc = np.mean(roc_auc_scores)

print(f'Average Accuracy: {avg_accuracy * 100:.2f}%')
print(f'Average Precision: {avg_precision * 100:.2f}%')
print(f'Average Recall: {avg_recall * 100:.2f}%')
print(f'Average F1 Score: {avg_f1 * 100:.2f}%')
print(f'Average ROC AUC: {avg_roc_auc:.2f}')

graph_create(dataset, accuracy_scores, precision_scores, recall_scores,
             f1_scores, roc_auc_scores, y_test_list, y_pred_prob_list)
