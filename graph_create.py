import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
import pandas as pd


def graph_create(dataset, accuracies, precision_scores, recall_scores, f1_scores, roc_auc_scores,
                 y_test_list, y_pred_prob_list):
    short_names = {
        'Pregnancies': 'Preg',
        'Glucose': 'Gluc',
        'BloodPressure': 'BP',
        'SkinThickness': 'Skin',
        'Insulin': 'Ins',
        'BMI': 'BMI',
        'DiabetesPedigreeFunction': 'DPF',
        'Age': 'Age',
        'Outcome': 'Outcome'
    }

    root = tk.Tk()
    root.title("Diabetes Dataset Visualization")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    root.geometry(f"{screen_width}x{screen_height}")

    global current_plot

    button_frame = tk.Frame(root)
    button_frame.pack()

    other_graphs_frame = tk.Frame(root)
    other_graphs_frame.pack()

    def clear_plot():
        global current_plot
        if current_plot:
            current_plot.get_tk_widget().pack_forget()

    current_plot = None

    def clear_plot():
        global current_plot
        if current_plot:
            current_plot.get_tk_widget().pack_forget()

    def clear_other_graphs_frame():
        for widget in other_graphs_frame.winfo_children():
            widget.destroy()

    def show_heatmap():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()
        plt.figure(figsize=(15, 7))
        heatmap = sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f',
                              xticklabels=[short_names[col] for col in dataset.columns],
                              yticklabels=[short_names[col] for col in dataset.columns])
        plt.text(10.5, 1,
                 'Preg: Pregnancies\nGluc: Glucose\nBP: BloodPressure\nSkin: SkinThickness\nIns: Insulin\nBMI: '
                 'BMI\nDPF:'
                 'DiabetesPedigreeFunction\nAge: Age\nOutcome: Outcome',
                 fontsize=10, va='top')
        plt.title('Correlation Heatmap')
        heatmap_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
        heatmap_canvas.draw()
        heatmap_canvas.get_tk_widget().pack()
        current_plot = heatmap_canvas

    def show_histograms():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()
        plt.figure(figsize=(9, 7))
        for i, column in enumerate(dataset.columns[:-1]):
            plt.subplot(3, 3, i + 1)
            sns.histplot(dataset[column], kde=True)
            plt.title(f'{column} Distribution')
        plt.tight_layout()
        histograms_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
        histograms_canvas.draw()
        histograms_canvas.get_tk_widget().pack()
        current_plot = histograms_canvas

    def show_all_metrics_heatmap():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()

        metrics_df = pd.DataFrame({
            'Accuracy': accuracies,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1 Score': f1_scores,
            'ROC AUC': roc_auc_scores
        })

        best_values = metrics_df.max()
        worst_values = metrics_df.min()
        average_values = metrics_df.mean()

        summary_df = pd.DataFrame({
            'Best Value': best_values,
            'Average Value': average_values,
            'Worst Value': worst_values
        })

        plt.figure(figsize=(12, 8))
        sns.heatmap(summary_df.T, annot=True, cmap='YlGnBu', fmt='.3f', cbar=True, linewidths=.5)
        plt.title('Metrics Summary - Best, Worst, and Average Values')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        metrics_heatmap_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
        metrics_heatmap_canvas.draw()
        metrics_heatmap_canvas.get_tk_widget().pack()

        current_plot = metrics_heatmap_canvas

    def plot_roc_curve(y_true, y_score, label):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        dense_fpr = np.linspace(0, 1, 100)

        interp_func = interp1d(fpr, tpr, kind='linear')
        interp_tpr = interp_func(dense_fpr)

        plt.plot(dense_fpr, interp_tpr, lw=1.5, label=f'{label} (AUC = {roc_auc:.2f})')

    def show_all_roc_auc_curves():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()

        plt.figure(figsize=(8, 6))

        for fold_index in range(len(roc_auc_scores)):
            y_test = y_test_list[fold_index]
            y_pred_prob = y_pred_prob_list[fold_index]

            plot_roc_curve(y_test, y_pred_prob, f'Fold {fold_index + 1}')

        plt.plot([0, 1], [0, 1], color='red', lw=1.5, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC AUC Curves for all Folds')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()

        roc_auc_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
        roc_auc_canvas.get_tk_widget().pack()

        roc_auc_canvas.draw()
        current_plot = roc_auc_canvas

    heatmap_button = tk.Button(button_frame, text="Heatmap", command=show_heatmap)
    heatmap_button.pack(side=tk.LEFT, padx=5, pady=5)

    histograms_button = tk.Button(button_frame, text="Histograms", command=show_histograms)
    histograms_button.pack(side=tk.LEFT, padx=5, pady=5)

    show_all_roc_auc_button = tk.Button(button_frame, text="Show All ROC AUC Curves", command=show_all_roc_auc_curves)
    show_all_roc_auc_button.pack(side=tk.LEFT, padx=5, pady=5)

    show_all_metrics_heatmap_button = tk.Button(button_frame, text="Show All Metrics Heatmap",
                                                command=show_all_metrics_heatmap)
    show_all_metrics_heatmap_button.pack(side=tk.LEFT, padx=5, pady=5)

    root.mainloop()
