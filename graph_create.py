import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc


def graph_create(dataset, accuracies, precision_scores, recall_scores, f1_scores, conf_matrices, roc_auc_scores,
                 y_test_list, y_pred_prob_list):
    # Custom mapping of shorter names for columns
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

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Diabetes Dataset Visualization")

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set the window size to fill the entire screen
    root.geometry(f"{screen_width}x{screen_height}")

    # Placeholder for current displayed graph
    global current_plot

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

    # Function to clear other graphs frame
    def clear_other_graphs_frame():
        for widget in other_graphs_frame.winfo_children():
            widget.destroy()

    # Function to display Pairplot
    def show_pairplot():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()
        # Selecting a subset of features for pairplot visualization
        subset_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                           'DiabetesPedigreeFunction', 'Age', 'Outcome']
        subset_data = dataset[subset_features]

        # Renaming columns to shorten axis labels
        new_labels = {
            'Pregnancies': 'Preg',
            'Glucose': 'Glu',
            'BloodPressure': 'BP',
            'SkinThickness': 'Skin',
            'Insulin': 'Ins',
            'BMI': 'BMI',
            'DiabetesPedigreeFunction': 'DPF',
            'Age': 'Age',
            'Outcome': 'Outcome'
        }
        subset_data = subset_data.rename(columns=new_labels)

        # Adjusting subplot size for the pairplot
        pairplot = sns.pairplot(subset_data, hue='Outcome', height=0.85, aspect=1.8)
        pairplot.fig.suptitle('Pairplot of Selected Features')
        pairplot.fig.tight_layout()
        pairplot.fig.subplots_adjust(top=0.95)

        pairplot._legend.remove()

        # Custom legend with updated labels specifying short and long names
        custom_legend = plt.figure(figsize=(13, 1))
        handles = []
        labels = []
        # Custom legend with updated labels specifying short and long names
        for long_name, short_name in new_labels.items():
            if long_name != 'Outcome' and long_name != 'Age' and long_name != 'BMI':
                handles.append(plt.Line2D([0], [0], linestyle='none', marker='', label=short_name))
                labels.append(f"{short_name}: {long_name}")

        # Additional handles and labels for Outcome legend
        handles.extend([
            plt.Line2D([], [], linestyle='none', marker='o', markersize=5, color='dodgerblue', label='Outcome = 0'),
            plt.Line2D([], [], linestyle='none', marker='o', markersize=5, color='orange', label='Outcome = 1')
        ])
        labels.extend(['Outcome = 0', 'Outcome = 1'])

        plt.legend(handles, labels, loc='center', ncol=4, frameon=False)
        plt.axis('off')

        # Display the pairplot and the custom legend in the Tkinter frame
        pairplot_canvas = FigureCanvasTkAgg(pairplot.fig, master=other_graphs_frame)
        pairplot_canvas.draw()
        pairplot_canvas.get_tk_widget().pack()

        legend_canvas = FigureCanvasTkAgg(custom_legend, master=other_graphs_frame)
        legend_canvas.draw()
        legend_canvas.get_tk_widget().pack()

        current_plot = pairplot_canvas

    # Function to display Correlation Heatmap
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

    # Function to display Histograms/Density Plots
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

    # Function to display Count Plot for Target Variable
    def show_countplot():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()
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
        clear_other_graphs_frame()
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

    def show_precision_bar_plot():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(precision_scores)), precision_scores, color='skyblue', edgecolor='black')

        for i, prec in enumerate(precision_scores):
            plt.text(i, prec + 0.01, f'{prec:.3f}', ha='center')

        plt.xlabel('Fold')
        plt.ylabel('Precision')
        plt.title('Precision for each Fold')
        plt.ylim(0, 1)
        plt.xticks(range(len(precision_scores)), [f'Fold {i + 1}' for i in range(len(precision_scores))])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        precision_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
        precision_canvas.draw()
        precision_canvas.get_tk_widget().pack()
        current_plot = precision_canvas

    def show_recall_bar_plot():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(recall_scores)), recall_scores, color='skyblue', edgecolor='black')

        for i, rec in enumerate(recall_scores):
            plt.text(i, rec + 0.01, f'{rec:.3f}', ha='center')

        plt.xlabel('Fold')
        plt.ylabel('Recall')
        plt.title('Recall for each Fold')
        plt.ylim(0, 1)
        plt.xticks(range(len(recall_scores)), [f'Fold {i + 1}' for i in range(len(recall_scores))])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        recall_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
        recall_canvas.draw()
        recall_canvas.get_tk_widget().pack()
        current_plot = recall_canvas

    def show_f1_score_bar_plot():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(f1_scores)), f1_scores, color='skyblue', edgecolor='black')

        for i, f1 in enumerate(f1_scores):
            plt.text(i, f1 + 0.01, f'{f1:.3f}', ha='center')

        plt.xlabel('Fold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score for each Fold')
        plt.ylim(0, 1)
        plt.xticks(range(len(f1_scores)), [f'Fold {i + 1}' for i in range(len(f1_scores))])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        f1_score_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
        f1_score_canvas.draw()
        f1_score_canvas.get_tk_widget().pack()
        current_plot = f1_score_canvas

    def show_roc_auc_bar_plot():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(roc_auc_scores)), roc_auc_scores, color='skyblue', edgecolor='black')

        for i, roc_auc in enumerate(roc_auc_scores):
            plt.text(i, roc_auc + 0.01, f'{roc_auc:.3f}', ha='center')

        plt.xlabel('Fold')
        plt.ylabel('ROC AUC')
        plt.title('ROC AUC for each Fold')
        plt.ylim(0, 1)
        plt.xticks(range(len(roc_auc_scores)), [f'Fold {i + 1}' for i in range(len(roc_auc_scores))])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        roc_auc_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
        roc_auc_canvas.draw()
        roc_auc_canvas.get_tk_widget().pack()
        current_plot = roc_auc_canvas

    def plot_roc_curve(y_true, y_score, label):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        dense_fpr = np.linspace(0, 1, 100)

        # Use interp1d for smoother interpolation
        interp_func = interp1d(fpr, tpr, kind='linear')
        interp_tpr = interp_func(dense_fpr)

        plt.plot(dense_fpr, interp_tpr, lw=1.5, label=f'{label} (AUC = {roc_auc:.2f})')

    def show_all_roc_auc_curves():
        global current_plot
        clear_plot()
        clear_other_graphs_frame()

        plt.figure(figsize=(8, 6))

        # Loop through each fold to generate and plot its ROC AUC curve
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

        # Create a canvas to embed the ROC AUC curves
        roc_auc_canvas = FigureCanvasTkAgg(plt.gcf(), master=other_graphs_frame)
        roc_auc_canvas.get_tk_widget().pack()

        # Redraw the canvas with all ROC AUC curves
        roc_auc_canvas.draw()
        current_plot = roc_auc_canvas

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

    precision_bar_plot_button = tk.Button(button_frame, text="Precision Bar Plot", command=show_precision_bar_plot)
    precision_bar_plot_button.pack(side=tk.LEFT, padx=5, pady=5)

    recall_bar_plot_button = tk.Button(button_frame, text="Recall Bar Plot", command=show_recall_bar_plot)
    recall_bar_plot_button.pack(side=tk.LEFT, padx=5, pady=5)

    f1_score_bar_plot_button = tk.Button(button_frame, text="F1 Score Bar Plot", command=show_f1_score_bar_plot)
    f1_score_bar_plot_button.pack(side=tk.LEFT, padx=5, pady=5)

    roc_auc_bar_plot_button = tk.Button(button_frame, text="ROC AUC Bar Plot", command=show_roc_auc_bar_plot)
    roc_auc_bar_plot_button.pack(side=tk.LEFT, padx=5, pady=5)

    show_all_roc_auc_button = tk.Button(button_frame, text="Show All ROC AUC Curves", command=show_all_roc_auc_curves)
    show_all_roc_auc_button.pack(side=tk.LEFT, padx=5, pady=5)

    # Run the Tkinter main loop
    root.mainloop()
