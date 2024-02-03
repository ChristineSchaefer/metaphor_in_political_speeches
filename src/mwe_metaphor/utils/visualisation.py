import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.config import BASE_DIR
from src.mwe_metaphor.models.dataset_model import Dataset
from src.utils.datetime import ts_now


def process_and_chart(dataset: Dataset, dataset_name: str, tokenized_inputs):
    """
        Utility function to remove special label in tokenized inputs from BERT
        to create bar chart.

        @param dataset: Dataset object
        @param dataset_name: name of the Dataset
        @param tokenized_inputs: input from BERT Tokenizer
    """
    new_labels = [
        [dataset.labels[l] for l in label if l != -100]
        for label in tokenized_inputs["labels"]
    ]
    create_bar_chart_for_label_representation(dataset_name, dataset.labels, new_labels)


def get_number_of_label_element(labels_of_word: list[list], labels: list):
    """
        Compute and return the count of each unique label present in the input list.

        @param labels_of_word: A list that consists of labels where each label is an int
        @param labels: A list with the string representations of the labels

        @returns dictionary where the keys are unique labels from the input list and the values are corresponding counts of those labels in the list
        """
    label_counts = {}
    for l in labels_of_word:
        for label in l:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
    for l in labels:
        if l not in label_counts:
            label_counts[l] = 0
    return label_counts


def create_bar_chart_for_label_representation(name: str, labels: list, labels_of_word: list):
    """
        Create a bar chart visualization for the label representation in 'labels_of_word'
        and save it in file.

        @param name: A string representing the name of the visualization
        @param labels: A list representing different labels
        @param labels_of_word: A list of lists where each sub-list consists of labels
    """
    label_counts = get_number_of_label_element(labels_of_word, labels)
    counts = list(label_counts.values())

    # Create a bar chart
    plt.bar(labels, counts)
    plt.xlabel('labels')
    plt.ylabel('count')
    plt.title(name)
    plt.savefig(os.path.join(BASE_DIR, f"data/plots/bar/{ts_now()}_{name}.png"))
    plt.close()


def create_confusion_matrix_for_prediction(name: str, labels: list, predictions: list, true_labels: list):
    """
        Create a confusion matrix for model predictions and save it in file.

        @param name: A string representing the name of the confusion matrix
        @param labels: A list representing different labels
        @param predictions: A list representing model predictions
        @param true_labels: A list representing true labels
    """

    ticks = [_ for _ in range(len(labels))]
    flat_predictions = np.concatenate(predictions)
    flat_true_labels = np.concatenate(true_labels)
    cm = confusion_matrix(y_pred=flat_predictions, y_true=flat_true_labels, labels=labels)
    print(cm)
    plt.figure(figsize=(16, 12))
    sns.set(font_scale=1.4, color_codes=True, palette="deep")
    sns.heatmap(pd.DataFrame(cm, index=labels, columns=ticks),
                annot=True,
                annot_kws={"size": 16},
                fmt="d",
                cmap="YlGnBu")
    plt.title(f"confusion matrix {name}")
    plt.xlabel("predicted value")
    plt.xticks(ticks, labels, rotation=45)
    plt.ylabel("true value")
    plt.xticks(ticks, labels)
    plt.savefig(os.path.join(BASE_DIR, f"data/plots/confusion_matrix/{ts_now()}_{name}.png"))
    plt.close()


def plot_history(history, name):
    """
    Plot the training history of a model and save it in file.

    @param history: A history object that includes 'eval_accuracy', 'eval_loss', 'eval_f1',
                    'eval_recall', and 'eval_precision' metrics' history over epochs
    @param name: A string representing the name of the plot
    """
    acc = [x for x in history['eval_accuracy'] if str(x) != "nan"]
    loss = [x for x in history['eval_loss'] if str(x) != "nan"]
    f1 = [x for x in history['eval_f1'] if str(x) != "nan"]
    recall = [x for x in history['eval_recall'] if str(x) != "nan"]
    precision = [x for x in history['eval_precision'] if str(x) != "nan"]
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='acc', color="red")
    plt.plot(x, loss, 'b', label='loss', color="blue")
    plt.plot(x, f1, 'b', label='F1', color="green")
    plt.plot(x, recall, 'b', label='recall', color="yellow")
    plt.plot(x, precision, 'b', label='precision', color="black")
    plt.legend()
    plt.title("training history")
    plt.savefig(os.path.join(BASE_DIR, f"data/plots/training_history/{ts_now()}_{name}.png"))
    plt.close()

    print("Lowest Validation Loss: epoch {}".format(np.argmin(loss) + 1))
    print("Highest Validation Accuracy: epoch {}".format(np.argmax(acc) + 1))
    print("Highest Validation Precision: epoch {}".format(np.argmax(precision) + 1))
    print("Highest Validation Recall: epoch {}".format(np.argmax(recall) + 1))
    print("Highest Validation F1: epoch {}".format(np.argmax(f1) + 1))
