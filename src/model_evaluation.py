import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Compute the accuracy
    """
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)

    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions: {total_predictions}")

    return correct_predictions / total_predictions

def confusion_matrix(y_true, y_pred):
    """
    Generate Confusion Matrix
    """
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for i in range(len(y_true)):
        true_idx = np.where(classes == y_true[i])[0][0]
        pred_idx = np.where(classes == y_pred[i])[0][0]
        matrix[true_idx, pred_idx] += 1

    return matrix


def precision_recall_f1(y_true, y_pred):
    """
    Compute Precision, Recall, and F1-score.
    """
    # Get the unique classes from both y_true and y_pred
    classes = np.unique(np.concatenate([y_true, y_pred]))

    # Initialize true positives, false positives, and false negatives
    tp = np.zeros(len(classes), dtype=int)  # True Positive
    fp = np.zeros(len(classes), dtype=int)  # False Positive
    fn = np.zeros(len(classes), dtype=int)  # False Negative

    # Create a mapping from class labels to indices
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Count tp, fp, fn
    for i in range(len(y_true)):
        true_idx = class_to_idx[y_true[i]]
        pred_idx = class_to_idx[y_pred[i]]

        if y_true[i] == y_pred[i]:
            tp[true_idx] += 1
        else:
            fp[pred_idx] += 1
            fn[true_idx] += 1

    # Compute precision, recall, and F1-score
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = 2 * np.divide(precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float),
                       where=(precision + recall) != 0)

    return precision, recall, f1

