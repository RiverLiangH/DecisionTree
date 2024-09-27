import numpy as np


def accuracy_score(y_true, y_pred):
    """
    计算模型的准确率：正确预测的数量除以总数量。
    """
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred):
    """
    生成混淆矩阵，显示预测分类的正确与错误情况。
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
    Comupute Precision, Recall and F1-score
    """
    classes = np.unique(y_true)
    tp = np.zeros(len(classes), dtype=int)  # True Positive
    fp = np.zeros(len(classes), dtype=int)  # False Positive
    fn = np.zeros(len(classes), dtype=int)  # False Negative

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            tp[y_true[i]] += 1
        else:
            fp[y_pred[i]] += 1
            fn[y_true[i]] += 1

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = 2 * np.divide(precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float),
                       where=(precision + recall) != 0)

    return precision, recall, f1

