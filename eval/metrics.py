# Metrics module

from sklearn.metrics import balanced_accuracy_score

def compute_bal_acc(true_labels, predicted_labels):
    return balanced_accuracy_score(true_labels, predicted_labels)
