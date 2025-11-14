class EvaluationMetrics:
    @staticmethod
    def compute_metrics(true_labels, pred_labels):
        """
        Compute evaluation metrics from true and predicted labels.
        """
        # Initialize variables
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for true, pred in zip(true_labels, pred_labels):
            if true == 1 and pred == 1:     # True positive: spam correctly detected
                tp += 1
            elif true == 0 and pred == 1:   # False positive: ham mistaken as spam
                fp += 1
            elif true == 0 and pred == 0:   # True negative: ham correctly detected
                tn += 1
            elif true == 1 and pred == 0:   # False negative: spam mistaken as ham
                fn += 1

        # Compute metrics while handling division by 0 error
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1_score}