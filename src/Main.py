from DataLoader import DataLoader
from NaiveBayes import NaiveBayes
from EvaluationMetrics import EvaluationMetrics
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

def main():
    data_path = 'SMSSpamCollection.txt'
    log_path = 'results.log'

    # Load data
    messages, labels = DataLoader.load_data(data_path)
    train_messages, train_labels, test_messages, test_labels = DataLoader.split_data(messages, labels)

    # Train the Naive Bayes model
    nb = NaiveBayes()
    nb.train(train_messages, train_labels)

    # Make predictions on training data
    train_pred = nb.predict(train_messages)

    # Make predictions on testing data
    test_pred = nb.predict(test_messages)

    # Compute evaluation metrics
    train_metrics = EvaluationMetrics.compute_metrics(train_labels, train_pred)
    test_metrics = EvaluationMetrics.compute_metrics(test_labels, test_pred)

    # Log results to results.log file
    with open(log_path, 'w') as log_file:
        log_file.write("Model Performance:\n")
        log_file.write(f"Number of training samples: {len(train_labels)}\n")
        log_file.write(f"Number of testing samples: {len(test_labels)}\n")
        log_file.write(f"Training Accuracy: {train_metrics['Accuracy']:.4f}\n")
        log_file.write(f"Testing Accuracy: {test_metrics['Accuracy']:.4f}\n")
        log_file.write(f"Training F1 Score: {train_metrics['F1 Score']:.4f}\n")
        log_file.write(f"Testing F1 Score: {test_metrics['F1 Score']:.4f}\n")
        log_file.write(f"TP: {test_metrics['TP']}, TN: {test_metrics['TN']}, "
                       f"FP: {test_metrics['FP']}, FN: {test_metrics['FN']}\n")

    print("Training and evaluation complete. Results logged to results.log.")

if __name__ == "__main__":
    main()
