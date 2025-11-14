import math
from collections import Counter

class NaiveBayes:
    """
    Implements a Naive Bayes classifier for text classification.
    """
    def __init__(self, smoothing=1):
        self.smoothing = smoothing
        self.class_priors = {}      # Dictionary storing P(class) for each class
        self.word_probs = {}        # Nested dictionary that serves as P(word|class) table for each word and class
        self.vocab = set()          # Set of unique words across all messages

    def train(self, train_messages, train_labels):
        """
        Trains the Naive Bayes model by computing class priors and word probabilities.
        """
        # Compute total number of train messages
        total_messages = len(train_labels)
        # Compute number of messages in each class
        class_counts = Counter(train_labels)
        # Compute prior probability for each class
        self.class_priors = {cls: class_counts[cls] / total_messages for cls in class_counts}

        word_counts = {cls: Counter() for cls in class_counts}      # Count of each word per class
        total_words = {cls: 0 for cls in class_counts}              # Total word count per class

        # Compute word count per class and build vocabulary set
        for message, label in zip(train_messages, train_labels):
            word_counts[label].update(message)
            self.vocab.update(message)
            total_words[label] += len(message)

        vocab_size = len(self.vocab)

        # Compute P(word|class) with Laplace smoothing
        self.word_probs = {cls: {} for cls in class_counts}
        for cls in class_counts:
            for word in self.vocab:
                self.word_probs[cls][word] = (word_counts[cls][word] + self.smoothing) / (total_words[cls] + vocab_size * self.smoothing)

    def predict(self, messages):
        """
        Predicts class labels (spam vs. ham) for messages.
        """
        predictions = []
        for message in messages:
            # Compute log class priors
            log_probs = {cls: math.log(self.class_priors[cls]) for cls in self.class_priors}

            # Compute log probabilities for words in message
            for word in message:
                if word in self.vocab:  # Only consider words that were seen
                    for cls in self.class_priors:
                        log_probs[cls] += math.log(self.word_probs[cls][word])  # Add log probability of the word

            # Choose class label with the higher probability
            predictions.append(max(log_probs, key=log_probs.get))

        return predictions