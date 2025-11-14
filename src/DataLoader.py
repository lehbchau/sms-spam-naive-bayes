from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random

class DataLoader:
    @staticmethod
    def preprocess(text):
        """
        Converts text to lowercase, removes special characters and punctuation, tokenizes, applies stemming.
        """
        # Convert text to lowercase
        text = text.lower()
        # Remove special chars and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Tokenize words
        tokens = word_tokenize(text)
        # Remove stopwords and apply stemming
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return tokens

    @staticmethod
    def load_data(file_path):
        """
        Loads and preprocesses data from a text file, returning features and labels.
        """
        messages = []   # List of messages
        labels = []     # List of labels
        # Open file in read mode with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Remove leading/trailing whitespaces from line, split line into label & message
                label, message = line.strip().split(sep='\t', maxsplit=1)
                # Add preprocessed message to list of messages
                messages.append(DataLoader.preprocess(message))
                # Add binary label to list of labels
                labels.append(1 if label.lower() == 'spam' else 0) # 1 is spam, 0 is ham
        return messages, labels

    @staticmethod
    def split_data(messages, labels, train_ratio = 0.8):
        """
        Shuffles and splits data into training and testing sets (80% training - 20% testing).
        """
        # Combined list of messages and labels
        data = list(zip(messages, labels))
        # Randomize data
        random.shuffle(data)
        # Separate training and testing data
        split_idx = int(len(data) * train_ratio)
        train_messages, train_labels = zip(*data[:split_idx])
        test_messages, test_labels = zip(*data[split_idx:])
        return list(train_messages), list(train_labels), list(test_messages), list(test_labels)