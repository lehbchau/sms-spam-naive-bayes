# SMS Spam Classifier with Naive Bayes

This project implements a **Naive Bayes SMS spam classifier from
scratch** in Python using only standard libraries. It includes custom
text preprocessing, Laplace-smoothed probability modeling, and an
evaluation module for accuracy, precision, recall, F1 score, and
confusion matrix.

The model achieves **98.65 percent accuracy** and **95.36 percent F1
score** on the SMSSpamCollection dataset.

## Features

-   Custom implementation of **Naive Bayes** without sklearn\
-   Complete text preprocessing pipeline
    -   lowercasing\
    -   regex cleaning\
    -   tokenization\
    -   stemming\
    -   stopword handling\
-   Laplace smoothing for probability estimation\
-   Log-probability computations for numerical stability\
-   Evaluation metrics (accuracy, precision, recall, F1, confusion
    matrix)

## Installation

1.  Create a virtual environment:

``` bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
```

2.  Install required dependencies:

``` bash
pip install nltk
```

3.  Download NLTK resources:

``` python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## How to Run

From inside the **src** directory:

``` bash
python Main.py
```

This will load data, preprocess text, train the classifier, evaluate
performance, and write results to `results/results.log`.

## Model Performance

-   **Training accuracy:** 99.35 percent\
-   **Training F1 score:** 97.49 percent\
-   **Testing accuracy:** 98.65 percent\
-   **Testing F1 score:** 95.36 percent\
-   **Confusion matrix (test):**
    -   True Positive: 154\
    -   True Negative: 946\
    -   False Positive: 5\
    -   False Negative: 10

## Dataset

Uses the SMSSpamCollection dataset (UCI), containing 5,574 labeled SMS
messages.

## License

This project is licensed under the **MIT License**.\
You are free to use, modify, and distribute this software with
attribution.
