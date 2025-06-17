# Spam-Classifier-from-Scratch
This project is a spam email classifier developed as part of the Mathematical Methods for Artificial Intelligence course at the Vietnam National University Ho Chi Minh City, University of Science. The goal is to classify emails as spam or ham (non-spam) using a from-scratch implementation of the Multinomial Naive Bayes algorithm, helping users filter unwanted emails effectively.

# Project Overview
This project implements a complete pipeline for spam email classification, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and interactive features. The classifier is built using a Multinomial Naive Bayes algorithm with Laplace smoothing, trained on email data from CSV files (train.csv and val.csv). The pipeline handles malformed CSV files, cleans text data, and provides visualizations such as class distribution and word clouds.
The project was developed in Python using libraries like pandas, numpy, matplotlib, seaborn, and wordcloud, with the core Naive Bayes algorithm implemented from scratch.

# Usage
To run the spam classifier, execute the Jupyter Notebook Spam_Classifier.ipynb with the provided dataset files:
1. Open the notebook: jupyter notebook Spam_Classifier.ipynb
2. Ensure train.csv (training data) and val.csv (validation data) are in the same directory as the notebook.
3. Run all cells in the notebook to:
- Load and preprocess the data.
- Perform EDA (class distribution and word clouds).
- Train the Naive Bayes classifier.
- Evaluate performance on training and validation sets.
- Use interactive features to predict custom emails or evaluate new CSV files.

# Model Performance
The Naive Bayes classifier achieved the following results:
- Training Set (Seen Data):
+ Accuracy: 99.15%
+ Precision: 99.21%
+ Recall: 99.13%
+ F1-score: 99.17%
- Validation Set (Unseen Data):
+ Accuracy: 98.90%
+ Precision: 98.85%
+ Recall: 98.98%
+ F1-score: 98.91%

# Analysis:
- The model performs exceptionally well on both training and validation sets, with accuracy above 98.9%.
- The small performance drop (~0.25%) from training to validation indicates good generalization and minimal overfitting.
- High precision and recall suggest the model effectively identifies spam emails while minimizing false positives (ham emails misclassified as spam) and false negatives (spam emails misclassified as ham).
- The balanced F1-score confirms robust performance across both classes.

# Summary
This project successfully implements a spam email classifier using a custom Multinomial Naive Bayes algorithm. Key accomplishments include:
- Robust handling of malformed CSV files with a custom parser.
- Comprehensive data preprocessing pipeline to clean and tokenize email text.
- Insightful EDA with visualizations to understand data characteristics.
- High model performance (98.9% accuracy on unseen data) with minimal overfitting.
- Interactive features for real-world usability, such as custom email prediction and evaluation on new datasets.
The project demonstrates a solid understanding of text classification, probability-based machine learning, and data preprocessing techniques.

# Conclusion
The spam classifier is a reliable and efficient solution for distinguishing spam from ham emails. Its high accuracy, precision, recall, and F1-score make it suitable for basic email filtering applications. The from-scratch implementation of Naive Bayes provides valuable insights into the algorithm's mechanics, while the interactive features enhance practical usability.
However, the model's simplicity and reliance on word independence assumptions may limit its performance on more complex or modern spam emails (e.g., those with embedded images or sophisticated obfuscation). Future improvements could address these limitations to make the classifier more robust for real-world deployment.
