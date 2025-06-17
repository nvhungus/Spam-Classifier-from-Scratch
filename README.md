Spam-Classifier-from-Scratch

This project is a spam email classifier developed as part of the Mathematical Methods for Artificial Intelligence course at the University of Science, Vietnam National University, Ho Chi Minh City. The goal is to classify emails as spam or ham (non-spam) using a from-scratch implementation of the Multinomial Naive Bayes algorithm, helping users filter unwanted emails effectively.

Table of Contents





Project Overview



Features



Requirements



Installation



Usage



Model Performance



Summary



Conclusion



Future Improvements



Contributing



License

Project Overview

This project implements a complete pipeline for spam email classification, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and interactive features. The classifier is built using a Multinomial Naive Bayes algorithm with Laplace smoothing, trained on email data from CSV files (train.csv and val.csv). The pipeline handles malformed CSV files, cleans text data, and provides visualizations such as class distribution and word clouds.

The project was developed in Python using libraries like pandas, numpy, matplotlib, seaborn, and wordcloud, with the core Naive Bayes algorithm implemented from scratch.

Features





Advanced CSV Parsing: Handles malformed CSV files where email content may span multiple lines.



Data Preprocessing: Cleans text by removing URLs, HTML tags, and non-alphabetic characters; combines subject and message fields; and tokenizes text.



Exploratory Data Analysis (EDA):





Visualizes class distribution (spam vs. ham) using a bar plot.



Generates word clouds to highlight frequent words in spam and ham emails.



Naive Bayes Classifier: Custom implementation of Multinomial Naive Bayes with Laplace smoothing for robust text classification.



Evaluation: Reports accuracy, precision, recall, and F1-score; visualizes performance with a confusion matrix.



Interactive Features:





Predicts spam/ham for user-input emails.



Evaluates the model on any provided CSV file.

Requirements





Python 3.8+



Libraries:

pandas
numpy
matplotlib
seaborn
wordcloud

Installation





Clone the repository:

git clone https://github.com/your-username/Spam-Classifier-from-Scratch.git
cd Spam-Classifier-from-Scratch



Install the required libraries:

pip install -r requirements.txt



Ensure the dataset files (train.csv and val.csv) are in the project directory.

Usage

To run the spam classifier, execute the Jupyter Notebook Spam_Classifier.ipynb with the provided dataset files:





Open the notebook:

jupyter notebook Spam_Classifier.ipynb



Ensure train.csv (training data) and val.csv (validation data) are in the same directory as the notebook.



Run all cells in the notebook to:





Load and preprocess the data.



Perform EDA (class distribution and word clouds).



Train the Naive Bayes classifier.



Evaluate performance on training and validation sets.



Use interactive features to predict custom emails or evaluate new CSV files.

Example Output:

--- [STEP 1] LOADING AND RECONSTRUCTING DATA ---
Parsing and reconstructing file: 'train.csv'...
Reconstruction complete. Found X records.
Parsing and reconstructing file: 'val.csv'...
Reconstruction complete. Found Y records.

--- [STEP 3] MODEL TRAINING & EVALUATION ---
|   MODEL PERFORMANCE ON: TRAIN SET (SEEN DATA)   |
- Accuracy:  0.9915
- Precision: 0.9921
- Recall:    0.9913
- F1-score:  0.9917

|   MODEL PERFORMANCE ON: VALIDATION SET (UNSEEN DATA)   |
- Accuracy:  0.9890
- Precision: 0.9885
- Recall:    0.9898
- F1-score:  0.9891

Model Performance

The Naive Bayes classifier achieved the following results:





Training Set (Seen Data):





Accuracy: 99.15%



Precision: 99.21%



Recall: 99.13%



F1-score: 99.17%



Validation Set (Unseen Data):





Accuracy: 98.90%



Precision: 98.85%



Recall: 98.98%



F1-score: 98.91%

Analysis:





The model performs exceptionally well on both training and validation sets, with accuracy above 98.9%.



The small performance drop (~0.25%) from training to validation indicates good generalization and minimal overfitting.



High precision and recall suggest the model effectively identifies spam emails while minimizing false positives (ham emails misclassified as spam) and false negatives (spam emails misclassified as ham).



The balanced F1-score confirms robust performance across both classes.

Summary

This project successfully implements a spam email classifier using a custom Multinomial Naive Bayes algorithm. Key accomplishments include:





Robust handling of malformed CSV files with a custom parser.



Comprehensive data preprocessing pipeline to clean and tokenize email text.



Insightful EDA with visualizations to understand data characteristics.



High model performance (98.9% accuracy on unseen data) with minimal overfitting.



Interactive features for real-world usability, such as custom email prediction and evaluation on new datasets.

The project demonstrates a solid understanding of text classification, probability-based machine learning, and data preprocessing techniques.

Conclusion

The spam classifier is a reliable and efficient solution for distinguishing spam from ham emails. Its high accuracy, precision, recall, and F1-score make it suitable for basic email filtering applications. The from-scratch implementation of Naive Bayes provides valuable insights into the algorithm's mechanics, while the interactive features enhance practical usability.

However, the model's simplicity and reliance on word independence assumptions may limit its performance on more complex or modern spam emails (e.g., those with embedded images or sophisticated obfuscation). Future improvements could address these limitations to make the classifier more robust for real-world deployment.

Future Improvements





Preprocessing:





Apply lemmatization or stemming to normalize words.



Remove stop words to reduce noise.



Incorporate additional features (e.g., email length, number of links, special characters).



Model:





Experiment with advanced algorithms (e.g., SVM, Random Forest, or BERT-based models).



Use TF-IDF or n-grams to capture contextual information.



Implement Complement Naive Bayes for imbalanced datasets.



Evaluation:





Add ROC-AUC or Precision-Recall curves for deeper performance analysis.



Test on diverse, real-world datasets to assess generalization.



Usability:





Develop a graphical user interface (GUI) for easier interaction.



Deploy the model as a web application using Flask or FastAPI.

Contributing

Contributions are welcome! Please follow these steps:





Fork the repository.



Create a new branch (git checkout -b feature-branch).



Commit your changes (git commit -m 'Add new feature').



Push to the branch (git push origin feature-branch).



Open a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.
