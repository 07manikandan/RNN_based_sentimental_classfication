Sentiment Analysis Using RNN with SpaCy Preprocessing
This project implements a sentiment analysis classifier using a Simple RNN model in TensorFlow/Keras. 
The text data is preprocessed using SpaCy's lemmatization and stopword removal. The model is trained to classify sentences based on their sentiment labels.

Features
Cleans and preprocesses text using SpaCy's English model (lemmatization, stopword removal, noise removal)

Uses one-hot encoding and sequence padding to prepare text for RNN input

Handles imbalanced classes with computed class weights

Trains a Simple RNN model with dropout regularization for sentiment classification

Supports interactive prediction for new text inputs with confidence scores

Evaluates model with confusion matrix and accuracy score

Requirements
Python 3.7+

pandas

numpy

scikit-learn

TensorFlow 2.x

SpaCy

SpaCy English model (en_core_web_sm)

Code Overview
Data loading & cleaning:
Drops missing values, resets indices.

Text preprocessing:
Removes URLs, mentions, numbers, special characters, stopwords, and applies lemmatization.

Text encoding:
One-hot encoding with vocabulary size of 5000 and sequence padding to length 40.

Label encoding:
Converts sentiment labels to numeric classes.

Model:
Embedding layer → SimpleRNN (128 units) with ReLU → Dropout → Dense layers → Softmax output.

Training:
Uses class weights for balancing and early stopping on validation loss.

Evaluation:
Prints confusion matrix and accuracy score.

Prediction:
Interactive function to predict sentiment class with confidence on new input sentences.



