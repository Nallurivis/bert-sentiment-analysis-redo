# Fine-Tuning BERT for Multi-Class Sentiment Analysis

## Overview
This project demonstrates the fine-tuning of a BERT model for sentiment analysis on the Yelp Reviews dataset. The model predicts one of five sentiment categories: Very Negative, Negative, Neutral, Positive, or Very Positive.

## Features
- Preprocesses Yelp reviews using Hugging Face's tokenizer.
- Fine-tunes BERT using Hugging Face's Trainer API.
- Evaluates model performance with metrics like accuracy and loss.
- Provides example predictions and confusion matrix visualizations.

## Installation and Usage
### Installation
To run this project, install the necessary dependencies by adding them to a `requirements.txt` file and executing:
```bash
pip install -r requirements.txt
```
### Model training:
```bash
python train_model.py
```
### Testing
```bash
python test_model.py

```
## Results
Epoch	Training Loss	Validation Loss	Accuracy
1	0.8249	0.6965	69.66%
2	0.5089	0.6950	70.12%
3	0.5252	0.7333	69.92%

## Example Predictions
Input: "The food was amazing and the ambiance delightful!"
Prediction: Very Positive
Input: "The service was terrible and overpriced."
Prediction: Very Negative
## Project Structure
train_model.py: Script for training and evaluation.
utils.py: Helper functions for preprocessing or additional tasks.
README.md: Project documentation.
requirements.txt: List of dependencies.
## Future Work
Hyperparameter tuning with Optuna or similar frameworks.
Deployment as an API using Flask or FastAPI.
Exploring alternative transformer architectures like RoBERTa.
License
## This project is licensed under the MIT License. See LICENSE for details.
