# 📰 Fake News Detection System

Fake news has become one of the most significant challenges in the age of digital information. This project presents a machine learning-based Fake News Detection System that can accurately classify news articles as **genuine** or **fake**, helping users and organizations safeguard against misinformation.

---

## Overview

The system leverages **machine learning algorithms** to analyze textual data from news articles. By identifying patterns in writing styles, vocabulary, and context, the model learns to distinguish between factual and misleading content.

This project involves:
- Data preprocessing and cleaning of news datasets  
- Feature extraction using TF-IDF vectorization  
- Training ML models (Logistic Regression, Random Forest Classifier, etc.)  
- Evaluating performance using precision, recall, F1-score, and accuracy  
- Building a basic web interface for testing input articles

---

## Features

- Binary classification: Fake vs Real  
- Multiple ML models compared for performance  
- Interactive and easy-to-use architecture  
- Scalable for larger datasets and real-world applications

---

## Machine Learning Models Used

- **Logistic Regression**
- **Decision Trees Classifier**
- **Gradient Boosting Classifier**
- **Random Forest Classifier**

---

## Dataset

The model is trained on a dataset which includes thousands of news articles from reliable and fake news sources.

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix

These metrics provide a well-rounded view of model performance and generalizability.

---

## Performance Metrics

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 97.85%   | 97.92%    | 97.78% | 97.85%   |
| Decision Tree      | 98.12%   | 98.30%    | 97.95% | 98.12%   |
| Gradient Boosting  | 98.76%   | 98.88%    | 98.63% | 98.75%   |
| Random Forest      | 99.03%   | 99.10%    | 98.95% | 99.02%   |

---

## Tools and Technologies

- Python  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib
- Jupyter Notebook  
- Flask for interface

---

## How to Run

  1. Clone this repo:
```bash
git clone https://github.com/IshanPaharia/Fake-New-Detection-System.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Webapp
```bash
python app.py
```
Then, open your browser and go to: http://127.0.0.1:5000

---

## Meet the Crew

| Name              |  Roll No. |
|-----------------------|------------------|
| **Ishan Paharia**       | 23ucs597         |
| **Gaurvi Singhal** | 23ucs580         |
| **Aishwarya Wadhwani**    | 23ucs521         |

---

## Contributions

Contributions are welcome! Please open issues or submit a pull request if you have suggestions or improvements.
