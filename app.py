from flask import Flask, render_template, request
import pandas as pd
import re
import string
import logging
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Paths for saved models and vectorizer
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
LR_MODEL_PATH = 'logistic_regression_model.pkl'
DT_MODEL_PATH = 'decision_tree_model.pkl'
GB_MODEL_PATH = 'gradient_boosting_model.pkl'
RF_MODEL_PATH = 'random_forest_model.pkl'

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Check if models and vectorizer exist
def models_exist():
    return all(os.path.exists(path) for path in [
        VECTORIZER_PATH, LR_MODEL_PATH, DT_MODEL_PATH, GB_MODEL_PATH, RF_MODEL_PATH
    ])

# Load or train models
if models_exist():
    logger.info("Loading saved vectorizer and models")
    vectorization = joblib.load(VECTORIZER_PATH)
    LR = joblib.load(LR_MODEL_PATH)
    DT = joblib.load(DT_MODEL_PATH)
    GB = joblib.load(GB_MODEL_PATH)
    RF = joblib.load(RF_MODEL_PATH)
    logger.info("Saved vectorizer and models loaded successfully")
else:
    logger.info("No saved models found, starting training process")
    
    # Load and preprocess data
    logger.info("Loading datasets: Fake.csv and True.csv")
    data_fake = pd.read_csv('Fake.csv')
    data_true = pd.read_csv('True.csv')
    logger.info("Datasets loaded successfully")

    # Assign class labels
    logger.info("Assigning class labels")
    data_fake["class"] = 0
    data_true["class"] = 1

    # Remove last 10 rows for manual testing
    logger.info("Removing last 10 rows for manual testing")
    data_fake_manual_testing = data_fake.tail(10)
    for i in range(23480, 23470, -1):
        data_fake.drop([i], axis=0, inplace=True)

    data_true_manual_testing = data_true.tail(10)
    for i in range(21416, 21406, -1):
        data_true.drop([i], axis=0, inplace=True)

    # Merge datasets
    logger.info("Merging fake and true datasets")
    data_merge = pd.concat([data_fake, data_true], axis=0)
    data = data_merge.drop(['title', 'subject', 'date'], axis=1)

    # Shuffle data
    logger.info("Shuffling dataset")
    data = data.sample(frac=1).reset_index(drop=True)

    # Apply preprocessing
    logger.info("Applying text preprocessing to dataset")
    data['text'] = data['text'].apply(wordopt)
    logger.info("Text preprocessing completed")

    # Define features and labels
    logger.info("Defining features and labels")
    x = data['text']
    y = data['class']

    # Split data
    logger.info("Splitting data into training and testing sets")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    logger.info("Data split completed")

    # Vectorize text
    logger.info("Starting text vectorization with TfidfVectorizer")
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)
    logger.info("Text vectorization completed")

    # Train models
    logger.info("Starting model training")

    # Logistic Regression
    logger.info("Training Logistic Regression model")
    LR = LogisticRegression()
    LR.fit(xv_train, y_train)
    logger.info("Logistic Regression training completed")

    # Decision Tree
    logger.info("Training Decision Tree model")
    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)
    logger.info("Decision Tree training completed")

    # Gradient Boosting
    logger.info("Training Gradient Boosting model")
    GB = GradientBoostingClassifier(random_state=0)
    GB.fit(xv_train, y_train)
    logger.info("Gradient Boosting training completed")

    # Random Forest
    logger.info("Training Random Forest model")
    RF = RandomForestClassifier(random_state=0)
    RF.fit(xv_train, y_train)
    logger.info("Random Forest training completed")

    logger.info("All model training completed")

    # Save vectorizer and models
    logger.info("Saving vectorizer and models to disk")
    joblib.dump(vectorization, VECTORIZER_PATH)
    joblib.dump(LR, LR_MODEL_PATH)
    joblib.dump(DT, DT_MODEL_PATH)
    joblib.dump(GB, GB_MODEL_PATH)
    joblib.dump(RF, RF_MODEL_PATH)
    logger.info("Vectorizer and models saved successfully")

# Function to convert prediction to label
def output_label(n):
    return "Fake News" if n == 0 else "Not Fake News"

# Function to predict news authenticity
def manual_testing(news):
    logger.info("Processing input news text for prediction")
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    logger.info("Prediction completed")
    return {
        "lr": output_label(pred_LR[0]),
        "dt": output_label(pred_DT[0]),
        "gb": output_label(pred_GB[0]),
        "rf": output_label(pred_RF[0])
    }

# Flask routes
@app.route('/')
def home():
    logger.info("Rendering home page")
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    logger.info("Received POST request for prediction")
    news_text = request.form['news_text']
    result = manual_testing(news_text)
    logger.info("Rendering prediction results")
    return render_template('index.html', result=result)

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(debug=True)
    logger.info("Flask server started on http://127.0.0.1:5000")