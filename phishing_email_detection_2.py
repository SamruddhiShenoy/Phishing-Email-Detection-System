"""
Phishing Email Detection System
--------------------------------
This script implements a simple phishing email detection system using:
- Machine Learning (Naive Bayes classifier) from scikit-learn
- Email text preprocessing with NLTK
- Tkinter UI for user input and interaction
- Logging of phishing detection results

Usage:
- Run this script to open the GUI.
- Use the "Train Model" button once to train on sample data.
- Enter or paste an email text and click "Detect" to classify it.
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import pandas as pd
import nltk
import string
import re
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Ensure nltk data downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Setup logging
logging.basicConfig(filename='phishing_detection.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize global variables for model and vectorizer
model = None
vectorizer = None
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Sample dataset for demonstration
# Usually, you'd load a large dataset from a csv file
# Here we create a small sample inline for illustration
sample_data = {
    'text': [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now.",
        "Important update: Your account has been suspended. Verify your information immediately.",
        "Hello friend, let's catch up over lunch tomorrow.",
        "Meeting tomorrow at 10 AM, please confirm your attendance.",
        "Your PayPal account has been limited, please log in to fix the issue.",
        "Project deadline moved to next Friday. Prepare your reports.",
        "Dear user, your mailbox is almost full. Please upgrade your storage.",
        "Cheap meds available without prescription! Buy now.",
        "Don't miss out on this limited time offer to save money.",
        "The report has been attached as requested. Let me know your feedback."
    ],
    'label': [
        'phishing',
        'phishing',
        'legitimate',
        'legitimate',
        'phishing',
        'legitimate',
        'phishing',
        'phishing',
        'phishing',
        'legitimate'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(sample_data)

def preprocess_text(text):
    """
    Preprocess email text for ML model:
    - Lowercasing
    - Remove URLs, emails, numbers, punctuation
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    # Lowercase
    text = text.lower()
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    # Remove digits and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Rejoin tokens
    return " ".join(tokens)

def train_model():
    global model, vectorizer
    # Preprocess dataset
    df['processed_text'] = df['text'].apply(preprocess_text)

    X = df['processed_text']
    y = df['label']

    # Vectorize text
    vectorizer = CountVectorizer()
    X_vectors = vectorizer.fit_transform(X)

    # Train-test split (for demonstration only)
    X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate accuracy on test
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.2f}")
    messagebox.showinfo("Model Training", f"Model trained successfully with accuracy: {accuracy:.2f}")

def detect_phishing(email_text):
    if model is None or vectorizer is None:
        messagebox.showwarning("Model Not Trained", "Please train the model first by clicking 'Train Model'.")
        return

    processed = preprocess_text(email_text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    proba = max(model.predict_proba(vector)[0])

    # Log prediction if phishing
    if prediction == 'phishing':
        logging.info(f"Phishing email detected: {email_text[:100]}... Probability: {proba:.2f}")

    return prediction, proba

def on_detect_click():
    email_text = email_input.get("1.0", tk.END).strip()
    if not email_text:
        messagebox.showwarning("Input Error", "Please enter or paste an email text to detect.")
        return
    prediction, proba = detect_phishing(email_text)
    if prediction == 'phishing':
        result_label.config(text=f"Result: PHISHING (Confidence: {proba:.2f})", fg='red')
    else:
        result_label.config(text=f"Result: Legitimate (Confidence: {proba:.2f})", fg='green')

def on_train_click():
    train_model()

# GUI Setup
root = tk.Tk()
root.title("Phishing Email Detection System")
root.geometry('700x600')

title = tk.Label(root, text="Phishing Email Detection System", font=("Arial", 20, "bold"))
title.pack(pady=10)

instruction = tk.Label(root, text="Enter or paste the email content below and click 'Detect' to classify the email.", font=("Arial", 12))
instruction.pack(pady=5)

email_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
email_input.pack(padx=10, pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

train_btn = tk.Button(btn_frame, text="Train Model", command=on_train_click, width=15, bg='blue', fg='white')
train_btn.grid(row=0, column=0, padx=10)

detect_btn = tk.Button(btn_frame, text="Detect", command=on_detect_click, width=15, bg='green', fg='white')
detect_btn.grid(row=0, column=1, padx=10)

result_label = tk.Label(root, text="Result: Not detected yet", font=("Arial", 14))
result_label.pack(pady=15)

footer = tk.Label(root, text="Powered by Naive Bayes Classifier and NLTK Text Processing", font=("Arial", 10, "italic"))
footer.pack(side=tk.BOTTOM, pady=10)

root.mainloop()

