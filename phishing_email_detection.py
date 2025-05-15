"""
Phishing Email Detection - MCA Cyber Security Project

This script demonstrates a simple machine learning pipeline to detect phishing emails using Python.
It includes:
- A sample dataset with labeled emails (phishing vs. not phishing)
- Text preprocessing using TF-IDF vectorization
- Training a Logistic Regression classifier
- Predicting new emails as phishing or legitimate

Requirements:
- Python 3.x
- scikit-learn
- pandas
- numpy

Run this script: python phishing_email_detection.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class PhishingEmailDetector:

    def __init__(self):
        # Initialize TF-IDF Vectorizer and Logistic Regression model
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        self.model = LogisticRegression(solver='liblinear', random_state=42)

    def preprocess(self, emails):
        """
        Convert raw emails to TF-IDF features.
        :param emails: list of email texts
        :return: TF-IDF feature matrix
        """
        return self.vectorizer.fit_transform(emails)

    def preprocess_transform(self, emails):
        """
        Transform emails to existing TF-IDF vectorizer.
        Use after fitting vectorizer on training data.
        """
        return self.vectorizer.transform(emails)

    def train(self, emails, labels):
        """
        Train the model on emails and labels.
        :param emails: list of email texts
        :param labels: corresponding list of labels (1=phishing, 0=not phishing)
        """
        print("[INFO] Preprocessing training data...")
        X = self.preprocess(emails)
        y = np.array(labels)
        print("[INFO] Training model...")
        self.model.fit(X, y)
        print("[INFO] Model training completed.")

    def evaluate(self, emails, labels):
        """
        Evaluate the model on test data.
        """
        X_test = self.preprocess_transform(emails)
        y_test = np.array(labels)
        y_pred = self.model.predict(X_test)
        print("[INFO] Model Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    def predict(self, email_text):
        """
        Predict whether an email is phishing or not.
        :param email_text: string, raw email content
        :return: string prediction result
        """
        X = self.preprocess_transform([email_text])
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0][prediction]
        label = "Phishing" if prediction == 1 else "Not Phishing"
        return f"Prediction: {label} (Confidence: {proba*100:.2f}%)"

def main():
    # Sample dataset - replace with real dataset as needed
    data = {
        'email': [
            "Congrats! You won a $1000 Walmart gift card. Click here to claim now.",
            "Dear user, your account has been suspended. Please login to verify.",
            "Meeting schedule for next week attached in this email.",
            "Your invoice for last month attached. Please review and pay.",
            "Update your billing information to avoid cancelation of service.",
            "Important security alert - password reset required immediately.",
            "Can we reschedule our appointment for next Tuesday?",
            "Congratulations, you have been selected for a free cruise trip!",
            "Don't forget our team outing this Friday. RSVP please.",
            "Your parcel delivery failed. Click the link to reschedule delivery."
        ],
        'label': [1, 1, 0, 0, 1, 1, 0, 1, 0, 1]  # 1 = phishing, 0 = not phishing
    }

    df = pd.DataFrame(data)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['email'], df['label'], test_size=0.3, random_state=42
    )

    detector = PhishingEmailDetector()
    detector.train(X_train.tolist(), y_train.tolist())
    detector.evaluate(X_test.tolist(), y_test.tolist())

    # Demo predictions
    test_emails = [
        "Your Amazon account was compromised. Reset your password immediately!",
        "Let's meet for lunch tomorrow at 1 PM.",
        "You won a lottery! Send your bank details to claim the reward."
    ]

    print("\n[DEMO Predictions]")
    for email in test_emails:
        print(f"\nEmail: {email}")
        print(detector.predict(email))

if __name__ == "__main__":
    main()

