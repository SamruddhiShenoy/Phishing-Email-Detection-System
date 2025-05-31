# Phishing-Email-Detection-System
(# Phishing Email Detection System)

## Overview
The Phishing Email Detection System (PEDS) is a Python-based application designed to identify and classify emails as phishing or legitimate. It utilizes machine learning techniques, specifically a Naive Bayes classifier, along with natural language processing (NLP) for text analysis. The application features a user-friendly graphical interface built with Tkinter, allowing users to input email content and receive real-time detection results.

## Features
- Real-time phishing email detection
- User-friendly interface for email input
- Logging of detected phishing attempts
- Model training using a sample dataset
- Text preprocessing using NLTK for improved accuracy

## Requirements
- Python 3.x
- Libraries:
  - Tkinter (included with Python)
  - pandas
  - scikit-learn
  - nltk
  - logging

## Installation
1. Clone the repository or download the project files.
2. Install the required libraries using pip:
   ```bash
   pip install pandas scikit-learn nltk
   ```
3. Ensure that the NLTK data is downloaded. You can run the following commands in a Python shell:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage
1. Run the application:
   ```bash
   python phishing_email_detection.py
   ```
2. Click the "Train Model" button to train the model using the sample dataset.
3. Enter or paste the email content into the text area.
4. Click the "Detect" button to classify the email as phishing or legitimate.
5. The result will be displayed below the input area, along with the confidence level.

## Logging
All detected phishing attempts are logged in a file named `phishing_detection.log`. This log file contains timestamps and details of the detected phishing emails.

## Sample Dataset
The application includes a small sample dataset for demonstration purposes. You can expand the dataset by adding more labeled email examples to improve the model's accuracy.

## Contributing
Contributions are welcome! If you have suggestions for improvements or additional features, feel free to create a pull request or open an issue.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- [scikit-learn](https://scikit-learn.org/stable/) for machine learning algorithms
- [NLTK](https://www.nltk.org/) for natural language processing
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for the graphical user interface
```

Feel free to modify any sections to better fit your project or add any additional information you think is necessary!
