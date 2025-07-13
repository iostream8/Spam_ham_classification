
# 📧 Spam vs. Ham 

An end-to-end machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using **Binary Logistic Classification**.

---

## 🔍 Project Overview

This project demonstrates the complete machine learning workflow:
- 📥 Load and inspect real-world SMS message dataset
- 🧹 Preprocess and clean text (lowercase, punctuation & URL removal)
- 🔢 Convert text to numerical format using **TF-IDF Vectorizer**
- 🤖 Train a **Binary Logistic Regression** model
- 📊 Evaluate performance with accuracy, confusion matrix, and sample predictions

---

## 🧠 Technologies Used

| Tool | Purpose |
|------|---------|
| **Python** | Programming language |
| **Pandas & NumPy** | Data analysis & manipulation |
| **Scikit-learn** | Feature extraction, model training & evaluation |

---

## 📁 Dataset

- **Source**: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Format**: CSV
- **Columns**:
  - `label`: `ham` or `spam`
  - `text`: The actual message content

---


## 🔠 Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical vectors:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)
```

---

## 🤖 Model Training

A **Binary Logistic regression** model is trained on the feature vectors:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, Y_train)

---


- **Accuracy**: ~96% (depending on test split)
- **Evaluation**:
  - Classification report
  - Prediction on new inputs

---

## 🧪 Sample Prediction

```python
def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vector = vectorizer.transform([msg_clean])
    pred = model.predict(msg_vector)
    return "Spam" if pred[0] == 1 else "Ham"

predict_message("Win a free iPhone now!")
# Output: Spam
```

---

## 📦 How to Run

1. Clone the repository:

```bash
git clone https://github.com/iostream8/spam-ham-classification.git
cd spam-ham-classification
```

2. Launch the notebook:

```bash
jupyter notebook spam_ham.ipynb
```

---

## 🌱 Future Enhancements

- Deploy as a web app using Flask or Streamlit
- Add lemmatization or emoji handling
- Try advanced models like SVM or LSTM

---

## 🙋‍♀️ Author

Made with ❤️ by [Priyanka Sharma](https://www.linkedin.com/in/priyanka-sharma-86b23024b/)  
📧 [priyankasharma945956@gmail.com](mailto:priyankasharma945956@gmail.com)

---
