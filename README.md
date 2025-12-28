# Naive Bayes Spam Classifier 

This project implements a **Spam vs Ham SMS classification system** using multiple **Naive Bayes variants**. The goal is to demonstrate how different Naive Bayes models perform on text data and to compare their effectiveness using standard evaluation metrics.

---

## 📌 Project Overview

Spam detection is a classic text classification problem. In this project, we:
- Preprocess raw SMS text data
- Convert text into numerical features using vectorization techniques
- Train and evaluate multiple Naive Bayes classifiers
- Compare model performance using accuracy, precision, recall, F1-score, and confusion matrices

---

## 🧠 Models Implemented

- **Multinomial Naive Bayes**
- **Bernoulli Naive Bayes**
- **Gaussian Naive Bayes**

Each model is evaluated on the same dataset to analyze how assumptions of different Naive Bayes variants affect performance.

---

## 📂 Dataset

- **SMS Spam Collection Dataset**
- Source: UCI Machine Learning Repository
- Size: 5,574 SMS messages

**Features:**
- `message` – SMS text content  
- `label` – `spam` or `ham`

---

## ⚙️ Methodology

1. **Data Cleaning**
   - Removed irrelevant columns
   - Encoded labels (`spam = 1`, `ham = 0`)

2. **Text Vectorization**
   - TF-IDF Vectorizer
   - Count Vectorizer (for Bernoulli NB)

3. **Model Training**
   - 80/20 Train-Test split
   - Separate training for each Naive Bayes variant

4. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix Visualization

---

## 📊 Results Summary

| Model               | Accuracy |
|--------------------|----------|
| Multinomial NB     | ~95%     |
| Bernoulli NB       | ~94%     |
| Gaussian NB        | ~88%     |

> Multinomial Naive Bayes performed best, which aligns with expectations for text-based frequency data.

---

## 📈 Visualizations

- Confusion matrices for each classifier
- Classification reports showing precision, recall, and F1-score

---

## 🛠️ Technologies Used

- Python  
- pandas, NumPy  
- scikit-learn  
- matplotlib, seaborn  
- Jupyter Notebook  

---

## 🚀 How to Run the Project
