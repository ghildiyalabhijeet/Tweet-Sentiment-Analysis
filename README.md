# **Twitter Sentiment Analysis Using ALBERT**

## **Table of Contents**
- [Project Overview](#project-overview)
- [Installation & Dependencies](#installation--dependencies)
- [Dataset](#dataset)
- [Pipeline Workflow](#pipeline-workflow)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Tokenization (ALBERT)](#2-tokenization-albert)
  - [3. Model Architecture](#3-model-architecture)
  - [4. Model Training](#4-model-training)
  - [5. Model Evaluation](#5-model-evaluation)
  - [6. Predictions](#6-predictions)
- [Visualization](#visualization)
- [Results & Performance](#results--performance)
- [Future Improvements](#future-improvements)
- [How to Run the Project](#how-to-run-the-project)
- [Conclusion](#conclusion)

## **Project Overview**
This project implements **sentiment analysis** on Twitter data using **ALBERT (A Lite BERT)** in TensorFlow/Keras. The goal is to classify tweets into three sentiment categories:
- **Negative**
- **Neutral**
- **Positive**

This project covers **data preprocessing, tokenization, model training, evaluation, and visualization** to build an optimized sentiment classification model.

---

## **Installation & Dependencies**
Ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

If you don't have `requirements.txt`, manually install:
```bash
pip install matplotlib seaborn emoji nltk wordcloud scikit-learn nlpaug tensorflow tf-keras transformers pandas
```

Additionally, download **NLTK resources**:
```python
import nltk
nltk.download('wordnet')
```

---

## **Dataset**
The project uses:
- **`train.csv`** - Training dataset with labeled tweets.
- **`test.csv`** - Test dataset for model evaluation.
- **Link** - https://www.kaggle.com/competitions/tweet-sentiment-extraction/data

The dataset contains:
- `text` - Raw tweet text
- `sentiment` - Sentiment label (`Negative`, `Neutral`, `Positive`)

---

## **Pipeline Workflow**
### **1. Data Preprocessing**
The text undergoes:
- **Removing URLs, emails, hashtags, mentions**
- **Emoji conversion to text**
- **Removing punctuation and numbers**
- **Applying lemmatization**

### **2. Tokenization (ALBERT)**
Tokenization is done using **ALBERT tokenizer (`albert-base-v2`)**:
```python
from transformers import AlbertTokenizer

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
```
Tokenized sequences are padded and truncated to a **maximum length of 128**.

### **3. Model Architecture**
A **custom ALBERT-based model** is built for sentiment classification:
- **Pre-trained ALBERT embeddings**
- **GlobalMaxPooling1D layer** for feature extraction
- **Dropout layer (0.2) for regularization**
- **Dense layer (Softmax) for multi-class classification**

The model is compiled with:
```python
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
metrics=['accuracy']
```

### **4. Model Training**
Training is done with **early stopping** to prevent overfitting:
```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True
)
```
The model is trained for **10 epochs** with a **batch size of 32**.

### **5. Model Evaluation**
After training, the model is evaluated using:
- **Classification report** (Precision, Recall, F1-score)
- **Confusion matrix**
- **Training/Validation accuracy and loss plots**

### **6. Predictions**
The model predicts sentiment probabilities, and the highest probability class is assigned:
```python
y_pred_probs = model.predict([X_test_ids, X_test_attention])
y_pred = np.argmax(y_pred_probs, axis=1)
```

---

## **Visualization**
- **Sentiment distribution** (Bar chart)
- **Confusion Matrix** (Heatmap using Seaborn)
- **Training Accuracy & Loss Over Epochs** (Line plots)

---

## **Results & Performance**
- The model achieves high **accuracy and F1-score** on test data.
- **Misclassifications** are analyzed using a confusion matrix.
- The training history is visualized to **check for overfitting.**

---

## **Future Improvements**
- **Data Augmentation** using `nlpaug`
- **Hyperparameter tuning** (batch size, dropout rate, learning rate)
- **Deployment** using FastAPI/Flask for real-time predictions

---

## **How to Run the Project**
1. Clone the repository:
```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Load `train.csv` and `test.csv`
4. Run the Jupyter Notebook:
```bash
jupyter notebook twitter-sentiment-analysis.ipynb
```
5. Train the ALBERT model and evaluate performance
6. Analyze predictions and visualize results

---

## **Conclusion**
This project successfully fine-tunes **ALBERT for sentiment analysis on Twitter data**, demonstrating efficient text preprocessing, deep learning-based classification, and model evaluation.

🚀 **Future work includes expanding the dataset, optimizing hyperparameters, and deploying the model as an API.**
