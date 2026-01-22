# 📩 SMS Spam Detection Project

A complete **end-to-end Machine Learning project** for detecting spam SMS messages, starting from **data preprocessing & NLP**, through **model training**, and ending with **MLOps using MLflow**, **API deployment (FastAPI)**, and **UI interface (Streamlit)**.

This project is designed with a **Data Scientist / ML Engineer mindset**, focusing on reproducibility, scalability, and production readiness.

---

## 🚀 Project Overview

The goal of this project is to classify SMS messages into:

* **Ham (0)** → Normal message
* **Spam (1)** → Unwanted or malicious message

The solution uses:

* **TF-IDF** for text vectorization
* **Multinomial Naive Bayes** for classification
* **MLflow** for experiment tracking, model packaging, registry, and production gating

---

## 🧠 Project Architecture

```
SMS_Spam_Classifier/
│
├── data_pipeline.py          # Data loading & cleaning
├── eda_text_prepro.py        # EDA + NLP preprocessing
├── visualization.py          # Data visualization
├── model.py                  # Model training & evaluation
├── mlflow.py                 # MLflow lifecycle & Model Registry
├── main.py                   # Entry point (MLflow Project)
│
├── MLproject                 # MLflow project configuration
├── conda.yaml                # Conda environment
│
├── api/                       # FastAPI for inference
├── app/                       # Streamlit UI
└── README.md
```

---

## 📊 Dataset

* **Source**: SMS Spam Collection Dataset
* **Encoding**: latin1
* **Columns**:

  * `label`: ham / spam
  * `message`: SMS content

### Data Cleaning

* Dropped unused columns (`Unnamed: 2,3,4`)
* Removed duplicates (403 rows)
* Verified no missing values

---

## 🔍 Exploratory Data Analysis (EDA)

Features engineered:

* `num_characters`
* `num_words`
* `num_sentences`

EDA includes:

* Class distribution analysis
* Statistical comparison between **Ham vs Spam**
* Correlation matrix
* Visualizations (histograms, heatmaps, pairplots)

---

## 🧹 Text Preprocessing Pipeline

Applied NLP steps:

1. Lowercasing
2. Tokenization (NLTK)
3. Removing non-alphabetic tokens
4. Stopwords removal
5. Stemming (PorterStemmer)
6. Final cleaned text (`final_message`)

---

## 🤖 Model Training

* **Vectorizer**: TF-IDF

  * `ngram_range=(1,2)`
  * `max_features` (dynamic)

* **Model**: Multinomial Naive Bayes

  * `alpha` (dynamic smoothing parameter)

* **Pipeline**:

```
TF-IDF → MultinomialNB
```

* **Train/Test Split**:

  * 80% / 20%
  * Stratified sampling

---

## 📈 Model Performance

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.973 |
| Precision | 0.94  |
| Recall    | 0.83  |
| F1-Score  | 0.88  |

> ⚠️ Precision is prioritized to minimize false positives (important in spam detection).

---

## 🧪 Experiment Tracking with MLflow

Tracked using **MLflow**:

* Parameters
* Metrics
* Confusion Matrix (Artifact)
* PyFunc Model
* Model Signature
* Input Example

### MLflow Lifecycle

1. Experiment Tracking
2. PyFunc Model Wrapping
3. Model Signature Inference
4. Model Registry
5. Automatic Stage Transition

### Quality Gate

```python
MIN_PRECISION_THRESHOLD = 0.95
MIN_F1_THRESHOLD = 0.90
```

* If passed → **Production 🚀**
* Else → **Staging 🛑**

---

## 📦 Model Packaging

* Wrapped as **MLflow PyFunc**
* Accepts DataFrame with:

```
final_message
```

This makes the model:

* Framework-agnostic
* Easy to deploy
* Production-ready

---

## 🌐 Deployment

### 🔌 FastAPI

* REST API for inference
* Accepts SMS text
* Returns prediction (Spam / Ham)

### 🖥️ Streamlit

* User-friendly interface
* Real-time spam detection
* Integrated with trained model

---

## ⚙️ MLflow Project

Run the project using:

```bash
mlflow run . -P alpha=0.1 -P max_features=3000
```

---

## 🐍 Environment Setup

```yaml
name: spam_classifier_env
python: 3.9
libraries:
- mlflow
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
```

---

## 🎯 Key Takeaways

* Clean **end-to-end ML pipeline**
* Proper **NLP preprocessing**
* Strong **evaluation mindset**
* Real **MLOps practices**
* Production-ready deployment

---

## 👨‍💻 Author

**Youssef Mahmoud**
Faculty of Computers & Information
Aspiring **Data Scientist / ML Engineer**

---
URL Linked in : [https://www.linkedin.com/in/youssef-mahmoud-63b243361?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BKBSoRAFOSyucvi6vDlDfbg%3D%3D]
⭐ If you like this project, consider giving it a star on GitHub!



