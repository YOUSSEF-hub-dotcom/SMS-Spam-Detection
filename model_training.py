import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging

logger = logging.getLogger(__name__)

def train_and_evaluate_model(df, alpha_val, max_feat):

    logger.info(f"================>>> Machine Learning Training (Alpha: {alpha_val}, Max Features: {max_feat})")

    X = df['final_message']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_feat, ngram_range=(1, 2))),
        ("nb", MultinomialNB(alpha=alpha_val))
    ])

    logger.info("Fitting the model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    logger.info("\nClassification Report:\n")
    logger.info(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Alpha={alpha_val})")
    plt.tight_layout()
    plt.show()

    training_results = {
        "best_model": pipeline,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "params": {
            "alpha": alpha_val,
            "max_features": max_feat
        }
    }

    return training_results
