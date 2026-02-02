import argparse
import data_pipeline
import eda_text
import visualization
import model_training
import mlflow_lifecycle
import logging
from logger_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Spam Classifier MLflow Project")

    parser.add_argument("--alpha", type=float, default=0.1, help="Smoothing parameter for Naive Bayes")
    parser.add_argument("--max_features", type=int, default=3000, help="Max features for TF-IDF Vectorizer")

    args = parser.parse_args()

    logger.info(" Starting Spam Classifier End-to-End Pipeline...")
    logger.info("=" * 60)
    logger.info(f"Current Settings -> Alpha: {args.alpha}, Max Features: {args.max_features}")
    logger.info("=" * 60)

    DATA_PATH = r"C:\Users\Hedaya_city\Downloads\spam.csv"
    logger.info("\n>>> Step 1: Loading and Cleaning Raw Data")
    df_cleaned = data_pipeline.prepare_data(DATA_PATH)

    logger.info("\n>>> Step 2: Text Preprocessing and EDA")
    df_processed, corr_matrix = eda_text.perform_eda_and_pre(df_cleaned)

    logger.info("\n>>> Step 3: Generating Visualizations")
    visualization.run_visualizations(df_processed, corr_matrix)

    logger.info("\n>>> Step 4: Training MultinomialNB Model")
    training_results = model_training.train_and_evaluate_model(
        df_processed,
        alpha_val=args.alpha,
        max_feat=args.max_features
    )


    logger.info("\n>>> Step 5: Logging to MLflow & Model Registry")
    run_id = mlflow_lifecycle.run_mlflow_lifecycle(training_results)

    logger.info("\n" + "=" * 60)
    logger.info(f" Full Project Execution Completed Successfully!")
    logger.info(f" MLflow Run ID: {run_id}")
    logger.info(f" Tracking UI: Run 'mlflow ui' to see the results.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
