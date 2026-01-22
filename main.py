import argparse
import data_pipeline
import eda_text
import visualization
import model_training
import mlflow_lifecycle


def main():
    parser = argparse.ArgumentParser(description="Spam Classifier MLflow Project")

    parser.add_argument("--alpha", type=float, default=0.1, help="Smoothing parameter for Naive Bayes")
    parser.add_argument("--max_features", type=int, default=3000, help="Max features for TF-IDF Vectorizer")

    args = parser.parse_args()

    print(" Starting Spam Classifier End-to-End Pipeline...")
    print("=" * 60)
    print(f"Current Settings -> Alpha: {args.alpha}, Max Features: {args.max_features}")
    print("=" * 60)

    DATA_PATH = r"C:\Users\Hedaya_city\Downloads\spam.csv"
    print("\n>>> Step 1: Loading and Cleaning Raw Data")
    df_cleaned = data_pipeline.prepare_data(DATA_PATH)

    print("\n>>> Step 2: Text Preprocessing and EDA")
    df_processed, corr_matrix = eda_text.perform_eda_and_pre(df_cleaned)

    print("\n>>> Step 3: Generating Visualizations")
    visualization.run_visualizations(df_processed, corr_matrix)

    print("\n>>> Step 4: Training MultinomialNB Model")
    training_results = model_training.train_and_evaluate_model(
        df_processed,
        alpha_val=args.alpha,
        max_feat=args.max_features
    )


    print("\n>>> Step 5: Logging to MLflow & Model Registry")
    run_id = mlflow_lifecycle.run_mlflow_lifecycle(training_results)

    print("\n" + "=" * 60)
    print(f" Full Project Execution Completed Successfully!")
    print(f" MLflow Run ID: {run_id}")
    print(f" Tracking UI: Run 'mlflow ui' to see the results.")
    print("=" * 60)

if __name__ == "__main__":
    main()