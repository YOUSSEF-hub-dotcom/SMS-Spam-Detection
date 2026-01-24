import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from mlflow.models.signature import infer_signature


class SpamClassifierPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_file"])

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        predictions = self.model.predict(model_input["final_message"])

        return pd.DataFrame({"is_spam": predictions})


def run_mlflow_lifecycle(results):
    print("\n================>>> Starting MLflow Lifecycle")

    best_model = results['best_model']
    acc = results['accuracy']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1_score']
    params = results['params']

    model_save_path = "spam_model.pkl"
    joblib.dump(best_model, model_save_path)

    mlflow.set_experiment("Spam_Classifier_PyFunc")

    with mlflow.start_run(run_name="MultinomialNB_SpamClassifier") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        mlflow.log_param("model_type", "MultinomialNB")
        mlflow.log_param("vectorizer", "TF-IDF (1,2)")
        mlflow.log_params(params)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })


        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        input_example = pd.DataFrame({"final_message": ["urgent prize win money now", "hello how are you"]})

        real_preds = best_model.predict(input_example["final_message"])
        output_example = pd.DataFrame({"is_spam": real_preds})

        signature = infer_signature(input_example, output_example)

        artifact_path = "spam_classifier_pyfunc"
        artifacts = {"model_file": model_save_path}

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=SpamClassifierPyFunc(feature_names=["final_message"]),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            pip_requirements=["joblib", "pandas", "scikit-learn"]
        )

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered_model_name = "Spam_Classifier"

    model_version_details = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    version = model_version_details.version

    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="Staging"
    )

    MIN_PRECISION_THRESHOLD = 0.95
    MIN_F1_THRESHOLD = 0.90

    if precision >= MIN_PRECISION_THRESHOLD and f1 >= MIN_F1_THRESHOLD:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        status = "Production 🚀"
    else:
        status = "Staging 🛑"

    print(f"\n{'═' * 50}")
    print(f" Model: {registered_model_name} | Version: {version}")
    print(f" Current Stage: {status}")
    print(f"{'═' * 50}")

    return run_id
