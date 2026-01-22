import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import pandas as pd
from mlflow.models.signature import infer_signature


class SpamClassifierPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input["final_message"])


def run_mlflow_lifecycle(results):

    print("================>>> Starting MLflow Lifecycle")

    best_model = results['best_model']
    acc = results['accuracy']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1_score']
    params = results['params']

    mlflow.set_experiment("Spam_Classifier_PyFunc")

    with mlflow.start_run(run_name="MultinomialNB_SpamClassifier") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        mlflow.log_param("model_type", "MultinomialNB")
        mlflow.log_param("vectorizer", "TF-IDF (1,2)")
        mlflow.log_params(params)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        input_example = pd.DataFrame({"final_message": ["sample text for spam detection"]})
        pyfunc_model = SpamClassifierPyFunc(best_model)
        output_example = pyfunc_model.predict(context=None, model_input=input_example)

        signature = infer_signature(input_example, output_example)

        pyfunc_model = SpamClassifierPyFunc(best_model)
        artifact_path = "spam_classifier_pyfunc"

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=pyfunc_model,
            signature=signature,
            input_example=input_example
        )


    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered_model_name = "Spam_Classifier"

    model_version_details = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    version = model_version_details.version
    print(f"Model registered with version: {version}")

    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="Staging"
    )

    MIN_PRECISION_THRESHOLD = 0.95
    MIN_F1_THRESHOLD = 0.90
    status = "Staging "

    if precision >= MIN_PRECISION_THRESHOLD and f1 >= MIN_F1_THRESHOLD:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        status = "Production "
        print(f" Quality Gate Passed (Precision: {precision:.2f}). Moved to Production.")
    else:
        print(f"Quality Gate Failed (Precision: {precision:.2f}). Kept in Staging.")

    print(f"\n{'═' * 50}")
    print(f" Model: {registered_model_name} | Version: {version}")
    print(f" Current Stage: {status}")
    print(f"{'═' * 50}")

    return run_id
