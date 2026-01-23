from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="SMS Spam Detection API with MLflow")

model_name = "Spam_Classifier"
try:
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    print("Loaded Production Model")
except:
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
    print(" Production model not found, loaded latest version instead.")


class MessageInput(BaseModel):
    message: str


@app.post("/predict")
def predict_message(input_data: MessageInput):
    if not input_data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        data = pd.DataFrame({"final_message": [input_data.message]})
        prediction = model.predict(data)

        res = int(prediction[0])
        label = "Spam" if res == 1 else "Ham"

        return {
            "prediction": label,
            "probability": 1.0,
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
