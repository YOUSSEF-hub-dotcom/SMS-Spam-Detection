from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import numpy as np

app = FastAPI(title="SMS Spam Classifier API with MLflow")

# --- تحميل الموديل بحذر ---
model_name = "Spam_Classifier"
try:
    # بيحاول يسحب اللي في Production
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    print("✅ Loaded Production Model")
except:
    # لو ملقاش، بيسحب أحدث نسخة مسجلة (عشان الـ API يشتغل بأي حال)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
    print("⚠️ Production model not found, loaded latest version instead.")


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

        # تأكد إن المسميات دي هي اللي Streamlit هينادي عليها
        return {
            "prediction": label,
            "probability": 1.0,  # بما إن الموديل MultinomialNB مش باعت Proba حالياً
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)