import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="insurance fraud detection api")

# load models once at startup
dt_model = joblib.load("models/decision_tree.pkl")
svm_model = joblib.load("models/svm.pkl")


class Transaction(BaseModel):
    # V1 to V28 + Amount (Time is dropped during preprocessing)
    V1: float; V2: float; V3: float; V4: float
    V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount: float


def to_array(t: Transaction) -> np.ndarray:
    return np.array([[
        t.V1, t.V2, t.V3, t.V4, t.V5, t.V6, t.V7,
        t.V8, t.V9, t.V10, t.V11, t.V12, t.V13, t.V14,
        t.V15, t.V16, t.V17, t.V18, t.V19, t.V20, t.V21,
        t.V22, t.V23, t.V24, t.V25, t.V26, t.V27, t.V28,
        t.Amount
    ]])


@app.get("/health")
def health():
    return {"status": "ok", "models": ["decision_tree", "svm"]}


@app.post("/predict/decision-tree")
def predict_dt(transaction: Transaction):
    X = to_array(transaction)
    pred = dt_model.predict(X)[0]
    prob = dt_model.predict_proba(X)[0][1]
    return {
        "model": "decision_tree",
        "prediction": "fraud" if pred == 1 else "genuine",
        "probability": round(float(prob), 4)
    }


@app.post("/predict/svm")
def predict_svm(transaction: Transaction):
    X = to_array(transaction)
    pred = svm_model.predict(X)[0]
    prob = svm_model.predict_proba(X)[0][1]
    return {
        "model": "svm",
        "prediction": "fraud" if pred == 1 else "genuine",
        "probability": round(float(prob), 4)
    }