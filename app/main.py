from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Real-Time ML Risk Platform",
    version="1.0.0"
)

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    country: str

@app.get("/")
def home():
    return {
        "message": "Welcome to the Real-Time ML Risk Platform"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(transaction: Transaction):
    score = 0.1

    if transaction.amount > 1000:
        score += 0.4

    return {
        "transaction_id": transaction.transaction_id,
        "risk_score": score,
        "prediction": "High Risk" if score > 0.4 else "Low Risk"
    }