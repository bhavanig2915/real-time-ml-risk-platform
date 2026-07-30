"""Generate synthetic transaction data for model training."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_TRANSACTIONS = 10_000

MERCHANT_CATEGORIES = [
    "grocery",
    "electronics",
    "travel",
    "restaurant",
    "gas_station",
    "online_retail",
]

COUNTRIES = ["US", "CA", "GB", "IN", "MX"]


def generate_transactions(
    number_of_transactions: int = NUMBER_OF_TRANSACTIONS,
) -> pd.DataFrame:
    """Generate synthetic normal and suspicious transactions."""

    rng = np.random.default_rng(RANDOM_SEED)

    amounts = np.round(
        rng.lognormal(
            mean=4.2,
            sigma=1.0,
            size=number_of_transactions,
        ),
        2,
    )

    transactions_last_hour = rng.poisson(
        lam=2,
        size=number_of_transactions,
    )

    failed_attempts = rng.poisson(
        lam=0.5,
        size=number_of_transactions,
    )

    new_device = rng.choice(
        [0, 1],
        size=number_of_transactions,
        p=[0.85, 0.15],
    )

    merchant_category = rng.choice(
        MERCHANT_CATEGORIES,
        size=number_of_transactions,
    )

    country = rng.choice(
        COUNTRIES,
        size=number_of_transactions,
        p=[0.70, 0.08, 0.07, 0.08, 0.07],
    )

    international_transaction = (country != "US").astype(int)

    average_customer_amount = np.round(
        rng.uniform(
            low=20,
            high=500,
            size=number_of_transactions,
        ),
        2,
    )

    amount_deviation = np.round(
        amounts / np.maximum(average_customer_amount, 1),
        2,
    )

    fraud_probability = (
        0.01
        + (amounts > 1000) * 0.20
        + (transactions_last_hour >= 6) * 0.20
        + (failed_attempts >= 3) * 0.20
        + (new_device == 1) * 0.10
        + (international_transaction == 1) * 0.10
        + (amount_deviation >= 5) * 0.20
    )

    fraud_probability = np.clip(
        fraud_probability,
        0,
        0.95,
    )

    is_fraud = rng.binomial(
        n=1,
        p=fraud_probability,
    )

    dataframe = pd.DataFrame(
        {
            "transaction_id": [
                f"TX-{uuid4().hex[:12].upper()}"
                for _ in range(number_of_transactions)
            ],
            "customer_id": [
                f"CUST-{rng.integers(1000, 5000)}"
                for _ in range(number_of_transactions)
            ],
            "amount": amounts,
            "merchant_category": merchant_category,
            "country": country,
            "new_device": new_device,
            "transactions_last_hour": transactions_last_hour,
            "failed_attempts": failed_attempts,
            "average_customer_amount": average_customer_amount,
            "amount_deviation": amount_deviation,
            "international_transaction": international_transaction,
            "is_fraud": is_fraud,
        }
    )

    return dataframe


def save_dataset(dataframe: pd.DataFrame) -> Path:
    """Save the generated dataset to the data/raw folder."""

    project_root = Path(__file__).resolve().parents[2]
    output_directory = project_root / "data" / "raw"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / "transactions.csv"
    dataframe.to_csv(output_file, index=False)

    return output_file


def main() -> None:
    """Generate and save the dataset."""

    dataframe = generate_transactions()
    output_file = save_dataset(dataframe)

    fraud_count = int(dataframe["is_fraud"].sum())
    fraud_rate = dataframe["is_fraud"].mean() * 100

    print(f"Generated {len(dataframe):,} transactions")
    print(f"Fraudulent transactions: {fraud_count:,}")
    print(f"Fraud rate: {fraud_rate:.2f}%")
    print(f"Dataset saved to: {output_file}")


if __name__ == "__main__":
    main()