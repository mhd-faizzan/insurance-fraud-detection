import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame, config: dict):
    """
    Scales Amount and Time, splits into train/test,
    then applies SMOTE to fix class imbalance on training set only.
    """
    # drop Time — not useful as a raw feature for fraud detection
    df = df.drop(columns=["Time"])

    # scale Amount to match the range of V1-V28
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # split before SMOTE 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y  # keeps fraud ratio same in both splits
    )

    logger.info("train size: %d, test size: %d", len(X_train), len(X_test))
    logger.info("fraud cases in train before SMOTE: %d", y_train.sum())

    # fix class imbalance on training set only
    smote = SMOTE(random_state=config["data"]["random_state"])
    X_train, y_train = smote.fit_resample(X_train, y_train)

    logger.info("train size after SMOTE: %d", len(X_train))
    logger.info("fraud cases after SMOTE: %d", y_train.sum())

    return X_train, X_test, y_train, y_test