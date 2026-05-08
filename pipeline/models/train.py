import logging
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def train_decision_tree(X_train, y_train, config: dict) -> DecisionTreeClassifier:
    """
    Trains a Decision Tree classifier using params from config.
    """
    params = config["models"]["decision_tree"]

    model = DecisionTreeClassifier(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"]
    )

    logger.info("training decision tree...")
    model.fit(X_train, y_train)
    logger.info("decision tree training done:-)")

    # save model to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, params["save_path"])
    logger.info("saved to %s", params["save_path"])

    return model


def train_svm(X_train, y_train, config: dict) -> SVC:
    """
    Trains an SVM classifier using params from config.
    """
    params = config["models"]["svm"]

    model = SVC(
        kernel=params["kernel"],
        C=params["C"],
        gamma=params["gamma"],
        random_state=params["random_state"],
        probability=True  # needed for ROC curve later
    )

    logger.info("training svm...")
    model.fit(X_train, y_train)
    logger.info("svm training done:-)")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, params["save_path"])
    logger.info("saved to %s", params["save_path"])

    return model