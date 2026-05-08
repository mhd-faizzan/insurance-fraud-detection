import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

logger = logging.getLogger(__name__)


def evaluate(dt_model, svm_model, X_test, y_test, config: dict) -> None:
    """
    Evaluates both models — confusion matrix, classification report, ROC curve.
    Saves plots to assets/.
    """
    os.makedirs("assets", exist_ok=True)

    models = {
        "decision_tree": dt_model,
        "svm": svm_model
    }

    results = {}

    for name, model in models.items():
        logger.info("--- %s ---", name)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, target_names=["genuine", "fraud"], output_dict=True)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {
            "precision": report["fraud"]["precision"],
            "recall": report["fraud"]["recall"],
            "f1": report["fraud"]["f1-score"],
            "auc": auc
        }

        logger.info("\n%s", classification_report(y_test, y_pred, target_names=["genuine", "fraud"]))

        # confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["genuine", "fraud"],
                    yticklabels=["genuine", "fraud"])
        plt.title(f"confusion matrix — {name}")
        plt.ylabel("actual")
        plt.xlabel("predicted")
        plt.tight_layout()
        plt.savefig(f"assets/confusion_matrix_{name}.png")
        plt.close()
        logger.info("confusion matrix saved for %s", name)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", label="random")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title(f"ROC curve — {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"assets/roc_curve_{name}.png")
        plt.close()
        logger.info("ROC curve saved for %s | AUC: %.4f", name, auc)

    # summary comparison
    logger.info("model comparison summary")
    logger.info("%-20s %-10s %-10s %-10s %-10s", "model", "precision", "recall", "f1", "auc")
    for name, metrics in results.items():
        logger.info("%-20s %-10.2f %-10.2f %-10.2f %-10.4f",
                    name,
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["auc"])