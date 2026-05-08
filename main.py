import logging
import yaml
from pipeline.data.load_data import load_raw_data
from pipeline.data.preprocess import preprocess
from pipeline.models.train import train_decision_tree, train_svm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main() -> None:
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    logger.info("starting pipeline")

    df = load_raw_data(config)
    X_train, X_test, y_train, y_test = preprocess(df, config)

    dt_model = train_decision_tree(X_train, y_train, config)
    svm_model = train_svm(X_train, y_train, config)

    logger.info("done")


if __name__ == "__main__":
    main()