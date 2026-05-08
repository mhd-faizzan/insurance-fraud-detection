import logging
import yaml
from pipeline.data.load_data import load_raw_data
from pipeline.data.preprocess import preprocess

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

    logger.info("done")


if __name__ == "__main__":
    main()