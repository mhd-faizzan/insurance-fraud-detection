import logging
import yaml
from pipeline.data.load_data import load_raw_data

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
    logger.info("shape: %s", str(df.shape))
    logger.info("columns: %s", list(df.columns))

    logger.info("done")


if __name__ == "__main__":
    main()