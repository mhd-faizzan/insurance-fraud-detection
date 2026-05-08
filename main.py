import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main() -> None:
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    logger.info("starting pipeline")
    logger.info("config sections loaded: %s", list(config.keys()))
    logger.info("done")


if __name__ == "__main__":
    main()