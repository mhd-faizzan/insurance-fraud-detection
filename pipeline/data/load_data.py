import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Loads raw CSV from path defined in config.
    Raises a clear error if the file isn't there yet.
    """
    raw_path = config["data"]["raw_path"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"no data file found at {raw_path} — drop your CSV in data/raw/ first"
        )

    df = pd.read_csv(raw_path)
    logger.info("loaded %d rows and %d columns from %s", len(df), len(df.columns), raw_path)
    return df