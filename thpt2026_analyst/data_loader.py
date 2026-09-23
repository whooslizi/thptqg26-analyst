import os
from typing import Tuple, Dict, Any
import pandas as pd

DEFAULT_DATA_PATH = os.path.join("data", "raw_diem_thpt_2026.csv")


def load_thpt_2026_data(
    source_url: str = None,
    local_path: str = DEFAULT_DATA_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads THPT 2026 score dataset from local CSV file or optional remote endpoint.

    Args:
        source_url: Optional URL to download raw dataset.
        local_path: Local file system path to load CSV from.

    Returns:
        Tuple containing loaded DataFrame and metadata dictionary.
    """
    metadata = {
        "status": "pending",
        "loaded_from": None,
    }

    if os.path.exists(local_path):
        df = pd.read_csv(local_path, dtype={"sbd": str})
        metadata["status"] = "success_local"
        metadata["loaded_from"] = local_path
        return df, metadata

    if source_url:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            df = pd.read_csv(source_url, dtype={"sbd": str})
            df.to_csv(local_path, index=False, encoding="utf-8")
            metadata["status"] = "success_remote"
            metadata["loaded_from"] = source_url
            return df, metadata
        except Exception as err:
            metadata["status"] = f"error: {str(err)}"

    columns = [
        "sbd",
        "dm1",
        "dm2",
        "dm3",
        "dm4",
        "dm5",
        "dm6",
        "dm7",
        "dm8",
        "dm9",
        "dm10",
        "dm11",
        "dm12",
        "dm13",
    ]
    return pd.DataFrame(columns=columns), metadata
