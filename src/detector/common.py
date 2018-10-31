from pathlib import Path

MODELS_PATH = Path("models")
DATA_PATH = Path("../data") / "detector"

if not MODELS_PATH.exists():
    MODELS_PATH.mkdir(parents=True)
