"""Inference для LogReg (TF-IDF + LogisticRegression, один sklearn Pipeline у model.joblib)."""
import json
from pathlib import Path

MODEL_FILE = "model.joblib"


def load_logreg_for_inference(run_dir):
    import joblib

    return joblib.load(Path(run_dir) / MODEL_FILE)


def logreg_version_warning(run_dir) -> str | None:
    """joblib/pickle чутливі до версії scikit-learn: якщо вона інша, ніж у Colab = попередження"""
    import sklearn

    try:
        summary = json.loads((Path(run_dir) / "run_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    trained = summary.get("sklearn_version")
    if trained and trained != sklearn.__version__:
        return (f"Модель натреновано зі scikit-learn {trained}, а встановлено {sklearn.__version__}: "
                f"результати можуть відрізнятися або завантаження може впасти. "
                f"Закріпіть scikit-learn=={trained} у requirements.txt.")
    return None


def predict_logreg(text, model):
    """Формат результату: {"prediction": 0|1, "probability": P(AI)}."""
    probability = float(model.predict_proba([str(text)])[0, 1])
    return {"prediction": int(probability >= 0.5), "probability": probability}
