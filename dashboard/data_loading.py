# не залежить від Streamlit, працює при відсутніх файлах
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_COLUMNS = ["dataset", "prompt_name", "source"]
# Метрики, які є й у val_*, і у test_*. Порядок = порядок у випадних списках.
METRIC_NAMES = ["macro_f1", "f1_ai", "roc_auc", "accuracy", "precision_ai", "recall_ai", "loss"]
BAR_METRICS = ["accuracy", "precision_ai", "recall_ai", "f1_ai", "macro_f1", "roc_auc"]  # без loss: інша шкала
LOWER_IS_BETTER = {"loss"}
ROLES = ("train", "val", "test")
# Типи моделей у фільтрі (порядок = порядок у UI). bilstm_* визначаються за config.embedding.
MODEL_KINDS = ["bilstm", "bilstm_glove", "bilstm_fasttext", "logreg", "distilbert"]


# допоміжне
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def overrides_str(overrides: dict | None) -> str:
    if not overrides:
        return "base"
    return ", ".join(f"{k}={v}" for k, v in overrides.items())


# запуск
@dataclass
class Run:
    scenario: str
    name: str                      # назва папки запуску (унікальна в межах сценарію)
    path: str                      # .../training_logs/<name>
    test_path: str                 # .../test_results/<name>
    model_type: str
    overrides: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    data: dict | None = None
    summary: dict = field(default_factory=dict)
    has_history: bool = False
    test_level: str = "none"       # "full" (є test_results/) | "metrics" (лише в summary) | "none"
    status: str = "empty"          # "finished" | "unfinished" | "empty"

    @property
    def label(self) -> str:
        """Назва моделі в UI = назва запуску (папки)."""
        return self.name

    @property
    def kind(self) -> str:
        """Тип моделі для фільтра: bilstm | bilstm_glove | bilstm_fasttext | logreg | distilbert.
        У config.json усі BiLSTM мають model_type="bilstm"; різновид визначає config.embedding
        (None | "glove" | "fasttext"). Якщо config.json немає — шукаємо підказку в назві запуску."""
        if self.model_type == "bilstm":
            embedding = self.config.get("embedding")
            if embedding in ("glove", "fasttext"):
                return f"bilstm_{embedding}"
            if not self.config:
                for e in ("glove", "fasttext"):
                    if e in self.name.lower():
                        return f"bilstm_{e}"
        return self.model_type

    @property
    def has_test(self) -> bool:
        return self.test_level != "none"

    @property
    def val_metrics(self) -> dict:
        return self.summary.get("val_metrics") or {}

    @property
    def test_metrics(self) -> dict:
        return self.summary.get("test_metrics") or {}

    def file(self, name: str) -> Path:
        return Path(self.path) / name


def list_scenarios(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "training_logs").is_dir()
    )


def scenario_signature(root: str | Path, scenario: str) -> tuple:
    """(кількість файлів, остання зміна) — ключ кешу: після нового push дані перечитаються."""
    base = Path(root) / scenario
    n, latest = 0, 0.0
    for dirpath, _, files in os.walk(base):
        for f in files:
            n += 1
            try:
                latest = max(latest, os.path.getmtime(os.path.join(dirpath, f)))
            except OSError:
                pass
    return n, latest


def load_runs(root: str | Path, scenario: str) -> list[Run]:
    base = Path(root) / scenario
    logs = base / "training_logs"
    if not logs.is_dir():
        return []

    runs = []
    for d in sorted(p for p in logs.iterdir() if p.is_dir()):
        cfg_json = _read_json(d / "config.json")
        summary = _read_json(d / "run_summary.json")
        test_dir = base / "test_results" / d.name

        model_type = cfg_json.get("model_type") or summary.get("model_type") or (d.name.split("__")[0] if "__" in d.name else "unknown")
        overrides = cfg_json.get("overrides") or summary.get("overrides") or {}
        data = cfg_json.get("data") or summary.get("data")

        has_history = (d / "history.csv").exists()
        if summary.get("val_metrics"):            # run_summary з val_metrics пишеться лише в кінці навчання
            status = "finished"
        elif has_history:
            status = "unfinished"                 # тренування триває або було перервано
        else:
            status = "empty"

        if (test_dir / "test_metrics.csv").exists():
            test_level = "full"
        elif summary.get("test_metrics"):
            test_level = "metrics"
        else:
            test_level = "none"

        runs.append(Run(
            scenario=scenario, name=d.name, path=str(d), test_path=str(test_dir),
            model_type=model_type, overrides=overrides, config=cfg_json.get("config") or {},
            data=data, summary=summary, has_history=has_history,
            test_level=test_level, status=status,
        ))
    return runs


def kinds_present(runs: list[Run]) -> list[str]:
    """Типи моделей, що є в сценарії, у фіксованому порядку MODEL_KINDS (невідомі — в кінці)."""
    found = {r.kind for r in runs}
    return [k for k in MODEL_KINDS if k in found] + sorted(found - set(MODEL_KINDS))


def sort_runs(runs: list[Run], key: str = "macro_f1") -> list[Run]:
    """За спаданням val-метрики; запуски без метрики — в кінці."""
    def k(r: Run):
        v = r.val_metrics.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return (1, 0.0)
        return (0, v if key in LOWER_IS_BETTER else -v)
    return sorted(runs, key=k)


# дані сценарію
def scenario_data(runs: list[Run]) -> dict | None:
    """Опис сценарію (plan + sizes) — однаковий у всіх запусків, беремо перший, де він є."""
    for r in runs:
        if r.data:
            return r.data
    return None


def plan_datasets(data: dict | None) -> dict[str, list[str]]:
    plan = (data or {}).get("plan") or {}
    return {role: list(plan.get(role) or []) for role in ROLES}


def sizes_table(data: dict | None) -> pd.DataFrame:
    rows = (data or {}).get("sizes") or []
    return pd.DataFrame(rows)


# таблиці
def leaderboard(runs: list[Run], include_test: bool = False) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {
            "run": r.name,
            "type": r.kind,
            "params": overrides_str(r.overrides),
            "status": r.status,
            "best_epoch": r.summary.get("best_epoch"),
            "minutes": (r.summary["duration_seconds"] / 60) if r.summary.get("duration_seconds") else None,
        }
        for m in ("loss", "macro_f1", "f1_ai", "roc_auc"):
            row[f"val_{m}"] = r.val_metrics.get(m)
        if include_test:
            for m in ("loss", "macro_f1", "f1_ai", "roc_auc"):
                row[f"test_{m}"] = r.test_metrics.get(m)
        rows.append(row)
    return pd.DataFrame(rows)


# файли запуску
def load_history(run: Run) -> pd.DataFrame | None:
    path = run.file("history.csv")
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    if df.empty or "epoch" not in df:
        return None
    return df.sort_values("epoch").reset_index(drop=True)


def load_group_history(run: Run) -> pd.DataFrame | None:
    path = run.file("group_metrics.csv")
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return None


def read_log(run: Run, tail: int = 200) -> str:
    path = run.file("training.log")
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-tail:])


def best_info(run: Run, history: pd.DataFrame | None) -> dict:
    """Найкраща епоха та метрики на ній. Для незавершених запусків рахується з history.csv."""
    monitor = run.summary.get("monitor") or run.config.get("monitor")
    metrics = dict(run.val_metrics)
    epoch = run.summary.get("best_epoch")
    derived = False

    if not metrics and history is not None and monitor and f"val_{monitor}" in history:
        col = history[f"val_{monitor}"]
        idx = col.idxmin() if monitor in LOWER_IS_BETTER else col.idxmax()
        epoch = float(history.loc[idx, "epoch"])
        metrics = {c[4:]: float(history.loc[idx, c]) for c in history.columns if c.startswith("val_")}
        derived = True

    score = metrics.get(monitor) if monitor else None
    return {"epoch": epoch, "monitor": monitor, "score": score, "metrics": metrics, "derived": derived}


def config_table(run: Run) -> pd.DataFrame:
    rows = [
        {"параметр": k, "значення": json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else str(v),
         "змінено": "●" if k in run.overrides else ""}
        for k, v in run.config.items()
    ]
    return pd.DataFrame(rows)


# тест
def load_test_details(run: Run) -> dict:
    """metrics (dict), confusion (df|None), groups ({група: df}), predictions_path (str|None)."""
    out = {"metrics": dict(run.test_metrics), "confusion": None, "groups": {}, "predictions_path": None}
    tdir = Path(run.test_path)
    if not tdir.is_dir():
        return out

    metrics_csv = tdir / "test_metrics.csv"
    if metrics_csv.exists():
        try:
            row = pd.read_csv(metrics_csv).iloc[0].to_dict()
            out["metrics"] = {k: float(v) for k, v in row.items()}
        except (OSError, ValueError, IndexError):
            pass

    cm_csv = tdir / "confusion_matrix.csv"
    if cm_csv.exists():
        try:
            out["confusion"] = pd.read_csv(cm_csv, index_col=0)
        except (OSError, ValueError):
            pass

    for g in GROUP_COLUMNS:
        p = tdir / f"metrics_by_{g}.csv"
        if p.exists():
            try:
                out["groups"][g] = pd.read_csv(p)
            except (OSError, ValueError):
                pass

    pred = _first_existing(tdir / "test_predictions.csv", tdir / "test_predictions.csv.gz")
    out["predictions_path"] = str(pred) if pred else None
    return out


def load_predictions(path: str) -> pd.DataFrame:
    """Лише колонки, потрібні для ROC / PR / розподілу ймовірностей (економить пам'ять)."""
    return pd.read_csv(path, usecols=["y_true", "y_pred", "y_prob"])


def group_frame(df: pd.DataFrame, group_type: str, epoch: float | None = None) -> pd.DataFrame:
    """Приводить таблицю метрик за групами до єдиного вигляду з колонкою `group`.

    * metrics_by_<група>.csv (test): колонка групи названа як сама група;
    * group_metrics.csv (val під час навчання): рядки всіх груп разом + `group_type`, `epoch`.
    """
    if "group_type" in df.columns:
        df = df[df["group_type"] == group_type]
        if epoch is not None:
            df = df[np.isclose(df["epoch"], epoch)]
    if group_type not in df.columns:
        return pd.DataFrame(columns=["group", "n", *BAR_METRICS])
    out = df.rename(columns={group_type: "group"}).copy()
    out["group"] = out["group"].astype(str)
    service = ["group_type", "epoch"] + [g for g in GROUP_COLUMNS if g != group_type]   # колонки інших груп (NaN)
    return out.drop(columns=[c for c in service if c in out.columns]).reset_index(drop=True)
