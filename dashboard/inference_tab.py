"""Вкладка «Inference»: вставити текст → отримати P(AI) від вибраної моделі та (за потреби) від моделей порівняння.

Залежить від пакета src/inference (лежить у корені репозиторію). torch / transformers імпортуються
лише коли справді потрібні, тож решта дашборду працює і без них.

Налаштування (змінні середовища або .streamlit/secrets.toml):
    HF_MODEL_REPO   id репозиторію моделей на Hugging Face Hub (для великих *.pt, напр. DistilBERT)
    HF_TOKEN        токен, якщо репозиторій приватний
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent       # dashboard/ лежить у корені репозиторію
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LABELS = {0: "Людина", 1: "AI"}
HELP = (
    "Модель дає P(AI) — ймовірність, що текст згенеровано; вердикт «AI», якщо P(AI) ≥ 0.5 (як під час "
    "навчання й тесту). BiLSTM і DistilBERT обрізають текст до max_length токенів. Моделі навчені на "
    "даних свого сценарію, тож для текстів з іншого домену результат може бути ненадійним."
)


def _setting(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:           # secrets.toml немає
        return None


@st.cache_resource(show_spinner="Завантажується модель…")
def _load_predictor(run_dir: str, kind: str, hf_repo: str | None, mtime: float):
    from src.inference import api

    return api.load_predictor(run_dir, kind, hf_repo, _setting("HF_TOKEN"))


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _predict_one(run, hf_repo: str | None, text: str) -> dict:
    """Рядок результату для однієї моделі; помилки не падають, а потрапляють у «примітку»."""
    from src.inference import api

    row = {"модель": run.label, "тип": run.kind, "вердикт": None, "P(AI)": None, "примітка": ""}
    ok, reason = api.check_inference(run.path, run.kind, hf_repo)
    if not ok:
        row["примітка"] = f"недоступно: {reason}"
        return row
    try:
        predictor = _load_predictor(run.path, run.kind, hf_repo, _mtime(Path(run.path) / "run_summary.json"))
        result = predictor.predict(text)
        row["вердикт"] = LABELS[result["prediction"]]
        row["P(AI)"] = result["probability"]
        notes = ([reason] if reason else []) + predictor.warnings
        row["примітка"] = "; ".join(notes)
    except ImportError as e:
        row["примітка"] = f"не встановлено залежність «{e.name}» (див. requirements-inference.txt)"
    except Exception as e:      # noqa: BLE001 — показуємо користувачу, а не валимо весь дашборд
        row["примітка"] = f"помилка: {type(e).__name__}: {e}"
    return row


def render(current, compare_runs, show_cards) -> None:
    """current — вибрана модель; compare_runs — моделі з бічної панелі «Порівняння»;
    show_cards — функція карток із app.py."""
    st.markdown("#### Inference", help=HELP)

    try:
        from src.inference import api  # noqa: F401
    except ImportError:
        st.error(f"Не знайдено пакет `src/inference` у `{REPO_ROOT}`. Додайте його в корінь репозиторію.")
        return

    hf_repo = _setting("HF_MODEL_REPO")
    runs = [current] + list(compare_runs)

    text = st.text_area("Текст для перевірки", height=220, key="infer_text",
                        placeholder="Вставте сюди текст (есе, відповідь, абзац…). Максимальна довжина - 512 (для трансформера) або 1024 слів (залишок тексту не враховується)")
    run_button = st.button("Передбачити", type="primary", disabled=not text.strip())
    if len(runs) > 1:
        st.caption(f"Той самий текст буде прогнано через {len(runs)} моделі(ей): вибрану та вибрані в «Порівнянні».")

    if run_button:
        with st.spinner("Рахує…"):
            rows = [_predict_one(run, hf_repo, text) for run in runs]
        st.session_state["infer_result"] = {
            "rows": rows,
            "text_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
            "chars": len(text),
            "words": len(text.split()),
        }

    result = st.session_state.get("infer_result")
    if not result:
        return

    if result["text_hash"] != hashlib.md5(text.encode("utf-8")).hexdigest():
        st.caption("⚠️ Показано результат для попереднього тексту — натисніть «Передбачити», щоб оновити.")

    rows = result["rows"]
    main = rows[0]
    if main["P(AI)"] is None:
        st.warning(f"Для моделі «{main['модель']}» результату немає — {main['примітка']}")
    else:
        p = main["P(AI)"]
        show_cards([
            {"label": "Вердикт", "value": main["вердикт"], "sub": f"{main['модель']}", "accent": True},
            {"label": "P(AI)", "value": f"{p:.1%}"},
            {"label": "P(людина)", "value": f"{1 - p:.1%}"},
            {"label": "Довжина тексту", "value": f"{result['words']} слів", "sub": f"{result['chars']} символів"},
        ], min_width=150)
        if main["примітка"]:
            st.caption(main["примітка"])

    if len(rows) > 1:
        st.markdown("#### Порівняння моделей")
    table = pd.DataFrame(rows)
    if len(rows) > 1 or main["P(AI)"] is None:
        st.dataframe(
            table, hide_index=True,
            column_config={"P(AI)": st.column_config.ProgressColumn("P(AI)", min_value=0.0, max_value=1.0, format="%.3f")},
        )
