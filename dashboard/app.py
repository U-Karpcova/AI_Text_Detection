from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import data_loading as dl
import plots

st.set_page_config(page_title="AI Detection — експерименти", page_icon="🔎", layout="wide")

ROOT = Path(os.environ.get("RUNS_ROOT", Path(__file__).resolve().parent.parent / "runs"))

# картки
# Колір фону й рамки напівпрозорі, текст успадковується
CARD_CSS = """
<style>
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: .6rem; margin: .2rem 0 1rem; }
.card { padding: .55rem .8rem .65rem; border-radius: .6rem; border: 1px solid rgba(128,128,128,.28);
        background: rgba(128,128,128,.09); min-width: 0; }
.card .lbl { font-size: .74rem; opacity: .72; line-height: 1.2; }
.card .val { font-size: 1.2rem; font-weight: 600; line-height: 1.35; overflow-wrap: anywhere; }
.card .sub { font-size: .7rem; opacity: .6; }
.card.accent { border-color: #ff4b4b; background: rgba(255,75,75,.09); }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)


# Порядок стовпчиків таблиці на вкладці "Огляд сценарію".
# стовпчики, яких тут немає, виводяться в кінці
OVERVIEW_COLUMNS = [
    "run",
    "type",
    "status",
    "best_epoch",
    "val_loss",
    "val_macro_f1",
    "val_f1_ai",
    "val_roc_auc",
    "test_loss",
    "test_macro_f1",
    "test_f1_ai",
    "test_roc_auc",
    "minutes",
    "params",
]

LOSS_HELP = (
    "★ — найкраща епоха. Суцільна лінія — val, пунктир — train. Крапкова лінія на всю ширину — "
    "baseline з одним значенням (LogReg). train_loss усереднено за інтервал при ввімкненому dropout, "
    "val_loss — в eval-режимі, тому їхні абсолютні значення порівнювати не зовсім коректно."
)
METRIC_HELP = "★ — найкраща епоха. Крапкова лінія на всю ширину — baseline з одним значенням (LogReg)."


def show_cards(items: list[dict]):
    """items: [{"label", "value", "sub"?, "accent"?}] — сітка карток; значення переносяться, а не обрізаються."""
    parts = []
    for it in items:
        sub = f'<div class="sub">{html.escape(str(it["sub"]))}</div>' if it.get("sub") else ""
        parts.append(
            f'<div class="card{" accent" if it.get("accent") else ""}">'
            f'<div class="lbl">{html.escape(str(it["label"]))}</div>'
            f'<div class="val">{html.escape(str(it["value"]))}</div>{sub}</div>'
        )
    st.markdown(f'<div class="cards">{"".join(parts)}</div>', unsafe_allow_html=True)


# кешування, дані перечитуються після push
def _mtime(path: Path | str | None) -> float:
    try:
        return os.path.getmtime(path) if path else 0.0
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def cached_runs(root: str, scenario: str, signature: tuple) -> list[dl.Run]:
    return dl.load_runs(root, scenario)


@st.cache_data(show_spinner=False)
def _history(_run: dl.Run, path: str, mtime: float):
    return dl.load_history(_run)


@st.cache_data(show_spinner=False)
def _group_history(_run: dl.Run, path: str, mtime: float):
    return dl.load_group_history(_run)


@st.cache_data(show_spinner=False)
def _test_details(_run: dl.Run, path: str, mtime: float):
    return dl.load_test_details(_run)


@st.cache_data(show_spinner=False)
def _predictions(path: str, mtime: float):
    return dl.load_predictions(path)


def get_history(run: dl.Run):
    p = run.file("history.csv")
    return _history(run, str(p), _mtime(p))


def get_group_history(run: dl.Run):
    p = run.file("group_metrics.csv")
    return _group_history(run, str(p), _mtime(p))


def get_test(run: dl.Run) -> dict:
    p = Path(run.test_path) / "test_metrics.csv"
    return _test_details(run, str(p), _mtime(p) + _mtime(run.file("run_summary.json")))


def get_predictions(details: dict):
    path = details.get("predictions_path")
    return _predictions(path, _mtime(path)) if path else None


#  утиліти UI
def fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def select_run(name: str):
    st.session_state["selected_run"] = name


def set_compare(names: list[str]):
    st.session_state["compare_runs"] = names


def metric_cards(metrics: dict, names=dl.METRIC_NAMES):
    items = [(n, metrics[n]) for n in names if n in metrics]
    if not items:
        return
    for col, (name, value) in zip(st.columns(len(items)), items):
        col.metric(name, fmt(value))


def build_curves(runs: list[dl.Run]) -> tuple[list[plots.Curve], list[str]]:
    """Криві для вибраної моделі (runs[0]) і порівнюваних. Колір залежить від позиції в runs."""
    curves, missing = [], []
    for i, run in enumerate(runs):
        history = get_history(run)
        if history is None:
            missing.append(run.label)
            continue
        curves.append(plots.Curve(
            label=run.label, color=plots.color_for(i), history=history,
            best_epoch=dl.best_info(run, history)["epoch"],
        ))
    return curves, missing


# бічна панель
with st.sidebar:
    st.header("🔎 AI-detection")

    scenarios = dl.list_scenarios(ROOT)
    if not scenarios:
        st.error(f"У `{ROOT}` немає сценаріїв (очікую `<сценарій>/training_logs/`).")
        st.stop()

    scenario = st.selectbox("Сценарій даних", scenarios, key="scenario")
    runs = cached_runs(str(ROOT), scenario, dl.scenario_signature(ROOT, scenario))
    if not runs:
        st.warning("У цьому сценарії ще немає запусків.")
        st.stop()
    by_name = {r.name: r for r in runs}

    # дані сценарію (згорнуто; детальна таблиця на вкладці Інформація)
    plan = dl.plan_datasets(dl.scenario_data(runs))
    with st.expander("Склад вибірок"):
        for role, title in (("train", "Train"), ("val", "Val"), ("test", "Test")):
            st.markdown(f"**{title}:** {', '.join(plan[role]) or '—'}")
        if plan["test"] and set(plan["test"]) - set(plan["train"]):
            st.caption("⚠️ test містить датасет, якого немає в train (out-of-domain)")

    st.divider()

    # вибір моделі
    types = dl.kinds_present(runs)
    chosen_types = st.multiselect("Тип моделі", types, default=types, key=f"types_{scenario}")
    visible = dl.sort_runs([r for r in runs if r.kind in chosen_types])
    options = [r.name for r in visible]
    if not options:
        st.warning("Жодної моделі за цим фільтром.")
        st.stop()

    def fmt_run(name: str) -> str:
        r = by_name[name]
        icons = ("⏳ " if r.status != "finished" else "") + ("🧪 " if r.has_test else "")
        return f"{icons}{r.label}"

    if st.session_state.get("selected_run") not in options:
        st.session_state.pop("selected_run", None)
    selected_name = st.selectbox("Модель", options, format_func=fmt_run, key="selected_run",
                                 help="Список відсортовано за val macro_f1. 🧪 є тестовий запуск, ⏳ запуск не завершено.")
    current = by_name[selected_name]

    st.divider()

    # порівняння (спільне для вкладок Тренування і Тестування)
    others = [r for r in dl.sort_runs(runs) if r.name != selected_name]
    other_names = [r.name for r in others]
    st.session_state["compare_runs"] = [n for n in st.session_state.get("compare_runs", []) if n in other_names]
    compare_names = st.multiselect("Порівняння", other_names, format_func=fmt_run, key="compare_runs",
                                   help="Моделі з цього ж сценарію; малюються разом із вибраною.")
    b1, b2, b3 = st.columns(3)
    b1.button("Топ-3", on_click=set_compare, args=(other_names[:3],), help="Три найкращі за val macro_f1")
    b2.button("Той же тип", on_click=set_compare,
              args=([r.name for r in others if r.kind == current.kind],))
    b3.button("Скинути", on_click=set_compare, args=([],))
    compare_runs = [by_name[n] for n in compare_names]

    st.divider()
    if st.button("🔄 Оновити дані"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Логи: `{ROOT}`")


# вкладки
st.subheader(current.label)
st.caption(f"Сценарій `{scenario}` · тип `{current.kind}`")

tab_overview, tab_info, tab_train, tab_test = st.tabs(
    ["📊 Огляд сценарію", "ℹ️ Інформація", "📈 Тренування", "🧪 Тестування"]
)


# огляд сценарію
with tab_overview:
    show_test = st.checkbox(
        "Показати test-метрики", value=False,
        help="Test призначений для фінальної оцінки вже обраних моделей, а не для вибору між ними.",
    )
    table = dl.leaderboard(visible, include_test=show_test)
    ordered = [c for c in OVERVIEW_COLUMNS if c in table.columns]
    table = table[ordered + [c for c in table.columns if c not in ordered]]
    config = {c: st.column_config.NumberColumn(format="%.4f") for c in table.columns if c.startswith(("val_", "test_"))}
    config["best_epoch"] = st.column_config.NumberColumn(format="%.2f")
    config["minutes"] = st.column_config.NumberColumn(format="%.1f")
    event = st.dataframe(table, hide_index=True, column_config=config, key="leaderboard",
                         on_select="rerun", selection_mode="single-row")
    st.caption("Відсортовано за val macro_f1. Клікніть рядок, щоб відкрити модель на інших вкладках.")
    picked_rows = event.selection.rows
    if picked_rows:
        picked = table.iloc[picked_rows[0]]["run"]
        if picked != selected_name:
            st.button(f"Відкрити «{by_name[picked].label}»", on_click=select_run, args=(picked,))


# Інформація
with tab_info:
    history = get_history(current)
    best = dl.best_info(current, history)

    if current.status == "unfinished":
        st.warning("Запуск не завершено (триває або був перерваний): найкращу епоху обчислено з history.csv.")
    elif current.status == "empty":
        st.info("Для цього запуску ще немає ні history.csv, ні run_summary.json.")

    s = current.summary
    st.markdown("#### Основна інформація")
    show_cards([
        {"label": "Тип моделі", "value": current.kind},
        {"label": "Навчальних параметрів",
         "value": f"{s['trainable_params']:,}".replace(",", " ") if s.get("trainable_params") else "—"},
        {"label": "Тривалість",
         "value": f"{s['duration_seconds'] / 60:.1f} хв" if s.get("duration_seconds") else "—"},
    ])

    st.markdown("#### Найкраща епоха")
    monitor = best["monitor"]
    cards = [{
        "label": "Епоха",
        "value": f"{best['epoch']:g}" if best["epoch"] is not None else "—",
        "sub": f"вибір за {monitor}" if monitor else None,
        "accent": True,
    }]
    cards += [
        {"label": name, "value": fmt(best["metrics"][name]), "accent": name == monitor}
        for name in dl.METRIC_NAMES if name in best["metrics"]
    ]
    show_cards(cards)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Конфігурація")
        cfg_table = dl.config_table(current)
        if cfg_table.empty:
            st.caption("config.json відсутній.")
        else:
            st.dataframe(cfg_table, hide_index=True, height=min(35 * len(cfg_table) + 38, 520))
            st.caption("● — параметр змінено відносно базової конфігурації (overrides)")
            with st.expander("Повний config.json"):
                st.json({"model_type": current.model_type, "overrides": current.overrides,
                         "config": current.config, "data": current.data})
    with right:
        st.markdown("#### Дані")
        plan_c = dl.plan_datasets(current.data)
        for role in dl.ROLES:
            st.markdown(f"**{role}:** {', '.join(plan_c[role]) or '—'}")
        size_c = dl.sizes_table(current.data)
        if not size_c.empty:
            st.dataframe(size_c, hide_index=True)

    log_text = dl.read_log(current)
    if log_text:
        with st.expander("training.log (останні рядки)"):
            st.code(log_text, language="text")


# Тренування
with tab_train:
    curves, missing = build_curves([current] + compare_runs)

    if not curves or curves[0].label != current.label:
        st.info("Для вибраної моделі немає history.csv — кривих навчання нема.")
    if missing:
        st.caption("Без history.csv (не показано): " + ", ".join(missing))

    if curves:
        # Loss: перемикачі поруч із заголовком графіка
        h1, h2, h3 = st.columns([3, 1.3, 1.5])
        h1.markdown("#### Loss", help=LOSS_HELP)
        show_train = h2.checkbox("train loss", value=False, key="show_train_loss")
        log_y = h3.checkbox("log-шкала", value=False, key="log_loss")
        st.plotly_chart(plots.loss_figure(curves, show_train, log_y), key="fig_loss")

        # Метрика валідації: вибір метрики поруч із заголовком графіка
        m1, m2 = st.columns([3, 2])
        m1.markdown("#### Метрика валідації", help=METRIC_HELP)
        metric = m2.selectbox("Метрика валідації", dl.METRIC_NAMES[:-1], key="val_metric",
                              label_visibility="collapsed")
        st.plotly_chart(plots.metric_figure(curves, metric), key="fig_metric")

        with st.expander("Learning rate"):
            lr_fig = plots.lr_figure(curves)
            if lr_fig.data:
                st.plotly_chart(lr_fig, key="fig_lr")
            else:
                st.caption("Немає даних про lr.")

    with st.expander("Метрики за групами під час навчання (вибрана модель)"):
        group_history = get_group_history(current)
        if group_history is None or group_history.empty or "group_type" not in group_history:
            st.caption("group_metrics.csv відсутній.")
        else:
            types_g = [g for g in dl.GROUP_COLUMNS if g in group_history["group_type"].unique()]
            epochs = sorted(group_history["epoch"].unique())
            best_epoch = dl.best_info(current, get_history(current))["epoch"]
            default_idx = min(range(len(epochs)), key=lambda i: abs(epochs[i] - best_epoch)) if best_epoch is not None else len(epochs) - 1

            k = current.name
            g1, g2, g3, g4, g5 = st.columns([1, 1, 1, 1, 1])
            group_type = g1.selectbox("Група", types_g, key=f"gt_{k}")
            epoch = g2.selectbox("Епоха", epochs, index=default_idx, key=f"ge_{k}")
            group_metric = g3.selectbox("Метрика", plots.BAR_METRICS, index=plots.BAR_METRICS.index("f1_ai"), key=f"gm_{k}")
            min_n = g4.number_input("мін. n", min_value=1, value=20, step=10, key=f"gn_{k}")
            top_n = g5.slider("Найгірших груп", 5, 40, 15, key=f"gtop_{k}")

            frame = dl.group_frame(group_history, group_type, epoch)
            fig = plots.group_bars_figure([(current.label, plots.color_for(0), frame)], group_metric, min_n, top_n)
            if fig.data:
                st.plotly_chart(fig, key="fig_groups_train")
            else:
                st.caption("Немає груп із таким n.")


# Тестування
with tab_test:
    if not current.has_test:
        st.info(
            "Для цієї моделі ще немає тестового запуску. Запустіть тестування в ноутбуці "
            "(`test_*_run`) і синхронізуйте папку з логами в репозиторій."
        )
    else:
        details = get_test(current)
        metrics = details["metrics"]
        color0 = plots.color_for(0)

        st.markdown("#### Метрики на test")
        metric_cards(metrics)

        # моделі для порівняння, у яких є тест
        rows = [(current.label, color0, metrics)]
        no_test = []
        for i, run in enumerate(compare_runs, start=1):
            if run.has_test:
                rows.append((run.label, plots.color_for(i), get_test(run)["metrics"]))
            else:
                no_test.append(run.label)
        if no_test:
            st.caption("Без тестового запуску (не показано): " + ", ".join(no_test))

        left, right = st.columns(2)
        with left:
            st.markdown("**Порівняння метрик**" if len(rows) > 1 else "**Метрики**")
            st.plotly_chart(plots.metrics_bars_figure(rows), key="fig_test_bars")
        with right:
            st.markdown("**Val vs test (вибрана модель)**")
            if current.val_metrics:
                st.plotly_chart(plots.val_vs_test_figure(current.val_metrics, metrics, color0), key="fig_val_test")
            else:
                st.caption("Немає val-метрик у run_summary.json.")

        with st.expander("Таблиця метрик (разом із loss)"):
            st.dataframe(
                pd.DataFrame({label: m for label, _, m in rows}).T.reindex(columns=dl.METRIC_NAMES),
                column_config={c: st.column_config.NumberColumn(format="%.4f") for c in dl.METRIC_NAMES},
            )

        if current.test_level == "metrics":
            st.info("Для цього запуску збережено лише підсумкові метрики (немає папки test_results): "
                    "confusion matrix, групи й криві недоступні.")
        else:
            # confusion matrix
            st.markdown("#### Confusion matrix")
            normalize = st.checkbox("у відсотках від справжнього класу", value=True, key="cm_norm")
            cm_items = [(current.label, details["confusion"])]
            cm_items += [(r.label, get_test(r)["confusion"]) for r in compare_runs if r.test_level == "full"]
            cm_items = [(label, cm) for label, cm in cm_items if cm is not None]
            for start in range(0, len(cm_items), 3):
                chunk = cm_items[start:start + 3]
                for col, (label, cm) in zip(st.columns(len(chunk)), chunk):
                    col.markdown(f"**{label}**")
                    col.plotly_chart(plots.confusion_figure(cm, normalize), key=f"fig_cm_{start}_{label}")

            # групи
            st.markdown("#### Метрики за групами")
            if not details["groups"]:
                st.caption("Файлів metrics_by_*.csv немає.")
            else:
                g1, g2, g3, g4 = st.columns(4)
                group_type = g1.selectbox("Група", list(details["groups"]), key="test_gt")
                group_metric = g2.selectbox("Метрика", plots.BAR_METRICS,
                                            index=plots.BAR_METRICS.index("f1_ai"), key="test_gm")
                min_n = g3.number_input("мін. n", min_value=1, value=20, step=10, key="test_gn")
                top_n = g4.slider("Найгірших груп", 5, 40, 15, key="test_gtop")

                frames = [(current.label, color0, dl.group_frame(details["groups"][group_type], group_type))]
                for i, run in enumerate(compare_runs, start=1):
                    if run.test_level == "full":
                        other_groups = get_test(run)["groups"]
                        if group_type in other_groups:
                            frames.append((run.label, plots.color_for(i), dl.group_frame(other_groups[group_type], group_type)))
                fig = plots.group_bars_figure(frames, group_metric, min_n, top_n)
                if fig.data:
                    st.plotly_chart(fig, key="fig_groups_test")
                    st.caption("Групи відібрано за найгіршим значенням метрики у вибраної моделі. "
                               "На малих n метрики шумні; roc_auc не визначений, якщо в групі один клас.")
                else:
                    st.caption("Немає груп із таким n.")

            # криві
            st.markdown("#### ROC, PR і розподіл ймовірностей")
            if st.checkbox("Показати (читає test_predictions.csv)", value=True, key="show_curves"):
                items = []
                for i, run in enumerate([current] + compare_runs):
                    if run.test_level != "full":
                        continue
                    preds = get_predictions(get_test(run))
                    if preds is not None:
                        items.append((run.label, plots.color_for(i), preds))
                if not items:
                    st.caption("test_predictions.csv відсутній.")
                else:
                    r1, r2, r3 = st.columns(3)
                    r1.markdown("**ROC**")
                    r1.plotly_chart(plots.roc_figure(items), key="fig_roc")
                    r2.markdown("**Precision–Recall**")
                    r2.plotly_chart(plots.pr_figure(items), key="fig_pr")
                    r3.markdown("**P(AI) за справжнім класом (вибрана модель)**")
                    if items[0][0] == current.label:
                        r3.plotly_chart(plots.prob_hist_figure(items[0][2]), key="fig_hist")
