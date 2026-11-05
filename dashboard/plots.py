# не залежить від Streamlit, ф-ї повертають графіки plotly (go.Figure)

"""
Позначення:
  * колір = модель (однаковий на всіх графіках), перша модель у списку — вибрана в бічній панелі;
  * суцільна лінія = val, пунктир = train;
  * (зірочка) = найкраща епоха; пунктир-крапки на всю ширину = baseline з одним значенням (LogReg).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

# Положення легенди на графіках навчання (Loss, метрика валідації, lr):
#   "bottom" - під графіком (на всю ширину, йде в кілька рядків)
#   "right" - праворуч (графік вужчий, але вищий)
LEGEND_POSITION = "bottom"

PALETTE = qualitative.D3          # 10 контрастних кольорів
BAR_METRICS = ["accuracy", "precision_ai", "recall_ai", "f1_ai", "macro_f1", "roc_auc"]


def color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


@dataclass
class Curve:
    label: str
    color: str
    history: pd.DataFrame
    best_epoch: float | None = None

    @property
    def is_baseline(self) -> bool:
        # запуск з одним записом (LogReg) = горизонтальна лінія
        return len(self.history) == 1


# навчання
def _x_extent(curves: list[Curve]) -> tuple[float, float]:
    multi = [c.history["epoch"] for c in curves if not c.is_baseline]
    if not multi:
        return 0.0, 1.0
    lo, hi = min(s.min() for s in multi), max(s.max() for s in multi)
    return (float(lo), float(hi)) if hi > lo else (float(lo), float(lo) + 1.0)


def _value_at(history: pd.DataFrame, column: str, epoch: float | None):
    if epoch is None or column not in history:
        return None
    idx = (history["epoch"] - epoch).abs().idxmin()
    value = history.loc[idx, column]
    return None if pd.isna(value) else float(value)


def _add_curve(fig, curve: Curve, column: str, x_range, *, train: bool = False, mark_best: bool = True):
    h = curve.history
    if column not in h or h[column].dropna().empty:
        return
    suffix = " · train" if train else ""
    name = f"{curve.label}{suffix}"
    hover = "%{y:.4f}<extra>" + name + "</extra>"

    if curve.is_baseline:
        y = float(h[column].iloc[0])
        fig.add_trace(go.Scatter(
            x=list(x_range), y=[y, y], mode="lines", name=name + " (baseline)",
            line=dict(color=curve.color, width=2, dash="dashdot" if train else "dot"),
            legendgroup=curve.label, hovertemplate=hover,
        ))
        return

    fig.add_trace(go.Scatter(
        x=h["epoch"], y=h[column], mode="lines+markers", name=name,
        line=dict(color=curve.color, width=2, dash="dash" if train else "solid"),
        marker=dict(size=5), legendgroup=curve.label, hovertemplate=hover,
    ))
    if mark_best and not train:
        y_best = _value_at(h, column, curve.best_epoch)
        if y_best is not None:
            fig.add_trace(go.Scatter(
                x=[curve.best_epoch], y=[y_best], mode="markers", showlegend=False,
                marker=dict(symbol="star", size=14, color=curve.color, line=dict(width=1, color="black")),
                legendgroup=curve.label, hovertemplate=f"найкраща епоха {curve.best_epoch}<extra>{curve.label}</extra>",
            ))


def _legend() -> dict:
    if LEGEND_POSITION == "right":
        return dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01)
    return dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0)   # "bottom"


def _style(fig, y_title: str, height: int = 440, log_y: bool = False):
    fig.update_layout(
        height=height, hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="епоха", yaxis_title=y_title, yaxis_type="log" if log_y else "linear",
        legend=_legend(),
    )
    return fig


def loss_figure(curves: list[Curve], show_train: bool = True, log_y: bool = False) -> go.Figure:
    fig = go.Figure()
    x_range = _x_extent(curves)
    for c in curves:
        _add_curve(fig, c, "val_loss", x_range)
        if show_train:
            _add_curve(fig, c, "train_loss", x_range, train=True)
    return _style(fig, "loss", log_y=log_y)


def metric_figure(curves: list[Curve], metric: str) -> go.Figure:
    fig = go.Figure()
    x_range = _x_extent(curves)
    for c in curves:
        _add_curve(fig, c, f"val_{metric}", x_range)
    return _style(fig, f"val_{metric}")


def lr_figure(curves: list[Curve]) -> go.Figure:
    fig = go.Figure()
    for c in curves:
        h = c.history
        if "lr" in h and h["lr"].notna().any() and not c.is_baseline:
            fig.add_trace(go.Scatter(x=h["epoch"], y=h["lr"], mode="lines", name=c.label,
                                     line=dict(color=c.color, width=2)))
    return _style(fig, "learning rate", height=260)


# групи
def group_bars_figure(frames: list[tuple[str, str, pd.DataFrame]], metric: str,
                      min_n: int = 1, top_n: int = 15) -> go.Figure:
    """Горизонтальні стовпчики метрики за групами. frames[0] — вибрана модель: за нею відбираються
    top_n найгірших груп (n ≥ min_n); інші моделі малюються для тих самих груп."""
    fig = go.Figure()
    base = frames[0][2]
    if base.empty or metric not in base:
        return fig
    base = base[base["n"] >= min_n]
    order = base.sort_values(metric, na_position="first").head(top_n)["group"].tolist()
    if not order:
        return fig

    for label, color, df in frames:
        d = df.drop_duplicates("group").set_index("group").reindex(order)
        fig.add_trace(go.Bar(
            y=order, x=d[metric] if metric in d else [None] * len(order), orientation="h",
            name=label, marker_color=color, customdata=d["n"] if "n" in d else None,
            hovertemplate="%{y}<br>" + metric + " = %{x:.3f}<br>n = %{customdata}<extra>" + label + "</extra>",
        ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=metric)
    fig.update_layout(barmode="group", height=max(320, 90 + 24 * len(order) * len(frames)),
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0))
    return fig


# тест
def metrics_bars_figure(rows: list[tuple[str, str, dict]]) -> go.Figure:
    """rows: [(label, color, {metric: value})]; стовпчики згруповано за метриками, loss тут не показуємо."""
    fig = go.Figure()
    for label, color, metrics in rows:
        fig.add_trace(go.Bar(
            x=BAR_METRICS, y=[metrics.get(m) for m in BAR_METRICS], name=label, marker_color=color,
            hovertemplate="%{x} = %{y:.4f}<extra>" + label + "</extra>",
        ))
    fig.update_layout(barmode="group", height=360, yaxis=dict(range=[0, 1.02], title="значення"),
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
    return fig


def val_vs_test_figure(val_metrics: dict, test_metrics: dict, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=BAR_METRICS, y=[val_metrics.get(m) for m in BAR_METRICS], name="val",
                         marker_color=color, opacity=0.45,
                         hovertemplate="%{x} = %{y:.4f}<extra>val</extra>"))
    fig.add_trace(go.Bar(x=BAR_METRICS, y=[test_metrics.get(m) for m in BAR_METRICS], name="test",
                         marker_color=color,
                         hovertemplate="%{x} = %{y:.4f}<extra>test</extra>"))
    fig.update_layout(barmode="group", height=360, yaxis=dict(range=[0, 1.02]),
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
    return fig


def confusion_figure(cm: pd.DataFrame, normalize: bool = False) -> go.Figure:
    z = cm.to_numpy(dtype=float)
    if normalize:
        z = z / np.clip(z.sum(axis=1, keepdims=True), 1, None)       # частка від справжнього класу
        text = [[f"{v:.1%}" for v in row] for row in z]
    else:
        text = [[f"{int(v):,}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(cm.columns), y=list(cm.index), text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=False, xgap=2, ygap=2,
        hovertemplate="%{y} → %{x}: %{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def roc_figure(items: list[tuple[str, str, pd.DataFrame]]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                             line=dict(color="gray", dash="dot"), hoverinfo="skip"))
    for label, color, df in items:
        if df["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{label} (AUC={auc(fpr, tpr):.3f})",
                                 line=dict(color=color, width=2)))
    fig.update_layout(height=360, xaxis_title="FPR", yaxis_title="TPR", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
    return fig


def pr_figure(items: list[tuple[str, str, pd.DataFrame]]) -> go.Figure:
    fig = go.Figure()
    for label, color, df in items:
        if df["y_true"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(df["y_true"], df["y_prob"])
        ap = average_precision_score(df["y_true"], df["y_prob"])
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{label} (AP={ap:.3f})",
                                 line=dict(color=color, width=2)))
    fig.update_layout(height=360, xaxis_title="recall (AI)", yaxis_title="precision (AI)",
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
    return fig


def prob_hist_figure(df: pd.DataFrame) -> go.Figure:
    """Розподіл y_prob окремо для справжніх людських і AI-текстів (для вибраної моделі)"""
    fig = go.Figure()
    for value, name, color in ((0, "людина (0)", "#1f77b4"), (1, "AI (1)", "#d62728")):
        fig.add_trace(go.Histogram(x=df.loc[df["y_true"] == value, "y_prob"], name=name, marker_color=color,
                                   opacity=0.6, xbins=dict(start=0, end=1, size=0.05), histnorm="probability"))
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
    fig.update_layout(barmode="overlay", height=360, xaxis_title="P(AI)", yaxis_title="частка",
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.2, x=0))
    return fig
