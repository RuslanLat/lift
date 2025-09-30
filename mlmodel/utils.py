"""Набор утилит для пайплайна детекции аномалий.

Содержит функции для:
- нормализации времени и построения часовой сетки;
- приведения единиц измерения и расчёта прокси ОДПУ по рециркуляции;
- расчёта флагов качества каналов (нули, залипание, всплески);
- расчёта баланса ХВС vs ГВС и формирования алертов/суточного отчёта;
- сборки инцидентов по результатам онлайн-детектора;
- построения единственного графика 03_daily_bars.png.

Все пороги/коэффициенты вынесены в config.py.
"""

from __future__ import annotations

from typing import Dict, Optional

import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlmodel import config as cfg

# ==== Имена колонок ====
COL_DATE: str = "Дата"
COL_HOURIDX: str = "Время суток, ч"
COL_SUPPLY: str = "Подача, м3"
COL_RETURN: str = "Обратка, м3"
COL_ODPU_GVS: str = "Потребление за период, м3 ГВС"
COL_ITP_XVS: str = "Потребление за период, м3 ХВС"
COL_T1: str = "T1 гвс, оС"
COL_T2: str = "T2 гвс, оС"

UNITS: Dict[str, float] = cfg.UNITS
THRESHOLDS: Dict[str, float] = cfg.THRESHOLDS
PLOT: Dict[str, object] = cfg.PLOT


# ---------------------------------------------------------------------
# 1) ВРЕМЯ / ЧАСОВАЯ СЕТКА
# ---------------------------------------------------------------------
def normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализует дату/время и формирует непрерывную часовую сетку.

    Ожидается, что колонка ``"Время суток, ч"`` содержит интервалы вида
    ``"0-1"`` ... ``"23-24"`` (используются только правые границы).

    Алгоритм:
      1. Парсинг дат в ``Дата`` и часов из ``Время суток, ч``.
      2. Формирование метки начала часа ``ts`` (правый конец N - начало N-1).
      3. Достройка непрерывной часовой сетки между min(ts) и max(ts).

    Args:
        df: Входной DataFrame с исходными колонками.

    Returns:
        DataFrame с добавленными колонками:
        - ``ts`` (DatetimeIndex-совместимая метка часа),
        - обновлённые ``Дата`` и ``Время суток, ч`` (согласованные с сеткой),
        - служебная ``hour_idx`` (int, 1..24).

    Notes:
        При невозможности распознать дату/час соответствующие строки отбрасываются.
    """
    out = df.copy()
    out[COL_DATE] = pd.to_datetime(
        out[COL_DATE], format="%d.%m.%Y", errors="coerce"
    ).fillna(pd.to_datetime(out[COL_DATE], errors="coerce"))
    s = (
        out[COL_HOURIDX]
        .astype(str)
        .str.replace("\u2013", "-", regex=False)
        .str.replace("\u2014", "-", regex=False)
        .str.strip()
    )
    hour_right = pd.to_numeric(s.str.extract(r"-(\d{1,2})$")[0], errors="coerce")
    out["hour_idx"] = hour_right
    mask = out[COL_DATE].notna() & out["hour_idx"].between(1, 24)
    out = out.loc[mask].copy()
    out["ts"] = out[COL_DATE] + pd.to_timedelta(out["hour_idx"] - 1, unit="h")

    if not out.empty:
        full_index = pd.date_range(out["ts"].min(), out["ts"].max(), freq="h")
        out = out.set_index("ts").reindex(full_index).rename_axis("ts").reset_index()
        out[COL_DATE] = out["ts"].dt.normalize()
        out[COL_HOURIDX] = (out["ts"].dt.hour + 1).astype("int64")
        out["hour_idx"] = out[COL_HOURIDX]
    else:
        out["hour_idx"] = out.get("hour_idx", pd.Series(dtype="int64"))
        out["ts"] = pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))
    return out


# ---------------------------------------------------------------------
# 2) ЕДИНИЦЫ / ПРОКСИ ОДПУ ИЗ РЕЦИРКУЛЯЦИИ
# ---------------------------------------------------------------------
def harmonize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит числовые столбцы к общим единицам из конфигурации.

    Args:
        df: Входной DataFrame.

    Returns:
        DataFrame, где значения в столбцах расхода умножены на соответствующие
        коэффициенты из ``config.UNITS``:
        - ``Потребление за период, м3 ХВС`` - ``itp_xvs_factor``,
        - ``Потребление за период, м3 ГВС`` - ``odpu_gvs_factor``,
        - ``Подача, м3`` - ``supply_factor``,
        - ``Обратка, м3`` - ``return_factor``.
    """
    out = df.copy()
    for c, k in [
        (COL_ITP_XVS, "itp_xvs_factor"),
        (COL_ODPU_GVS, "odpu_gvs_factor"),
        (COL_SUPPLY, "supply_factor"),
        (COL_RETURN, "return_factor"),
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * UNITS[k]
    return out


def apply_recirculation(
    df: pd.DataFrame,
    *,
    enabled: bool = False,
    write_proxy_col: str = "odpu_proxy_net",
    use_proxy_if_missing: bool = True,
    override_odpu: bool = False,
) -> pd.DataFrame:
    """Рассчитывает прокси-ОДПУ по рециркуляции (Подача − Обратка, ≥0).

    Если ``enabled=True`` и есть колонки Подача/Обратка, вычисляет
    ``proxy = max(Подача − Обратка, 0)`` и:
      * при ``override_odpu=True`` заменяет столбец ОДПУ прокси;
      * иначе при ``use_proxy_if_missing=True`` подставляет прокси,
        только если столбец ОДПУ отсутствует;
      * в любом случае пишет прокси в ``write_proxy_col``.

    Args:
        df: Входной DataFrame.
        enabled: Включить расчёт прокси.
        write_proxy_col: Имя столбца, куда писать proxy.
        use_proxy_if_missing: Подставлять proxy, если ОДПУ нет в данных.
        override_odpu: Полностью заменить ОДПУ значениями proxy.

    Returns:
        DataFrame с добавленным столбцом прокси и, опционально, с обновлённым
        столбцом ОДПУ.

    Notes:
        Если одной из колонок Подача/Обратка нет — возвращается исходный df
        (прокси заполняется NaN).
    """
    if not enabled:
        return df
    out = df.copy()
    if not (COL_SUPPLY in out.columns and COL_RETURN in out.columns):
        out[write_proxy_col] = np.nan
        return out
    s_supply = pd.to_numeric(out[COL_SUPPLY], errors="coerce")
    s_return = pd.to_numeric(out[COL_RETURN], errors="coerce")
    proxy = (s_supply - s_return).clip(lower=0)
    out[write_proxy_col] = proxy
    if override_odpu:
        out[COL_ODPU_GVS] = proxy
    elif use_proxy_if_missing and (COL_ODPU_GVS not in out.columns):
        out[COL_ODPU_GVS] = proxy
    return out


# ---------------------------------------------------------------------
# 3) КАЧЕСТВО КАНАЛОВ
# ---------------------------------------------------------------------
def _run_length_flags(binary_series: pd.Series) -> pd.Series:
    """Вычисляет длину текущего непрерывного фрагмента единиц (run-length).

    Args:
        binary_series: Булева/0-1 серия (NaN трактуется как 0).

    Returns:
        Серия той же длины: для каждого индекса — длина текущего «забега»
        единиц, иначе 0.
    """
    b = binary_series.astype(int).fillna(0)
    return b * (b.groupby((b != b.shift()).cumsum()).cumcount() + 1)


def quality_flags(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """Строит флаги качества для канала расхода.

    Рассчитываются признаки:
      * ``{prefix}_zero_anomaly`` — минимум подряд часов нулей
        (``THRESHOLDS["zero_anomaly_hours"]``);
      * ``{prefix}_flat_anomaly`` — залипание значения не менее
        (``THRESHOLDS["flatline_hours"]``) часов;
      * ``{prefix}_spike`` — всплеск по критерию IQR на окне 24 ч.

    Args:
        df: Входной DataFrame.
        col: Имя числовой колонки канала (например, ХВС/ГВС).
        prefix: Префикс для имён флагов (например, ``"itp_xvs"``).

    Returns:
        DataFrame с добавленными бинарными флагами качества канала.

    Notes:
        Если столбца ``col`` нет — возвращается входной df без изменений.
    """
    out = df.copy()
    if col not in out.columns:
        return out
    s = pd.to_numeric(out[col], errors="coerce")

    # Нули
    zero = (s.fillna(0) == 0).astype(int)
    out[f"{prefix}_zero"] = zero
    out[f"{prefix}_zero_run"] = _run_length_flags(zero)
    out[f"{prefix}_zero_anomaly"] = (
        out[f"{prefix}_zero_run"] >= THRESHOLDS["zero_anomaly_hours"]
    )

    # Залипание
    flat = (s == s.shift()).astype(int)
    out[f"{prefix}_flat"] = flat
    out[f"{prefix}_flat_run"] = _run_length_flags(flat)
    out[f"{prefix}_flat_anomaly"] = (
        out[f"{prefix}_flat_run"] >= THRESHOLDS["flatline_hours"]
    )

    # Всплеск по IQR
    med = s.rolling(24, min_periods=8, center=True).median()
    q1 = s.rolling(24, min_periods=8, center=True).quantile(0.25)
    q3 = s.rolling(24, min_periods=8, center=True).quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    out[f"{prefix}_spike"] = (s > med + THRESHOLDS["spike_iqr_k"] * iqr) | (
        s < med - THRESHOLDS["spike_iqr_k"] * iqr
    )
    return out


# ---------------------------------------------------------------------
# 4) БАЛАНС И ОТЧЁТЫ
# ---------------------------------------------------------------------
def compute_balance_and_flags(df: pd.DataFrame, rel_diff_limit: float) -> pd.DataFrame:
    """Считает баланс ХВС vs ГВС и формирует базовые флаги/подсказки.

    Для каждого часа вычисляет:
      * ``diff_m3 = GVS - XVS``;
      * ``rel_diff = |GVS - XVS| / max(GVS, XVS)``;
      * ``flag_over10`` — rel_diff выше порога ``THRESHOLDS["rel_diff_limit"]``.

    Также вызывает :func:`quality_flags` для ХВС и ГВС (если колонки присутствуют)
    и записывает текстовую подсказку ``reason_hint`` на базе простых эвристик.

    Args:
        df: Входной DataFrame после нормализации времени/единиц.

    Returns:
        DataFrame с добавленными колонками ``diff_m3``, ``rel_diff``,
        ``flag_over10``, наборами флагов качества по каналам и
        человеко-ориентированной подсказкой ``reason_hint``.
    """
    print(rel_diff_limit)
    out = df.copy()
    eps = 1e-12
    xvs = pd.to_numeric(out.get(COL_ITP_XVS, np.nan), errors="coerce").clip(lower=0)
    gvs = pd.to_numeric(out.get(COL_ODPU_GVS, np.nan), errors="coerce").clip(lower=0)
    denom = np.maximum(gvs, xvs).clip(lower=eps)
    out["rel_diff"] = (gvs - xvs).abs() / denom
    out["diff_m3"] = gvs - xvs
    out["flag_over10"] = out["rel_diff"] > rel_diff_limit

    if COL_ITP_XVS in out.columns:
        out = quality_flags(out, COL_ITP_XVS, "itp_xvs")
    if COL_ODPU_GVS in out.columns:
        out = quality_flags(out, COL_ODPU_GVS, "odpu_gvs")

    # Человеческая подсказка (не влияет на алгоритм)
    reasons = []
    for _, r in out.iterrows():
        reason = []
        if r.get("flag_over10", False):
            if r.get("itp_xvs_zero_anomaly"):
                reason.append("ИТП-ХВС нули")
            if r.get("odpu_gvs_zero_anomaly"):
                reason.append("ОДПУ-ГВС нули")
            if r.get("itp_xvs_spike"):
                reason.append("ИТП-ХВС скачок")
            if r.get("odpu_gvs_spike"):
                reason.append("ОДПУ-ГВС скачок")
            if r.get("itp_xvs_flat_anomaly"):
                reason.append("ИТП-ХВС застой")
            if r.get("odpu_gvs_flat_anomaly"):
                reason.append("ОДПУ-ГВС застой")
            if (
                COL_T1 in out.columns
                and COL_T2 in out.columns
                and pd.notna(r.get(COL_T1))
                and pd.notna(r.get(COL_T2))
                and (r.get(COL_T1) - r.get(COL_T2) < 5)
            ):
                reason.append("малый дельта T — возможна рециркуляция/подпитка")
        reasons.append("; ".join(reason))
    out["reason_hint"] = reasons
    return out


def build_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает таблицу «подозрительных часов».

    Включаются часы, где:
      * ``flag_over10`` истинна, и/или
      * любой из ``*_anomaly`` истинен.

    Args:
        df: DataFrame после :func:`compute_balance_and_flags`.

    Returns:
        Отсортированный по ``ts`` DataFrame с ключевыми колонками для разбора.
    """
    cols = [
        COL_DATE,
        COL_HOURIDX,
        "ts",
        COL_ITP_XVS,
        COL_ODPU_GVS,
        "diff_m3",
        "rel_diff",
        "itp_xvs_zero_anomaly",
        "odpu_gvs_zero_anomaly",
        "itp_xvs_flat_anomaly",
        "odpu_gvs_flat_anomaly",
        "itp_xvs_spike",
        "odpu_gvs_spike",
        "reason_hint",
    ]
    cols = [c for c in cols if c in df.columns]
    alerts = df.loc[
        df["flag_over10"] | df.filter(like="_anomaly").any(axis=1), cols
    ].copy()
    return alerts.sort_values(["ts"])


def daily_report(df: pd.DataFrame) -> pd.DataFrame:
    """Считает суточные суммы и агрегированные метрики качества.

    Args:
        df: Почасовой DataFrame после :func:`compute_balance_and_flags`.

    Returns:
        DataFrame на уровне суток со столбцами:
        - ``itp_xvs_m3``, ``odpu_gvs_m3`` — суточные суммы;
        - ``hours_total`` — всего часов в дне;
        - ``hours_over10`` — часов с ``flag_over10``;
        - ``daily_rel_diff`` и ``daily_rel_diff_%`` — относительное расхождение;
        - ``hours_over10_%`` — доля «плохих» часов.
    """
    grp = (
        df.groupby(df[COL_DATE])
        .agg(
            itp_xvs_m3=(COL_ITP_XVS, "sum"),
            odpu_gvs_m3=(COL_ODPU_GVS, "sum"),
            hours_total=("rel_diff", "size"),
            hours_over10=("flag_over10", "sum"),
        )
        .reset_index()
    )
    denom = np.maximum(grp["odpu_gvs_m3"], grp["itp_xvs_m3"]).clip(lower=1e-12)
    grp["daily_rel_diff"] = (grp["odpu_gvs_m3"] - grp["itp_xvs_m3"]).abs() / denom
    grp["daily_rel_diff_%"] = (grp["daily_rel_diff"] * 100).round(1)
    grp["hours_over10_%"] = (grp["hours_over10"] / grp["hours_total"] * 100).round(1)
    return grp


# ---------------------------------------------------------------------
# 5) ИНЦИДЕНТЫ И ЕДИНСТВЕННЫЙ ГРАФИК
# ---------------------------------------------------------------------
def build_incidents(
    decisions_df: pd.DataFrame,
    min_len_hours: Optional[int] = None,
    ru_cause_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Склеивает подряд идущие часы с одинаковой причиной и is_anomaly=True.

    Разбивка на инциденты выполняется при изменении флага аномалии/причины или
    разрыве по времени более чем на 1 час.

    Args:
        decisions_df: Таблица решений онлайн-детектора (почасовая).
        min_len_hours: Минимальная длительность инцидента в часах.
            По умолчанию берётся из ``config.INCIDENTS["min_len_hours"]``.
        ru_cause_map: Отображение cause_en - cause_ru (по умолчанию ``config.RU_CAUSE``).

    Returns:
        DataFrame инцидентов со столбцами:
        ``start``, ``end``, ``duration_h``, ``cause_en``, ``cause_ru``,
        ``confidence`` (медиана p), ``delta_m3_sum``, ``delta_m3_mean``,
        ``worst_z``, ``hours``.
    """
    if decisions_df is None or decisions_df.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "duration_h",
                "cause_en",
                "cause_ru",
                "confidence",
                "delta_m3_sum",
                "delta_m3_mean",
                "worst_z",
                "hours",
            ]
        )

    # Единый параметр длительности: сначала новый ключ, затем старый (бэкомпат)
    if min_len_hours is None:
        min_len_hours = cfg.INCIDENTS.get(
            "min_duration_h",
            cfg.INCIDENTS.get("min_len_hours", 1),
        )
    if ru_cause_map is None:
        ru_cause_map = cfg.RU_CAUSE

    d = decisions_df.copy().sort_values("ts")
    d["is_anomaly"] = d["is_anomaly"].astype(bool)
    d["cause_en"] = d["top_cause"].astype(str)

    d["grp"] = (
        (d["is_anomaly"] != d["is_anomaly"].shift())
        | (d["cause_en"] != d["cause_en"].shift())
        | ((d["ts"] - d["ts"].shift()) > pd.Timedelta(hours=1))
    ).cumsum()

    rows = []
    for _, g in d.groupby("grp"):
        if not g["is_anomaly"].iloc[0]:
            continue
        hours = int(len(g))
        if hours < min_len_hours:
            continue

        cause_en = str(g["cause_en"].iloc[0])
        cause_ru = ru_cause_map.get(cause_en, cause_en)

        conf = float(g.get("top_cause_prob", pd.Series([np.nan])).median())
        delta_sum = float(g.get("delta_m3", pd.Series([0.0])).sum())
        delta_mean = float(g.get("delta_m3", pd.Series([np.nan])).mean())
        worst_z = float(np.nanmax(np.abs(g.get("z_delta", pd.Series([np.nan])))))

        rows.append(
            {
                "start": g["ts"].iloc[0],
                "end": g["ts"].iloc[-1],
                "duration_h": hours,
                "cause_en": cause_en,
                "cause_ru": cause_ru,
                "confidence": round(conf, 3) if not math.isnan(conf) else None,
                "delta_m3_sum": round(delta_sum, 4),
                "delta_m3_mean": round(delta_mean, 4)
                if not math.isnan(delta_mean)
                else None,
                "worst_z": round(worst_z, 2) if not math.isnan(worst_z) else None,
                "hours": hours,
            }
        )

    return pd.DataFrame(rows).sort_values(["start", "cause_en"])


def plot_daily_bars(daily: pd.DataFrame) -> None:
    """Строит и сохраняет график 03_daily_bars.png.

    На графике сравниваются суточные суммы ИТП-ХВС и ОДПУ-ГВС, а над столбцами
    подписывается процентная величина суточного расхождения.

    Args:
        daily: Суточная сводка из :func:`daily_report`.
        path_png: Путь к PNG-файлу для сохранения.

    Returns:
        None. Файл графика сохраняется на диск.

    Notes:
        Параметры размера и DPI берутся из ``config.PLOT``.
    """
    if daily is None or daily.empty:
        plt.figure(figsize=PLOT["no_data_figsize"])
        plt.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        plt.axis("off")
        # plt.savefig(path_png, dpi=PLOT["dpi"])
        plt.close()
        return

    x = daily[COL_DATE].astype(str).values
    itp = daily["itp_xvs_m3"].values
    odpu = daily["odpu_gvs_m3"].values
    diff_pct = (daily["daily_rel_diff"] * 100).round(1).values

    idx = np.arange(len(x))
    w = 0.4

    fig, ax = plt.subplots(figsize=PLOT["daily_bars_figsize"])
    ax.bar(idx - w / 2, itp, width=w, label="ИТП-ХВС, м³")
    ax.bar(idx + w / 2, odpu, width=w, label="ОДПУ-ГВС, м³")

    for i, p in enumerate(diff_pct):
        ymax = max(itp[i], odpu[i])
        ax.text(i, (ymax * 1.02 if ymax > 0 else 0.1), f"{p}%", ha="center", fontsize=9)

    ax.set_xticks(idx)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("м³")
    ax.set_title("Суточные суммы и относительное расхождение")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    # ax.figure.savefig(path_png, dpi=PLOT["dpi"])
    # plt.close(fig)
