from __future__ import annotations
"""
Построение признаков и меток для почасовой/дневной моделей.

Содержит:
  * build_hourly_table: базовая почасовая таблица + бинарная метка;
  * add_time_lag_roll_feats: лаги/роллинги и гармоники по часам;
  * make_horizon_labels: сдвиг метки на H часов вперёд;
  * make_daily_future_labels: суточные метки (будет ли ≥1 аномальный час);
  * build_daily_table: агрегирование почасовых в дневные;
  * add_day_lag_roll_feats: лаги/роллинги/диффы на дневном уровне.
"""

from typing import Tuple

import numpy as np
import pandas as pd


# ---------- ПОЧАСОВЫЕ ----------

def build_hourly_table(
    hourly_df: pd.DataFrame,
    decisions_df: pd.DataFrame | None,
    rel_diff_limit: float,
    use_online_labels: bool,
) -> pd.DataFrame:
    """Собирает базовый почасовой DataFrame и бинарную метку.

    Требуемые колонки:
      * ts (datetime),
      * 'Потребление за период, м3 ХВС',
      * 'Потребление за период, м3 ГВС'.

    Метка:
      * если есть decisions_df и use_online_labels=True -> берём is_anomaly_final;
      * иначе суррогат: (|GVS-XVS| / max(GVS,XVS)) > rel_diff_limit.

    Args:
        hourly_df: Почасовая сырая таблица.
        decisions_df: Таблица решений онлайн-детектора (опционально).
        rel_diff_limit: Порог относительной разницы.
        use_online_labels: Использовать ли онлайн-метки.

    Returns:
        Почасовой DataFrame с базовыми фичами и колонкой `label`.
    """
    df = hourly_df.copy()
    if "ts" not in df.columns:
        raise KeyError("В hourly_df отсутствует колонка 'ts'. Проверьте processed_hourly.csv")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    def to_num(s: pd.Series | None) -> pd.Series:
        return pd.to_numeric(s, errors="coerce") if s is not None else pd.Series(np.nan, index=df.index)

    xvs = to_num(df.get("Потребление за период, м3 ХВС"))
    gvs = to_num(df.get("Потребление за период, м3 ГВС"))
    eps = 1e-12

    df["delta_m3"] = (xvs - gvs).astype(float)
    df["ratio"] = (gvs / np.maximum(xvs, eps)).replace([np.inf, -np.inf], np.nan)
    df["rel_diff"] = (gvs - xvs).abs() / np.maximum(gvs, xvs).clip(lower=eps)

    if use_online_labels and decisions_df is not None and not decisions_df.empty and "is_anomaly_final" in decisions_df.columns:
        dec = decisions_df[["ts", "is_anomaly_final"]].copy()
        dec["ts"] = pd.to_datetime(dec["ts"], errors="coerce")
        df = df.merge(dec, on="ts", how="left")
        df["label"] = df["is_anomaly_final"].astype("boolean").fillna(False).astype(bool)
    else:
        df["label"] = (df["rel_diff"] > rel_diff_limit)

    # календарные/сезонные фичи
    df["dow"] = df["ts"].dt.dayofweek
    df["hour"] = df["ts"].dt.hour
    df["how"] = df["dow"] * 24 + df["hour"]  # 0..167

    # если присутствуют погодные/температурные каналы — привести к числу
    for c in ["Погода", "T1 гвс, оС", "T2 гвс, оС"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def add_time_lag_roll_feats(
    df: pd.DataFrame,
    lags_hours: Tuple[int, ...],
    roll_windows_h: Tuple[int, ...],
) -> pd.DataFrame:
    """Добавляет почасовые лаги/роллинги и гармоники недели.

    Args:
        df: Почасовая базовая таблица (из build_hourly_table).
        lags_hours: Набор лагов в часах.
        roll_windows_h: Окна роллингов в часах.

    Returns:
        Расширенный DataFrame с новыми признаками.
    """
    out = df.copy().sort_values("ts").reset_index(drop=True)
    base_cols = ["delta_m3", "ratio", "rel_diff"]
    for c in ["itp_xvs_zero_anomaly", "odpu_gvs_zero_anomaly",
              "itp_xvs_flat_anomaly", "odpu_gvs_flat_anomaly",
              "itp_xvs_spike", "odpu_gvs_spike"]:
        if c in out.columns:
            base_cols.append(c)

    for col in base_cols:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        feats = {}
        for L in lags_hours:
            feats[f"{col}_lag{L}h"] = s.shift(L)
        for W in roll_windows_h:
            minp = max(2, W // 2)
            feats[f"{col}_roll{W}h_mean"] = s.rolling(W, min_periods=minp).mean()
            feats[f"{col}_roll{W}h_std"]  = s.rolling(W, min_periods=minp).std()
            feats[f"{col}_roll{W}h_min"]  = s.rolling(W, min_periods=minp).min()
            feats[f"{col}_roll{W}h_max"]  = s.rolling(W, min_periods=minp).max()
        out = pd.concat([out, pd.DataFrame(feats, index=out.index)], axis=1)

    out["sin_how"] = np.sin(2*np.pi*out["how"]/168.0)
    out["cos_how"] = np.cos(2*np.pi*out["how"]/168.0)
    return out


def make_horizon_labels(df: pd.DataFrame, horizons_hours: Tuple[int, ...]) -> dict[int, pd.Series]:
    """Создаёт бинарные метки: аномалия через h часов.

    Args:
        df: Почасовая таблица с колонкой `label`.
        horizons_hours: Горизонты в часах.

    Returns:
        Словарь {h: Series}.
    """
    labels: dict[int, pd.Series] = {}
    y = df["label"].astype(int)
    for h in horizons_hours:
        labels[h] = y.shift(-h).fillna(0).astype(int)
    return labels


def make_daily_future_labels(df: pd.DataFrame, day_horizons: Tuple[int, ...]) -> dict[int, pd.Series]:
    """Создаёт суточные метки: будет ли ≥1 аномальный час в день t+H.

    Args:
        df: Почасовая таблица с `ts` и `label`.
        day_horizons: Горизонты в днях.

    Returns:
        Словарь {H: Series с индексом по дням}.
    """
    tmp = df[["ts", "label"]].copy()
    tmp["day"] = tmp["ts"].dt.floor("D")
    daily = tmp.groupby("day")["label"].max().astype(int)
    labels: dict[int, pd.Series] = {}
    for H in day_horizons:
        labels[H] = daily.shift(-H).reindex(daily.index).fillna(0).astype(int)
    return labels


# ---------- ДНЕВНЫЕ ----------

def build_daily_table(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует почасовую таблицу в дневной уровень.

    Возвращает набор дневных агрегатов + устойчиво-вариативные признаки
    (day_idx, week_sin/cos), чтобы избегать проблем «все признаки константы».

    Args:
        hourly_df: Почасовая таблица (лучше — после build_hourly_table).

    Returns:
        Дневной DataFrame с колонкой `day`.
    """
    df = hourly_df.copy()
    df["day"] = df["ts"].dt.floor("D")
    agg = (
        df.groupby("day")
          .agg(
              itp_xvs_m3=("Потребление за период, м3 ХВС", "sum"),
              odpu_gvs_m3=("Потребление за период, м3 ГВС", "sum"),
              delta_m3_sum=("delta_m3", "sum"),
              delta_m3_mean=("delta_m3", "mean"),
              ratio_mean=("ratio", "mean"),
              rel_diff_mean=("rel_diff", "mean"),
              hours=("ts", "size"),
              bad_hours=("label", "sum"),
              dow=("day", lambda s: s.dt.dayofweek.iloc[0]),
          )
          .reset_index()
    )
    agg["bad_hours_share"] = (agg["bad_hours"] / agg["hours"]).fillna(0.0)

    agg = agg.sort_values("day").reset_index(drop=True)
    agg["day_idx"] = np.arange(len(agg))
    agg["dow_cat"] = agg["dow"].astype("int64")
    agg["week_sin"] = np.sin(2*np.pi*(agg["dow_cat"]/7.0))
    agg["week_cos"] = np.cos(2*np.pi*(agg["dow_cat"]/7.0))
    return agg


def add_day_lag_roll_feats(
    daily_df: pd.DataFrame,
    lags_days: Tuple[int, ...],
    roll_windows_d: Tuple[int, ...],
) -> pd.DataFrame:
    """Добавляет дневные лаги/роллинги и диффы.

    Args:
        daily_df: Дневной DataFrame (из build_daily_table).
        lags_days: Список лагов в днях.
        roll_windows_d: Окна роллингов в днях.

    Returns:
        Расширенный дневной DataFrame.
    """
    out = daily_df.copy().sort_values("day").reset_index(drop=True)
    base_cols = [
        "itp_xvs_m3", "odpu_gvs_m3", "delta_m3_sum", "delta_m3_mean",
        "ratio_mean", "rel_diff_mean", "bad_hours_share",
        "day_idx", "week_sin", "week_cos"
    ]
    for col in base_cols:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        feats = {}
        for L in lags_days:
            feats[f"{col}_lag{L}d"] = s.shift(L)
        for W in roll_windows_d:
            minp = max(2, W // 2)
            feats[f"{col}_roll{W}d_mean"] = s.rolling(W, min_periods=minp).mean()
            feats[f"{col}_roll{W}d_std"]  = s.rolling(W, min_periods=minp).std()
            feats[f"{col}_roll{W}d_min"]  = s.rolling(W, min_periods=minp).min()
            feats[f"{col}_roll{W}d_max"]  = s.rolling(W, min_periods=minp).max()
        feats[f"{col}_diff1"] = s.diff(1)
        out = pd.concat([out, pd.DataFrame(feats, index=out.index)], axis=1)
    return out
