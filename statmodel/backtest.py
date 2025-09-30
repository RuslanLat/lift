from __future__ import annotations

"""
Backtest для почасовой и дневной моделей с безутечным выравниванием.

Содержит:
  * ts_expanding_splits: экспандинговые разбиения по времени;
  * backtest_hourly_fixed_h: backtest для одного часового горизонта;
  * backtest_daily_fixed_d: backtest для одного дневного горизонта;
  * backtest_hourly_multi: объединённый backtest для H=1..6 в один CSV;
  * backtest_daily_multi: объединённый backtest для D=1..3 в один CSV.
"""

import os
from typing import Dict, Tuple, Iterator, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from statmodel.models import build_model


def _only_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет числовые/булевы признаки (булевы -> int8)."""
    X = df.select_dtypes(include=["number", "bool"]).copy()
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")
    return X


def ts_expanding_splits(
    df: pd.DataFrame, n_splits: int, min_train_days: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Экспандинговые временные разбиения.

    Args:
        df: Таблица, содержащая колонку 'ts' (datetime).
        n_splits: Количество фолдов (как в TimeSeriesSplit).
        min_train_days: Минимальная длительность трен-сплита в днях.

    Yields:
        Кортежи массивов индексов (train_idx, test_idx).
    """
    if "ts" not in df.columns:
        raise KeyError("ts_expanding_splits: ожидаю колонку 'ts' в датафрейме")
    tss = TimeSeriesSplit(n_splits=n_splits)
    idx = np.arange(len(df))
    for tr, te in tss.split(idx):
        ts_tr = pd.to_datetime(df["ts"].iloc[tr], errors="coerce")
        span_days = (ts_tr.max() - ts_tr.min()).days + 1
        if span_days < min_train_days:
            continue
        yield tr, te


def backtest_hourly_fixed_h(
    data_with_ts: pd.DataFrame,
    y_series: pd.Series,
    model_name: str,
    horizon_h: int,
    n_splits: int,
    min_train_days: int,
    class_weight: Optional[dict],
) -> pd.DataFrame:
    """Backtest почасовой модели для одного горизонта (H часов).

    Без утечек: признаки X.shift(horizon_h).

    Args:
        data_with_ts: Таблица с 'ts' и числовыми фичами.
        y_series: Бинарная метка (аномалия через H часов).
        model_name: Имя модели для build_model.
        horizon_h: Горизонт в часах.
        n_splits: Количество временных фолдов.
        min_train_days: Минимальная длительность трейна (в днях).
        class_weight: Вес классов для обучения.

    Returns:
        Таблица метрик по фолдам.
    """
    data = data_with_ts.copy().reset_index(drop=True)
    data["ts"] = pd.to_datetime(data["ts"], errors="coerce")
    data = data.sort_values("ts").reset_index(drop=True)

    raw_feats = data.drop(columns=["ts"], errors="ignore")
    Xnum = _only_numeric_features(raw_feats)
    Xh = Xnum.shift(horizon_h)
    df_cv = pd.concat([data[["ts"]], y_series.rename("y"), Xh], axis=1).dropna(
        subset=["y"]
    )

    rows: List[dict] = []
    for i, (tr, te) in enumerate(ts_expanding_splits(df_cv, n_splits, min_train_days)):
        tr_df, te_df = df_cv.iloc[tr].copy(), df_cv.iloc[te].copy()
        Xtr = (
            tr_df.drop(columns=["ts", "y"])
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )
        Xte = (
            te_df.drop(columns=["ts", "y"])
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        var_cols = [c for c in Xtr.columns if Xtr[c].nunique(dropna=False) > 1]
        if not var_cols:
            continue
        Xtr, Xte = Xtr[var_cols], Xte[var_cols]

        ytr, yte = tr_df["y"].astype(int), te_df["y"].astype(int)
        if ytr.nunique() < 2 or len(Xtr) == 0 or len(Xte) == 0:
            continue

        mdl = build_model(model_name)
        mdl.fit(Xtr, ytr, class_weight=class_weight)
        proba = mdl.predict_proba(Xte)
        m = mdl.metrics(yte, proba, thr=0.5)
        rows.append(
            {
                "horizon_h": horizon_h,
                "fold": i,
                **m,
                "n_train": len(Xtr),
                "n_test": len(Xte),
                "n_pos_te": int(yte.sum()),
                "train_start": tr_df["ts"].min(),
                "train_end": tr_df["ts"].max(),
                "test_start": te_df["ts"].min(),
                "test_end": te_df["ts"].max(),
            }
        )
    return pd.DataFrame(rows)


def backtest_daily_fixed_d(
    daily_df: pd.DataFrame,
    y_series: pd.Series,
    model_name: str,
    horizon_d: int,
    n_splits: int,
    min_train_days: int,
    class_weight: Optional[dict],
    cat_feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Backtest дневной модели для одного горизонта (D суток).

    Без утечек: признаки X.shift(horizon_d) по дневному индексу.

    Args:
        daily_df: Дневная таблица с колонкой 'day' и фичами.
        y_series: Бинарная метка (аномалия в день t+D).
        model_name: Имя модели для build_model.
        horizon_d: Горизонт в днях.
        n_splits: Количество временных фолдов.
        min_train_days: Минимальная длительность трейна (в днях).
        class_weight: Вес классов.
        cat_feature_names: Имена категориальных признаков (если есть).

    Returns:
        Таблица метрик по фолдам.
    """
    df = daily_df.copy().sort_values("day").reset_index(drop=True)
    X = df.drop(columns=["day"])
    if cat_feature_names:
        to_add = [
            c for c in cat_feature_names if c not in X.columns and c in df.columns
        ]
        if to_add:
            X = X.join(df[to_add])
    X = _only_numeric_features(X)
    Xd = X.shift(horizon_d)

    dfx = pd.concat([df[["day"]], y_series.rename("y"), Xd], axis=1).dropna(
        subset=["y"]
    )
    dfx = dfx.rename(columns={"day": "ts"}).reset_index(drop=True)

    rows: List[dict] = []
    for i, (tr, te) in enumerate(ts_expanding_splits(dfx, n_splits, min_train_days)):
        tr_df, te_df = dfx.iloc[tr].copy(), dfx.iloc[te].copy()
        Xtr = (
            tr_df.drop(columns=["ts", "y"])
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )
        Xte = (
            te_df.drop(columns=["ts", "y"])
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        var_cols = [c for c in Xtr.columns if Xtr[c].nunique(dropna=False) > 1]
        if not var_cols:
            continue
        Xtr, Xte = Xtr[var_cols], Xte[var_cols]

        ytr, yte = tr_df["y"].astype(int), te_df["y"].astype(int)
        if ytr.nunique() < 2:
            continue

        mdl = build_model(model_name)
        mdl.fit(Xtr, ytr, class_weight=class_weight)
        proba = mdl.predict_proba(Xte)
        m = mdl.metrics(yte, proba, thr=0.5)
        rows.append(
            {
                "horizon_d": horizon_d,
                "fold": i,
                **m,
                "n_train": len(Xtr),
                "n_test": len(Xte),
                "n_pos_te": int(yte.sum()),
                "train_start": tr_df["ts"].min(),
                "train_end": tr_df["ts"].max(),
                "test_start": te_df["ts"].min(),
                "test_end": te_df["ts"].max(),
            }
        )
    return pd.DataFrame(rows)


def backtest_hourly_multi(
    data_with_ts: pd.DataFrame,
    y_by_h: Dict[int, pd.Series],
    model_name: str,
    horizons: List[int],
    n_splits: int,
    min_train_days: int,
    class_weight: Optional[dict],
) -> pd.DataFrame:
    """Единый backtest по нескольким часовым горизонтам.

    Returns:
        Конкатенированная таблица с колонкой `horizon_h`.
    """
    rows = []
    for h in horizons:
        bt = backtest_hourly_fixed_h(
            data_with_ts=data_with_ts,
            y_series=y_by_h[h],
            model_name=model_name,
            horizon_h=h,
            n_splits=n_splits,
            min_train_days=min_train_days,
            class_weight=class_weight,
        )
        if not bt.empty:
            rows.append(bt)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame([])


def backtest_daily_multi(
    daily_df: pd.DataFrame,
    y_by_d: Dict[int, pd.Series],
    model_name: str,
    horizons: List[int],
    n_splits: int,
    min_train_days: int,
    class_weight: Optional[dict],
    cat_feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Единый backtest по нескольким дневным горизонтам.

    Returns:
        Конкатенированная таблица с колонкой `horizon_d`.
    """
    rows = []
    for d in horizons:
        bt = backtest_daily_fixed_d(
            daily_df=daily_df,
            y_series=y_by_d[d],
            model_name=model_name,
            horizon_d=d,
            n_splits=n_splits,
            min_train_days=min_train_days,
            class_weight=class_weight,
            cat_feature_names=cat_feature_names,
        )
        if not bt.empty:
            rows.append(bt)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame([])
