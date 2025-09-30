from __future__ import annotations

"""
Онлайн-предсказания по обученным мультигоризонтным моделям.

Генерирует:
  * почасовые прогнозы на H=1..6 часов вперёд от последнего известного часа;
  * дневные прогнозы на D=1..3 суток вперёд от последнего известного дня.

Сохраняет:
  - forecast_output/pred_hourly.csv (h, ts_for, p_anomaly, is_anomaly)
  - forecast_output/pred_daily.csv  (d, day_for, p_anomaly, is_anomaly)
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from statmodel.config import Config
from statmodel.features import (
    build_hourly_table,
    add_time_lag_roll_feats,
    build_daily_table,
    add_day_lag_roll_feats,
)


def _only_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только числовые/булевы признаки (булевы -> int8)."""
    out = df.select_dtypes(include=["number", "bool"]).copy()
    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        out[bool_cols] = out[bool_cols].astype("int8")
    return out


def _prepare_hourly_X(base: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Строит почасовые фичи для инференса (без утечек)."""
    feats = add_time_lag_roll_feats(base, cfg.lags_hours, cfg.roll_windows_h)
    drop_cols = ["label", "is_anomaly_final"]
    return _only_numeric_features(
        feats[[c for c in feats.columns if c not in drop_cols]]
    )


def _prepare_daily_X(base: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Строит дневные фичи для инференса (без утечек)."""
    daily = build_daily_table(base)
    daily_feats = add_day_lag_roll_feats(daily, cfg.lags_days, cfg.roll_windows_d)
    Xd = _only_numeric_features(daily_feats.drop(columns=["day"]))
    return daily_feats[["day"]].join(Xd)


def _threshold(p: float, thr: float = 0.5) -> int:
    """Пороговая интерпретация вероятности аномалии."""
    return int(float(p) >= float(thr))


def predict_all(cfg: Config = Config()) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Делает мультигоризонтные предсказания (часы 1..6, дни 1..3).

    Args:
        cfg: Конфигурация (пути и фиче-параметры).

    Returns:
        (df_hourly, df_daily) — таблицы с прогнозами.
    """
    bundle_path = os.path.join(cfg.outdir, "models.pkl")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Не найден {bundle_path}. Сначала запустите обучение.")

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    # Загрузка исходных данных
    hourly = pd.read_csv(cfg.hourly_path)
    hourly["ts"] = pd.to_datetime(hourly["ts"], errors="coerce")
    dec = None
    if os.path.exists(cfg.online_decisions_path):
        dec = pd.read_csv(cfg.online_decisions_path)

    # База (почасовая)
    base = build_hourly_table(hourly, dec, cfg.rel_diff_limit, cfg.use_online_labels)

    # --- ЧАСЫ ---
    Xh_all = _prepare_hourly_X(base, cfg)
    last_ts = pd.to_datetime(base["ts"].max())

    preds_h: List[dict] = []
    if "hourly" in bundle and bundle["hourly"]["models"]:
        for h, pack in bundle["hourly"]["models"].items():
            mdl = pack["model"]
            cols = pack["features"]
            if not cols:
                continue
            # доступные в момент t признаки для предсказания y_{t+h}
            row = Xh_all.tail(1)[cols]
            if row.empty:
                continue
            p = float(mdl.predict_proba(row)[0])
            preds_h.append(
                {
                    "h": int(h),
                    "ts_for": last_ts + pd.Timedelta(hours=int(h)),
                    "p_anomaly": round(p, 4),
                    "is_anomaly": int(_threshold(p, 0.5)),
                }
            )
    df_hourly = (
        pd.DataFrame(preds_h).sort_values("h")
        if preds_h
        else pd.DataFrame(columns=["h", "ts_for", "p_anomaly", "is_anomaly"])
    )

    # --- ДНИ ---
    daily_feats = _prepare_daily_X(base, cfg)  # содержит колонку day
    last_day = (
        pd.to_datetime(daily_feats["day"].max()) if not daily_feats.empty else pd.NaT
    )

    preds_d: List[dict] = []
    if "daily" in bundle and bundle["daily"]["models"]:
        Xd_all = daily_feats.drop(columns=["day"])
        for d, pack in bundle["daily"]["models"].items():
            mdl = pack["model"]
            cols = pack["features"]
            if not cols:
                continue
            row = Xd_all.tail(1)[cols]
            if row.empty:
                continue
            p = float(mdl.predict_proba(row)[0])
            preds_d.append(
                {
                    "d": int(d),
                    "day_for": (last_day + pd.Timedelta(days=int(d)))
                    if pd.notna(last_day)
                    else pd.NaT,
                    "p_anomaly": round(p, 4),
                    "is_anomaly": int(_threshold(p, 0.5)),
                }
            )
    df_daily = (
        pd.DataFrame(preds_d).sort_values("d")
        if preds_d
        else pd.DataFrame(columns=["d", "day_for", "p_anomaly", "is_anomaly"])
    )

    # Сохранение артефактов
    out_hourly = os.path.join(cfg.outdir, "pred_hourly.csv")
    out_daily = os.path.join(cfg.outdir, "pred_daily.csv")
    df_hourly.to_csv(out_hourly, index=False)
    df_daily.to_csv(out_daily, index=False)

    return df_hourly, df_daily


if __name__ == "__main__":
    h, d = predict_all()
    print("Hourly predictions:\n", h)
    print("Daily predictions:\n", d)
