from __future__ import annotations

"""
Обучение мультигоризонтных моделей (часы 1..6, дни 1..3).

Модуль:
  1) Загружает почасовые данные и (опционально) онлайн-метки.
  2) Строит почасовые признаки, проводит backtest для горизонтов H=1..6
     с безутечным выравниванием X.shift(H).
  3) Обучает финальные часовые модели для каждого H.
  4) Агрегирует в дневной уровень, строит дневные признаки,
     проводит backtest для горизонтов D=1..3 (также без утечек),
     обучает финальные дневные модели.
  5) Сохраняет backtest-таблицы, metrics.json и models.pkl.
"""

import os
import json
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from statmodel.config import Config
from statmodel.features import (
    build_hourly_table,
    add_time_lag_roll_feats,
    make_daily_future_labels,
    build_daily_table,
    add_day_lag_roll_feats,
)
from statmodel.backtest import (
    backtest_hourly_multi,
    backtest_daily_multi,
)
from statmodel.models import build_model


def _only_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только числовые/булевы признаки.

    Args:
        df: Исходная таблица признаков.

    Returns:
        DataFrame с числовыми и булевыми колонками (булевы приведены к int8).
    """
    out = df.select_dtypes(include=["number", "bool"]).copy()
    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        out[bool_cols] = out[bool_cols].astype("int8")
    return out


def _impute_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Очищает матрицу признаков от inf и импутирует пропуски.

    Политика:
      * заменяем +/-inf на NaN;
      * затем ffill/bfill;
      * затем остатки NaN -> 0.0.

    Важно:
        Применять к уже «сдвинутым» матрицам (например, X.shift(h)), чтобы
        исключить утечки по времени.
    """
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


def _drop_constant_cols(Xtr: pd.DataFrame) -> List[str]:
    """Возвращает список признаков, имеющих вариативность на трейне.

    Args:
        Xtr: Матрица признаков на трейне.

    Returns:
        Имена колонок, у которых более одного уникального значения.
    """
    return [c for c in Xtr.columns if Xtr[c].nunique(dropna=False) > 1]


def main(cfg: Config = Config()) -> None:
    """Точка входа обучения.

    Процедура:
      * Почасовой backtest (H=1..6) -> `backtest_hourly.csv`.
      * Дневной backtest (D=1..3) -> `backtest_daily.csv`.
      * Метрики усреднённые по фолдам -> `metrics.json`.
      * Пакет моделей -> `models.pkl`.

    Args:
        cfg: Конфигурация пайплайна.
    """
    os.makedirs(cfg.outdir, exist_ok=True)

    # 1) Загрузка
    hourly = pd.read_csv(cfg.hourly_path)
    if "ts" not in hourly.columns:
        raise KeyError(f"В {cfg.hourly_path} отсутствует колонка 'ts'")
    hourly["ts"] = pd.to_datetime(hourly["ts"], errors="coerce")

    dec = None
    if os.path.exists(cfg.online_decisions_path):
        dec = pd.read_csv(cfg.online_decisions_path)

    # 2) Базовые почасовые фичи
    base = build_hourly_table(hourly, dec, cfg.rel_diff_limit, cfg.use_online_labels)
    if "is_anomaly_final" in base.columns:
        matched = base["is_anomaly_final"].notna().sum()
        print(f"[DEBUG] matched online labels: {matched} of {len(base)}")
        print(f"[DEBUG] label ones: {int(base['label'].sum())}")

    feats = add_time_lag_roll_feats(base, cfg.lags_hours, cfg.roll_windows_h)

    drop_cols = ["label", "is_anomaly_final"]
    X_hourly_all = _only_numeric_features(
        feats[[c for c in feats.columns if c not in drop_cols]]
    )
    data_with_ts = pd.concat([feats[["ts"]].copy(), X_hourly_all], axis=1)

    # 3) Backtest по часам
    hourly_horizons: List[int] = [1, 2, 3, 4, 5, 6]
    y_by_h: Dict[int, pd.Series] = {
        h: feats["label"].astype(int).shift(-h).fillna(0).astype(int)
        for h in hourly_horizons
    }
    print(f"[DEBUG][hourly] targets prepared for horizons: {hourly_horizons}")

    bt_h_all = backtest_hourly_multi(
        data_with_ts=data_with_ts,
        y_by_h=y_by_h,
        model_name=cfg.model_name,
        horizons=hourly_horizons,
        n_splits=cfg.n_splits,
        min_train_days=cfg.min_train_days,
        class_weight=cfg.class_weight,
    )
    bt_h_path = os.path.join(cfg.outdir, "backtest_hourly.csv")
    bt_h_all.to_csv(bt_h_path, index=False)

    hourly_metrics_by_h: Dict[int, Dict[str, float]] = {}
    if not bt_h_all.empty:
        metric_cols = [c for c in ["f1", "roc_auc"] if c in bt_h_all.columns]
        if metric_cols:
            agg_h = bt_h_all.groupby("horizon_h")[metric_cols].mean().round(4)
            for h in agg_h.index:
                hourly_metrics_by_h[int(h)] = {
                    m: float(agg_h.loc[h, m]) for m in metric_cols
                }

    # 4) Финальные часовые модели
    hourly_models: Dict[int, object] = {}
    for h in hourly_horizons:
        y_h = y_by_h[h]
        Xh = X_hourly_all.shift(h)
        df_fit = pd.concat([Xh.reset_index(drop=True), y_h.rename("y")], axis=1).dropna(
            subset=["y"]
        )
        if df_fit.empty or df_fit["y"].nunique() < 2:
            print(f"[TRAIN][hourly H={h}] пропуск — пусто или один класс в таргете.")
            continue
        Xh_imp = _impute_and_clean(df_fit.drop(columns=["y"]))
        var_cols = _drop_constant_cols(Xh_imp)
        if not var_cols:
            print(f"[TRAIN][hourly H={h}] пропуск — все признаки константны.")
            continue
        mdl = build_model(cfg.model_name)
        mdl.fit(
            Xh_imp[var_cols], df_fit["y"].astype(int), class_weight=cfg.class_weight
        )
        hourly_models[h] = (mdl, var_cols)

    # 5) Дневной уровень
    daily = build_daily_table(base)
    daily_feats = add_day_lag_roll_feats(daily, cfg.lags_days, cfg.roll_windows_d)

    daily_horizons: List[int] = [1, 2, 3]
    daily_labels_map = make_daily_future_labels(base, tuple(daily_horizons))
    y_by_d: Dict[int, pd.Series] = {
        d: daily_labels_map[d].reindex(daily_feats["day"]).fillna(0).astype(int)
        for d in daily_horizons
    }
    print(f"[DEBUG][daily] targets prepared for horizons: {daily_horizons}")

    bt_d_all = backtest_daily_multi(
        daily_df=daily_feats,
        y_by_d=y_by_d,
        model_name=cfg.model_name,
        horizons=daily_horizons,
        n_splits=cfg.n_splits,
        min_train_days=cfg.min_train_days_daily,
        class_weight=cfg.class_weight,
        cat_feature_names=["dow_cat"],
    )
    bt_d_path = os.path.join(cfg.outdir, "backtest_daily.csv")
    bt_d_all.to_csv(bt_d_path, index=False)

    daily_metrics_by_d: Dict[int, Dict[str, float]] = {}
    if not bt_d_all.empty:
        metric_cols = [c for c in ["f1", "roc_auc"] if c in bt_d_all.columns]
        if metric_cols:
            agg_d = bt_d_all.groupby("horizon_d")[metric_cols].mean().round(4)
            for d in agg_d.index:
                daily_metrics_by_d[int(d)] = {
                    m: float(agg_d.loc[d, m]) for m in metric_cols
                }

    # 6) Финальные дневные модели
    daily_models: Dict[int, object] = {}
    Xd_all = _only_numeric_features(daily_feats.drop(columns=["day"]))
    for d in daily_horizons:
        y_d = y_by_d[d]
        Xd = Xd_all.shift(d)
        df_fit_d = pd.concat(
            [Xd.reset_index(drop=True), y_d.reset_index(drop=True).rename("y")], axis=1
        ).dropna(subset=["y"])
        if df_fit_d.empty or df_fit_d["y"].nunique() < 2:
            print(f"[TRAIN][daily D={d}] пропуск — пусто или один класс в таргете.")
            continue
        Xd_imp = _impute_and_clean(df_fit_d.drop(columns=["y"]))
        var_cols = _drop_constant_cols(Xd_imp)
        if not var_cols:
            print(f"[TRAIN][daily D={d}] пропуск — все признаки константны.")
            continue
        mdl_d = build_model(cfg.model_name)
        mdl_d.fit(
            Xd_imp[var_cols], df_fit_d["y"].astype(int), class_weight=cfg.class_weight
        )
        daily_models[d] = (mdl_d, var_cols)

    # 7) Метрики
    metrics_payload: Dict[str, Dict[int, Dict[str, float]]] = {
        "hourly_cv_mean_by_h": hourly_metrics_by_h,
        "daily_cv_mean_by_d": daily_metrics_by_d,
    }
    with open(os.path.join(cfg.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    # 8) Сериализация
    bundle = {
        "model_name": cfg.model_name,
        "hourly": {
            "features_all": list(X_hourly_all.columns),
            "horizons": hourly_horizons,
            "models": {
                int(h): {"model": mdl, "features": feats}
                for h, (mdl, feats) in hourly_models.items()
            },
        },
        "daily": {
            "features_all": list(Xd_all.columns),
            "horizons": daily_horizons,
            "models": {
                int(d): {"model": mdl, "features": feats}
                for d, (mdl, feats) in daily_models.items()
            },
        },
    }
    with open(os.path.join(cfg.outdir, "models.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    print(
        "OK. Обучены мультигоризонтные модели (часы 1..6, дни 1..3). Результаты в:",
        cfg.outdir,
    )


if __name__ == "__main__":
    main()
