"""Запуск пайплайна: подготовка данных, отчёты и онлайн-детектор.

Создаёт артефакты:
- processed_hourly.csv
- alerts.csv
- daily_report.csv
- 03_daily_bars.png
- online_decisions.csv
- current_decision.json
- events.csv
- summary.json

Все алгоритмические настройки вынесены в config.py.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import pandas as pd

import utils
import config as cfg
from anomaly_detector import (
    fit_baseline,
    OnlineState,
    OnlineConfig,
    detect_one,
    split_train_test,
)


def main() -> None:
    """Точка входа: читает CSV, строит отчёты и при необходимости запускает online-детектор.

    Шаги пайплайна:
      1. Загрузка CSV с авто-подбором кодировки (fallback на cp1251).
      2. Подготовка: нормализация времени, приведение единиц, опциональная
         рециркуляция как прокси ОДПУ, расчёт баланса и флагов качества.
      3. Сохранение почасовых данных и alert-таблицы.
      4. Суточный отчёт и единственный график (03_daily_bars.png).
      5. (Опционально) Онлайн-детектор: baseline по train-окну и решения по test-окну.
         - online_decisions.csv — почасовая таблица решений с вероятностями.
         - current_decision.json — решение за последний час test-окна.
      6. Сборка инцидентов и итоговая summary.json.

    Аргументы CLI:
        --input:     Путь к CSV-файлу.
        --outdir:    Папка для результатов (создаётся при отсутствии).
        --encoding:  Кодировка входного файла (по умолчанию utf-8).
        --sep:       Разделитель CSV (по умолчанию ",").
        --recirculation: Использовать (Подача−Обратка) как прокси ОДПУ (при наличии каналов).
        --online:    Включить онлайн-детектор (train/test окна по дням).
        --train-days / --test-days: Размеры окон.

    Returns:
        None. Результаты сохраняются на диск.
    """
    ap = argparse.ArgumentParser(description="Детектор аномалий ИТП-ХВС vs ОДПУ-ГВС (урезанный)")
    ap.add_argument("--input", required=True, help="путь к CSV с данными")
    ap.add_argument("--outdir", default="output", help="папка для результатов")
    ap.add_argument("--encoding", default="utf-8", help="кодировка CSV (utf-8/cp1251 и т.п.)")
    ap.add_argument("--sep", default=",", help="разделитель CSV")
    ap.add_argument("--recirculation", action="store_true", help="использовать Подача-Обратка как прокси ОДПУ (если нужно)")
    ap.add_argument("--online", action="store_true", help="включить онлайн-детектор (последние N дней)")
    # <-- дефолты теперь None, чтобы подтянуть их из cfg.TRAINING
    ap.add_argument("--train-days", type=int, default=None, help="дней для обучения baseline (по умолчанию из config.TRAINING)")
    ap.add_argument("--test-days", type=int, default=None, help="дней для окна онлайна (по умолчанию из config.TRAINING)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Загрузка
    try:
        df = pd.read_csv(args.input, encoding=args.encoding, sep=args.sep)
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding="cp1251", sep=args.sep)

    # --- Подготовка
    df = (
        df
        .pipe(utils.normalize_time)
        .pipe(utils.harmonize_units)
        .pipe(utils.apply_recirculation, enabled=args.recirculation)
        .pipe(utils.compute_balance_and_flags)
    )

    # processed_hourly.csv
    hourly_path = os.path.join(args.outdir, "processed_hourly.csv")
    df.to_csv(hourly_path, index=False)

    # alerts.csv
    alerts = utils.build_alerts(df)
    alerts_path = os.path.join(args.outdir, "alerts.csv")
    alerts.to_csv(alerts_path, index=False)

    # daily_report.csv
    daily = utils.daily_report(df)
    daily_path = os.path.join(args.outdir, "daily_report.csv")
    daily.to_csv(daily_path, index=False)

    # 03_daily_bars.png — единственный график
    daily_bars_png = os.path.join(args.outdir, "03_daily_bars.png")
    utils.plot_daily_bars(daily, daily_bars_png)

    # --- Онлайн-детектор
    decisions = pd.DataFrame()
    dec_path = os.path.join(args.outdir, "online_decisions.csv")
    current_json_path = os.path.join(args.outdir, "current_decision.json")
    current = None

    train_days = int(args.train_days) if args.train_days is not None else int(cfg.TRAINING["train_days"])
    test_days  = int(args.test_days)  if args.test_days  is not None else int(cfg.TRAINING["test_days"])

    if args.online:
        df_sorted = df.sort_values("ts")
        df_train, df_test = split_train_test(df_sorted, train_days=train_days, test_days=test_days)
        if len(df_train) > 0 and len(df_test) > 0:
            baseline = fit_baseline(df_train)
            # OnlineConfig подтянет пороги/приоры из config.ONLINE
            state = OnlineState(baseline=baseline, cfg=OnlineConfig())

            # Построчное решение по каждому часу test-окна
            decisions = pd.DataFrame([detect_one(state, row) for _, row in df_test.iterrows()])

            # -------- Post-hoc устойчивость по длительности (минимум 2 часа) --------
            # Берём параметр из конфигурации; по умолчанию 2
            min_h = int(cfg.INCIDENTS.get("min_duration_for_anomaly_h", 2))
            if not decisions.empty and min_h > 1:
                decisions = decisions.sort_values("ts").copy()
                # скользящая сумма по бинарному флагу
                s = decisions["is_anomaly"].astype(int).rolling(min_h, min_periods=min_h).sum()
                # is_anomaly становится True только если подряд >= min_h часов с True
                # Присваиваем «текущему» часу: удобнее сдвинуть окно на конец
                decisions["is_anomaly"] = (s >= min_h).fillna(False).astype(bool).values

                # опционально: можно сбросить вероятности к «норме» для часов,
                # которые отсеялись пост-фильтром
                mask_norm = ~decisions["is_anomaly"]
                for k in ["normal","leak","odpu_fault","itp_fault","tamper","tech"]:
                    col = f"p_{k}"
                    if col in decisions.columns:
                        decisions.loc[mask_norm, col] = (1.0 if k=="normal" else 0.0)
                # top_cause пересчитать по обновлённым probs
                decisions["top_cause"] = decisions[["p_normal","p_leak","p_odpu_fault","p_itp_fault","p_tamper","p_tech"]].idxmax(axis=1).str.replace("^p_","",regex=True)

            if not decisions.empty:
                def pct(x):
                    """Локальный форматтер: долю - проценты с 1 знаков после запятой."""
                    try:
                        return round(float(x) * 100, 1)
                    except Exception:
                        return x

                # Вероятности на русском
                for k, rus in cfg.RU_CAUSE.items():
                    col_src = f"p_{k}"
                    col_dst = f"P({rus}), %"
                    if col_src in decisions.columns:
                        decisions[col_dst] = decisions[col_src].apply(pct)

                # Топ-причина на русском + правило и флаг аномалии
                decisions["Причина (top)"] = decisions["top_cause"].map(cfg.RU_CAUSE).fillna("—")
                decisions["Интерпретация"] = decisions["top_cause"].map(cfg.RU_RULE).fillna("—")
                decisions["Аномалия"] = decisions["is_anomaly"].astype(bool)

            decisions.to_csv(dec_path, index=False)

            # current_decision.json
            current = decisions.iloc[-1].to_dict() if not decisions.empty else None
            with open(current_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "ts": str(current.get("ts")) if current else None,
                    "is_anomaly": bool(current.get("is_anomaly")) if current else None,
                    "top_cause": cfg.RU_CAUSE.get(current.get("top_cause"), str(current.get("top_cause"))) if current else None,
                    "top_cause_prob": round(float(current.get("top_cause_prob", 0.0)), 3) if current else None,
                    "delta_m3": round(float(current.get("delta_m3", 0.0)), 4) if current else None,
                    "ratio": (round(float(current.get("ratio")), 4) if (current and pd.notna(current.get("ratio"))) else None),
                    "z_delta": round(float(current.get("z_delta", 0.0)), 2) if current else None,
                    "z_ratio": round(float(current.get("z_ratio", 0.0)), 2) if current else None,
                }, f, ensure_ascii=False, indent=2)
        else:
            # Пустая разметка при нехватке данных
            decisions.to_csv(dec_path, index=False)
            with open(current_json_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    else:
        # Онлайн-режим отключён: всё равно выпустим пустые файлы,
        # чтобы набор артефактов был полным и предсказуемым.
        decisions.to_csv(dec_path, index=False)
        with open(current_json_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    # events.csv (из decisions)
    inc = utils.build_incidents(decisions) if not decisions.empty else pd.DataFrame()
    events_path = os.path.join(args.outdir, "events.csv")
    inc.to_csv(events_path, index=False)

    # summary.json
    summary = {
        "training_days_used": train_days,
        "test_days_used": test_days,
        "rows_processed": int(len(df)),
        "alerts_count": int(len(alerts)),
        "days": int(daily.shape[0]),
        "hourly_csv": hourly_path,
        "alerts_csv": alerts_path,
        "daily_report_csv": daily_path,
        "events_csv": events_path,
        "online_decisions_csv": dec_path,
        "current_decision_json": current_json_path,
        "daily_bars_png": daily_bars_png,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Готово. Результаты в:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
