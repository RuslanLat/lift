import streamlit as st
import pandas as pd
from mlmodel.anomaly_detector import (
    fit_baseline,
    OnlineState,
    OnlineConfig,
    detect_one,
    split_train_test,
)
from mlmodel import utils
from mlmodel import config as cfg

st.set_page_config(
    page_title='"МосТруба"',
    # page_icon="app/images/favicon.ico",
    layout="wide",
)  # layout = "wide"

st.header("Прогнозирование аномалий")

col1, col2, col3 = st.columns((1, 3, 1))
with col2:
    indication_form = st.form("indication_form")
    indication_form.write("Прогнозирование аномалий")
    recirculation = indication_form.checkbox(
        label="использовать Подача-Обратка как прокси ОДПУ",
        help="использовать Подача-Обратка как прокси ОДПУ (если нужно)",
        key="recirculation",
    )
    # online = indication_form.checkbox(
    #     label="включить онлайн-детектор (последние N дней)",
    #     help="включить онлайн-детектор (последние N дней)",
    #     key="online",
    # )
    train_days = indication_form.slider(
        min_value=7,
        max_value=30,
        value=29,
        step=1,
        label="дней для обучения baseline",
        help="по умолчанию 29 дней",
        key="train_days",
    )
    test_days = indication_form.slider(
        min_value=7,
        max_value=30,
        value=1,
        step=1,
        label="дней для окна онлайна",
        help="по умолчанию 1 день",
        key="test_days",
    )
    indication_submitted = indication_form.form_submit_button("прогноз")

if indication_submitted:
    # --- Загрузка
    df = pd.read_csv("data/data_jkh.csv", encoding="utf-8", sep=",")

    # --- Подготовка
    df = (
        df.pipe(utils.normalize_time)
        .pipe(utils.harmonize_units)
        .pipe(utils.apply_recirculation, enabled=recirculation)
        .pipe(utils.compute_balance_and_flags)
    )

    # --- Онлайн-детектор
    decisions = pd.DataFrame()
    current = None

    if True:
        df_sorted = df.sort_values("ts")
        df_train, df_test = split_train_test(
            df_sorted, train_days=train_days, test_days=test_days
        )
        if len(df_train) > 0 and len(df_test) > 0:
            baseline = fit_baseline(df_train)
            # OnlineConfig подтянет пороги/приоры из config.ONLINE
            state = OnlineState(baseline=baseline, cfg=OnlineConfig())

            # Построчное решение по каждому часу test-окна
            decisions = pd.DataFrame(
                [detect_one(state, row) for _, row in df_test.iterrows()]
            )

            # -------- Post-hoc устойчивость по длительности (минимум 2 часа) --------
            # Берём параметр из конфигурации; по умолчанию 2
            min_h = int(cfg.INCIDENTS.get("min_duration_for_anomaly_h", 2))
            if not decisions.empty and min_h > 1:
                decisions = decisions.sort_values("ts").copy()
                # скользящая сумма по бинарному флагу
                s = (
                    decisions["is_anomaly"]
                    .astype(int)
                    .rolling(min_h, min_periods=min_h)
                    .sum()
                )
                # is_anomaly становится True только если подряд >= min_h часов с True
                # Присваиваем «текущему» часу: удобнее сдвинуть окно на конец
                decisions["is_anomaly"] = (s >= min_h).fillna(False).astype(bool).values

                # опционально: можно сбросить вероятности к «норме» для часов,
                # которые отсеялись пост-фильтром
                mask_norm = ~decisions["is_anomaly"]
                for k in [
                    "normal",
                    "leak",
                    "odpu_fault",
                    "itp_fault",
                    "tamper",
                    "tech",
                ]:
                    col = f"p_{k}"
                    if col in decisions.columns:
                        decisions.loc[mask_norm, col] = 1.0 if k == "normal" else 0.0
                # top_cause пересчитать по обновлённым probs
                decisions["top_cause"] = (
                    decisions[
                        [
                            "p_normal",
                            "p_leak",
                            "p_odpu_fault",
                            "p_itp_fault",
                            "p_tamper",
                            "p_tech",
                        ]
                    ]
                    .idxmax(axis=1)
                    .str.replace("^p_", "", regex=True)
                )

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
                decisions["Причина (top)"] = (
                    decisions["top_cause"].map(cfg.RU_CAUSE).fillna("—")
                )
                decisions["Интерпретация"] = (
                    decisions["top_cause"].map(cfg.RU_RULE).fillna("—")
                )
                decisions["Аномалия"] = decisions["is_anomaly"].astype(bool)

            # decisions.to_csv(dec_path, index=False)

            # current_decision.json
            current = decisions.iloc[-1].to_dict() if not decisions.empty else None
            # with open(current_json_path, "w", encoding="utf-8") as f:
            #     json.dump(
            #         {
            #             "ts": str(current.get("ts")) if current else None,
            #             "is_anomaly": bool(current.get("is_anomaly"))
            #             if current
            #             else None,
            #             "top_cause": cfg.RU_CAUSE.get(
            #                 current.get("top_cause"), str(current.get("top_cause"))
            #             )
            #             if current
            #             else None,
            #             "top_cause_prob": round(
            #                 float(current.get("top_cause_prob", 0.0)), 3
            #             )
            #             if current
            #             else None,
            #             "delta_m3": round(float(current.get("delta_m3", 0.0)), 4)
            #             if current
            #             else None,
            #             "ratio": (
            #                 round(float(current.get("ratio")), 4)
            #                 if (current and pd.notna(current.get("ratio")))
            #                 else None
            #             ),
            #             "z_delta": round(float(current.get("z_delta", 0.0)), 2)
            #             if current
            #             else None,
            #             "z_ratio": round(float(current.get("z_ratio", 0.0)), 2)
            #             if current
            #             else None,
            #         },
            #         f,
            #         ensure_ascii=False,
            #         indent=2,
            #     )
        else:
            # Пустая разметка при нехватке данных
            pass
            # decisions.to_csv(dec_path, index=False)
            # with open(current_json_path, "w", encoding="utf-8") as f:
            #     json.dump({}, f, ensure_ascii=False, indent=2)
    else:
        # Онлайн-режим отключён: всё равно выпустим пустые файлы,
        # чтобы набор артефактов был полным и предсказуемым.
        pass
        # decisions.to_csv(dec_path, index=False)
        # with open(current_json_path, "w", encoding="utf-8") as f:
        #     json.dump({}, f, ensure_ascii=False, indent=2)

    st.subheader("Результат прогнозирования")
    st.dataframe(decisions)
    incidents = utils.build_incidents(decisions)
    st.subheader("Результирующие инциденты")
    st.dataframe(incidents)
    st.subheader("Сравнительный график прогноза")
    daily = utils.daily_report(df)
    utils.plot_daily_bars(daily)
