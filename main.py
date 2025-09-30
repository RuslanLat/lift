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
from statmodel.train import main as train_model
from statmodel.predict import predict_all

st.set_page_config(
    page_title='"МосТруба"',
    # page_icon="app/images/favicon.ico",
    layout="wide",
)  # layout = "wide"

css = """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Стат Модель", "ML Модель"])

with tab1:
    col1, col2, col3 = st.columns((1, 3, 1))
    with col2:
        indication_form = st.form("indication_form")
        indication_form.write("Прогнозирование аномалий")
        rel_diff_limit = indication_form.slider(
            min_value=0,
            max_value=20,
            value=10,
            step=1,
            label="предел корреляции показания ХВС и ГВС по ОДПУ, %",
            help="по умолчанию 10%",
            key="rel_diff_limit",
        )
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
        # cfg.THRESHOLDS["rel_diff_limit"] = rel_diff_limit / 100
        # --- Загрузка
        df = pd.read_csv("data/data_jkh.csv", encoding="utf-8", sep=",")

        # --- Подготовка
        df = (
            df.pipe(utils.normalize_time)
            .pipe(utils.harmonize_units)
            .pipe(utils.apply_recirculation, enabled=recirculation)
            .pipe(utils.compute_balance_and_flags, rel_diff_limit=rel_diff_limit / 100)
        )

        # processed_hourly.csv
        hourly_path = "./output/processed_hourly.csv"
        df.to_csv(hourly_path, index=False)

        # alerts.csv
        alerts = utils.build_alerts(df)
        alerts_path = "./output/alerts.csv"
        alerts.to_csv(alerts_path, index=False)

        # daily_report.csv
        daily = utils.daily_report(df)
        daily_path = "./output/daily_report.csv"
        daily.to_csv(daily_path, index=False)

        dec_path = "./output/online_decisions.csv"
        dec_full_path = "./output/online_decisions_full.csv"
        # --- Онлайн-детектор
        decisions = pd.DataFrame()
        decisions_full = pd.DataFrame()
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

                def _postprocess_decisions(decisions_df: pd.DataFrame) -> pd.DataFrame:
                    if decisions_df is None or decisions_df.empty:
                        return decisions_df
                    d = decisions_df.sort_values("ts").reset_index(drop=True).copy()
                    # 1) severe
                    if "severe" in d.columns:
                        severe_mask = d["severe"].astype(bool)
                    else:
                        z_severe = float(cfg.ONLINE.get("z_severe", 5.0))
                        severe_mask = (d["z_delta"].abs() >= z_severe) | (
                            d["z_ratio"].abs() >= z_severe
                        )
                    # 2) базовый флаг
                    base_flag = (
                        d["raw_is_anomaly"]
                        if "raw_is_anomaly" in d.columns
                        else d["is_anomaly"]
                    )
                    base_flag = base_flag.astype(int)
                    # 3) длительность
                    min_h = int(
                        cfg.INCIDENTS.get(
                            "min_duration_h",
                            cfg.INCIDENTS.get("min_duration_for_anomaly_h", 2),
                        )
                    )
                    if min_h < 1:
                        min_h = 1
                    roll = (
                        base_flag.rolling(min_h, min_periods=min_h)
                        .sum()
                        .fillna(0)
                        .astype(int)
                    )
                    persist_ok = roll.ge(min_h)
                    # 4) финальный флаг
                    d["is_anomaly_final"] = (persist_ok | severe_mask).astype(bool)
                    d["is_anomaly"] = d["is_anomaly_final"]
                    # 5) синхронизируем вероятности
                    prob_cols = [c for c in d.columns if c.startswith("p_")]
                    if prob_cols:
                        mask_norm = ~d["is_anomaly"]
                        for k in [
                            "normal",
                            "leak",
                            "odpu_fault",
                            "itp_fault",
                            "tamper",
                            "tech",
                        ]:
                            col = f"p_{k}"
                            if col in d.columns:
                                d.loc[mask_norm, col] = 1.0 if k == "normal" else 0.0
                        d["top_cause"] = (
                            d[prob_cols]
                            .idxmax(axis=1)
                            .str.replace("^p_", "", regex=True)
                        )

                    # 6) человеко-понятные поля
                    def pct(x):
                        try:
                            return round(float(x) * 100, 1)
                        except Exception:
                            return x

                    for k, rus in cfg.RU_CAUSE.items():
                        col_src = f"p_{k}"
                        col_dst = f"P({rus}), %"
                        if col_src in d.columns:
                            d[col_dst] = d[col_src].apply(pct)
                    d["Причина (top)"] = d["top_cause"].map(cfg.RU_CAUSE).fillna("—")
                    d["Интерпретация"] = d["top_cause"].map(cfg.RU_RULE).fillna("—")
                    d["Аномалия"] = d["is_anomaly"].astype(bool)
                    return d

                # Построчное решение по каждому часу test-окна
                decisions = pd.DataFrame(
                    [detect_one(state, row) for _, row in df_test.iterrows()]
                )
                decisions = _postprocess_decisions(decisions)

                decisions_full = pd.DataFrame(
                    [detect_one(state, row) for _, row in df_sorted.iterrows()]
                )
                decisions_full = _postprocess_decisions(decisions_full)
                # decisions.to_csv(dec_path, index=False)
                decisions.to_csv(dec_path, index=False)
                decisions_full.to_csv(dec_full_path, index=False)
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

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False


def handle_button_click():
    """Callback function to update session state on button click."""
    st.session_state.button_clicked = not st.session_state.button_clicked


with tab2:
    if st.button("обучить", on_click=handle_button_click):
        with st.spinner("идёт обучение модели...", show_time=True):
            train_model()
        st.write("модель обучена")
    if st.session_state.button_clicked:
        if st.button("предсказать"):
            h, d = predict_all()
            st.subheader("Прогнозы на ближайшие 1–6 часов")
            st.dataframe(h)
            st.subheader("Прогнозы на ближайшие 1–3 дня")
            st.dataframe(d)
