"""Обучение байевской модели по «часу недели» и онлайн-детекция аномалий.

Модуль включает:
- Построение концепции по скользящему окну обучения (медианы, MAD, погодные беты).
- Онлайн-оценку отклонений по z-оценкам и байесовский скоринг гипотез причин.
- Вспомогательные функции для нормализации входов.

Все числовые пороги/приоры тянутся из config.py (через OnlineConfig).
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from mlmodel import config as cfg

# -------- колонки --------
COL_DATE: str = "Дата"
COL_HOURIDX: str = "Время суток, ч"
COL_SUPPLY: str = "Подача, м3"
COL_RETURN: str = "Обратка, м3"
COL_ODPU_GVS: str = "Потребление за период, м3 ГВС"
COL_ITP_XVS: str = "Потребление за период, м3 ХВС"
COL_T1: str = "T1 гвс, оС"
COL_T2: str = "T2 гвс, оС"
COL_WEATHER: str = "Погода"
COL_WEEKDAY: str = "День недели"


# ========== вспомогалки ==========
def _norm_weekday_series(s: pd.Series) -> pd.Series:
    """Нормализует поле «День недели» к диапазону 0..6 (Пн=0).

    Поддерживаются входы 0..6 и 1..7. Иначе возвращаются NA.

    Args:
        s: Серия с номерами дня недели.

    Returns:
        Серия типа Int64 со значениями 0..6 или NA.
    """
    sd = pd.to_numeric(s, errors="coerce")
    if sd.dropna().between(0, 6).all():
        return sd.astype("Int64")
    if sd.dropna().between(1, 7).all():
        return (sd - 1).astype("Int64")
    return pd.Series([pd.NA] * len(s), dtype="Int64")


def _to_float(x, default: float = 0.0) -> float:
    """Безопасно приводит значение к float с поддержкой NA/None/NaN.

    Args:
        x: Произвольное значение.
        default: Значение по умолчанию, если привести не удалось.

    Returns:
        Число с плавающей точкой.
    """
    try:
        v = pd.to_numeric(x, errors="coerce")
        return float(v) if pd.notna(v) else float(default)
    except Exception:
        return float(default)


def _hour_of_week_from_row(row: pd.Series) -> int:
    """Возвращает индекс часа недели 0..167 для строки.

    Если присутствует колонка «День недели», она имеет приоритет, иначе берётся
    из ``ts``.

    Args:
        row: Строка данных (серия) с полем ``ts`` и опционально «День недели».

    Returns:
        Целое 0..167.
    """
    if COL_WEEKDAY in row and pd.notna(row[COL_WEEKDAY]):
        wd_series = _norm_weekday_series(pd.Series([row[COL_WEEKDAY]]))
        wd_val = wd_series.iloc[0]
        wd = (
            int(wd_val)
            if pd.notna(wd_val)
            else int(pd.to_datetime(row["ts"]).dayofweek)
        )
    else:
        ts = pd.to_datetime(row["ts"])
        wd = int(ts.dayofweek)
    hour = int(pd.to_datetime(row["ts"]).hour)
    return wd * 24 + hour


def split_train_test(
    df_sorted: pd.DataFrame,
    *,
    train_days: Optional[int] = None,
    test_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разбивает хвост данных на train/test по дням.

     Если значения не переданы, берутся из cfg.TRAINING.

    Args:
        df_sorted: Данные, отсортированные по ``ts``.
        train_days: Кол-во суток в обучающем окне.
        test_days: Кол-во суток в тестовом (онлайн) окне.

    Returns:
        Пара (train_df, test_df).
    """
    assert "ts" in df_sorted.columns
    if train_days is None:
        train_days = int(cfg.TRAINING["train_days"])
    if test_days is None:
        test_days = int(cfg.TRAINING["test_days"])

    hours = (train_days + test_days) * 24
    tail = df_sorted.tail(hours)
    train = tail.head(train_days * 24)
    test = tail.tail(test_days * 24)
    return train, test


# ========== БАЗИС ==========
@dataclass
class Baseline:
    """Носитель базовых статистик по «часу недели».

    Attributes:
        stats: Таблица по индексам how=0..167 с медианами/@/погодными бетами.
        global_sigma_delta: Глобальная MAD-оценка для delta (подстраховка).
        global_sigma_ratio: Глобальная MAD-оценка для ratio (подстраховка).
    """

    stats: pd.DataFrame  # index=0..167
    global_sigma_delta: float
    global_sigma_ratio: float


def _MAD(x: pd.Series) -> float:
    """Робастная дисперсия: 1.4826 * median(|x - median(x)|) + eps.

    Совместима со старыми версиями pandas, где нет Series.mad().

    Args:
        x: Числовая серия.

    Returns:
        Робастная оценка @.
    """
    v = pd.to_numeric(x, errors="coerce")
    med = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - med)))
    return 1.4826 * mad + 1e-6


def fit_baseline(df_train: pd.DataFrame) -> Baseline:
    """Строит baseline по обучающему окну.

    Для каждого часа недели (how=0..167) вычисляет:
    медианы mu_* (itp, odpu, delta, ratio), робастные (MAD),
    а также погодные поправки (беты) при наличии столбца «Погода».

    Args:
        df_train: Тренировочное окно с колонкой ``ts`` и измерениями.

    Returns:
        Экземпляр Baseline с готовыми статистиками и глобальными @.
    """
    assert "ts" in df_train.columns, "ожидаю колонку ts после normalize_time"
    df = df_train.copy()

    xvs = pd.to_numeric(df.get(COL_ITP_XVS), errors="coerce").clip(lower=0)
    gvs = pd.to_numeric(df.get(COL_ODPU_GVS), errors="coerce").clip(lower=0)
    eps = 1e-12
    delta = xvs - gvs
    ratio = gvs / np.maximum(xvs, eps)

    if COL_WEEKDAY in df.columns:
        wd = (
            _norm_weekday_series(df[COL_WEEKDAY])
            .fillna(df["ts"].dt.dayofweek)
            .astype(int)
        )
        how = wd * 24 + df["ts"].dt.hour
    else:
        how = df["ts"].dt.dayofweek * 24 + df["ts"].dt.hour

    weather = (
        pd.to_numeric(df[COL_WEATHER], errors="coerce")
        if (COL_WEATHER in df.columns)
        else None
    )
    w_med = float(np.nanmedian(weather)) if weather is not None else np.nan
    w_c = weather - w_med if weather is not None else None

    tmp = pd.DataFrame(
        {
            "how": how.astype(int),
            "itp": xvs.values,
            "odpu": gvs.values,
            "delta": delta.values,
            "ratio": ratio.values,
            "weather": weather.values if weather is not None else np.nan,
            "wc": w_c.values if w_c is not None else np.nan,
        }
    )

    def _robust_beta(y: pd.Series, wc: pd.Series) -> float:
        """Оценивает наклон в модели y ~ a + b*wc на подвыборке одного how.

        При недостатке валидных наблюдений возвращает 0.0.

        Args:
            y: Целевая величина.
            wc: Центрированная погода (weather - median).

        Returns:
            Оценка наклона b.
        """
        y = pd.to_numeric(y, errors="coerce")
        wc = pd.to_numeric(wc, errors="coerce")
        m = y.notna() & wc.notna()
        if m.sum() < 12:
            return 0.0
        denom = float((wc[m] ** 2).sum())
        if denom <= 0:
            return 0.0
        num = float((wc[m] * y[m]).sum())
        return num / denom

    def _agg(g: pd.DataFrame) -> pd.Series:
        """Агрегатор по одному how: возвращает набор μ/@/β."""
        res = {
            "mu_itp": g["itp"].median(),
            "mu_odpu": g["odpu"].median(),
            "mu_delta": g["delta"].median(),
            "mu_ratio": g["ratio"].median(),
            "sigma_itp": _MAD(g["itp"]),
            "sigma_odpu": _MAD(g["odpu"]),
            "sigma_delta": _MAD(g["delta"]),
            "sigma_ratio": _MAD(g["ratio"]),
            "p_low_delta": g["delta"].quantile(cfg.DETECTOR.get("q_low", 0.05)),
            "p_high_delta": g["delta"].quantile(cfg.DETECTOR.get("q_high", 0.95)),
            "p_low_ratio": g["ratio"].quantile(cfg.DETECTOR.get("q_low", 0.05)),
            "p_high_ratio": g["ratio"].quantile(cfg.DETECTOR.get("q_high", 0.95)),
        }
        if g["wc"].notna().any():
            res["w_med"] = float(np.nanmedian(g["weather"]))
            res["beta_delta_w"] = _robust_beta(g["delta"], g["wc"])
            res["beta_ratio_w"] = _robust_beta(g["ratio"], g["wc"])
        else:
            res["w_med"] = np.nan
            res["beta_delta_w"] = 0.0
            res["beta_ratio_w"] = 0.0
        return pd.Series(res)

    stats = tmp.groupby("how").apply(_agg)
    stats = stats.reindex(range(168)).ffill().bfill()

    global_sigma_delta = float(_MAD(tmp["delta"]))
    global_sigma_ratio = float(_MAD(tmp["ratio"]))
    return Baseline(
        stats=stats,
        global_sigma_delta=global_sigma_delta,
        global_sigma_ratio=global_sigma_ratio,
    )


# ========== Онлайн-детектор ==========
@dataclass
class OnlineConfig:
    """Конфигурация онлайн-детектора.

    Значения по умолчанию подтягиваются из config.ONLINE.

    Attributes:
        zdelta_alarm: Порог |z_delta| для срабатывания аномалии.
        zratio_alarm: Порог |z_ratio| для срабатывания аномалии.
        min_abs_delta_m3: Минимальная |Δ| в м³ для принятия сигнала.
        posterior_alarm: Порог постериорной вероятности (для UI/логики).
        prior: Словарь априорных вероятностей по гипотезам.
    """

    zdelta_alarm: float | None = None
    zratio_alarm: float | None = None
    min_abs_delta_m3: float | None = None
    posterior_alarm: float | None = None
    prior: Dict[str, float] | None = None

    # опциональный квантильный коридор
    use_quantile_corridor: bool | None = None
    q_low: float | None = None
    q_high: float | None = None

    # Гистерезис и severe
    zdelta_enter: float | None = None
    zdelta_exit: float | None = None
    zratio_enter: float | None = None
    zratio_exit: float | None = None
    z_severe: float | None = None
    # Низкие потоки — гейт
    min_flow_m3h: float | None = None

    def __post_init__(self) -> None:
        """Заполняет None значениями из конфигурации."""
        C = cfg.ONLINE
        if self.zdelta_alarm is None:
            self.zdelta_alarm = C["zdelta_alarm"]
        if self.zratio_alarm is None:
            self.zratio_alarm = C["zratio_alarm"]
        if self.min_abs_delta_m3 is None:
            self.min_abs_delta_m3 = C["min_abs_delta_m3"]
        if self.posterior_alarm is None:
            self.posterior_alarm = C["posterior_alarm"]
        if self.prior is None:
            self.prior = dict(C["prior"])
        # гистерезис + severe
        if self.zdelta_enter is None:
            self.zdelta_enter = C.get("zdelta_enter", self.zdelta_alarm)
        if self.zdelta_exit is None:
            self.zdelta_exit = C.get("zdelta_exit", self.zdelta_alarm)
        if self.zratio_enter is None:
            self.zratio_enter = C.get("zratio_enter", self.zratio_alarm)
        if self.zratio_exit is None:
            self.zratio_exit = C.get("zratio_exit", self.zratio_alarm)
        if self.z_severe is None:
            self.z_severe = C.get("z_severe", 5.0)
        D = cfg.DETECTOR
        if self.use_quantile_corridor is None:
            self.use_quantile_corridor = bool(D.get("use_quantile_corridor", False))
        if self.q_low is None:
            self.q_low = float(D.get("q_low", 0.05))
        if self.q_high is None:
            self.q_high = float(D.get("q_high", 0.95))
        if self.min_flow_m3h is None:
            self.min_flow_m3h = float(D.get("min_flow_m3h", 0.2))


@dataclass
class OnlineState:
    """Состояние онлайн-детектора: baseline + конфиг."""

    baseline: Baseline
    cfg: OnlineConfig


def _gauss_ll(z: float) -> float:
    """Лог-правдоподобие для |z| в гауссовой аппроксимации."""
    return -0.5 * z * z


def _bern_ll(flag: bool, p_true: float = 0.8) -> float:
    """Лог-правдоподобие для бинарного признака с ``p_true``.

    Args:
        flag: Наблюдаемое значение признака.
        p_true: Ожидаемая вероятность истинного флага.

    Returns:
        Логарифм правдоподобия.
    """
    p = p_true if flag else (1 - p_true + 1e-9)
    return float(np.log(p))


def detect_one(state: OnlineState, row: pd.Series) -> Dict[str, Any]:
    """Оценивает один час: аномалия/нет и постериоры по причинам.

    Этапы:
      1) Вычисляет Δ=ХВС−ГВС, ratio=ГВС/ХВС, корректирует ожидания μ по погоде.
      2) Считает z-оценки по baseline (по «часу недели»).
      3) Фиксирует «быстрые» флаги качества (нули, залипание, всплески).
      4) Решает «аномалия?» по порогам |z| и |Δ|.
      5) Байесовский скоринг гипотез с учётом z-оценок и флагов.

    Args:
        state: Состояние детектора (baseline + конфиг).
        row: Почасовая строка с признаками и служебными полями.

    Returns:
        Словарь с полями:
        - ts, delta_m3, ratio, z_delta, z_ratio,
        - is_anomaly,
        - top_cause, top_cause_prob,
        - p_* по всем гипотезам.
    """
    bsl, cfg_ = state.baseline, state.cfg
    ts = pd.to_datetime(row["ts"])
    how = _hour_of_week_from_row(row)

    xvs = _to_float(row.get(COL_ITP_XVS), default=0.0)
    gvs = _to_float(row.get(COL_ODPU_GVS), default=0.0)
    eps = 1e-12
    delta = xvs - gvs
    ratio = (gvs / max(xvs, eps)) if xvs > 0 else np.nan

    s = bsl.stats.loc[int(how)]
    mu_delta, sigma_delta = float(s["mu_delta"]), float(s["sigma_delta"])
    mu_ratio, sigma_ratio = float(s["mu_ratio"]), float(s["sigma_ratio"])
    if sigma_delta <= 0:
        sigma_delta = bsl.global_sigma_delta
    if sigma_ratio <= 0:
        sigma_ratio = bsl.global_sigma_ratio

    # Погодная поправка (если есть)
    w = (
        _to_float(row.get(COL_WEATHER), default=np.nan)
        if (COL_WEATHER in row)
        else np.nan
    )
    if pd.notna(w) and "w_med" in s and pd.notna(s["w_med"]):
        w_adj = w - float(s["w_med"])
        mu_delta = mu_delta + float(s.get("beta_delta_w", 0.0)) * w_adj
        mu_ratio = mu_ratio + float(s.get("beta_ratio_w", 0.0)) * w_adj

    # Z-скоры
    z_delta = (delta - mu_delta) / (sigma_delta + 1e-9)
    z_ratio = (ratio - mu_ratio) / (sigma_ratio + 1e-9) if np.isfinite(ratio) else 0.0
    # -------- Доп. критерий: квантильные «коридоры» (опционально) --------
    corridor_flag = False
    if state.cfg.use_quantile_corridor:
        # значения сохранялись при обучении; если вдруг NaN, считаем, что коридор не активен
        pld = float(s.get("p_low_delta", np.nan))
        phd = float(s.get("p_high_delta", np.nan))
        plr = float(s.get("p_low_ratio", np.nan))
        phr = float(s.get("p_high_ratio", np.nan))
        out_delta = pd.notna(pld) and pd.notna(phd) and (delta < pld or delta > phd)
        out_ratio = (
            np.isfinite(ratio)
            and pd.notna(plr)
            and pd.notna(phr)
            and (ratio < plr or ratio > phr)
        )
        corridor_flag = bool(out_delta or out_ratio)

    # Быстрые флаги качества
    flags = {
        "itp_zero_run": bool(row.get("itp_xvs_zero_anomaly", False)),
        "odpu_zero_run": bool(row.get("odpu_gvs_zero_anomaly", False)),
        "itp_flat": bool(row.get("itp_xvs_flat_anomaly", False)),
        "odpu_flat": bool(row.get("odpu_gvs_flat_anomaly", False)),
        "itp_spike": bool(row.get("itp_xvs_spike", False)),
        "odpu_spike": bool(row.get("odpu_gvs_spike", False)),
    }

    # Гейт по низкому потоку
    flow_gate = max(xvs, gvs) < cfg_.min_flow_m3h

    # Гистерезис по z: вход/выход
    z_enter = (abs(z_delta) >= cfg_.zdelta_enter) or (abs(z_ratio) >= cfg_.zratio_enter)
    is_anomaly_z = z_enter  # для одиночного часа используем «enter»

    # severe: одиночный пик очень большой |z|
    is_severe = (abs(z_delta) >= cfg_.z_severe) or (abs(z_ratio) >= cfg_.z_severe)

    # Решение об аномалии
    abs_delta_ok = abs(delta) >= cfg_.min_abs_delta_m3
    # «Сырой» флаг (до персистентности): z/коридор/дельта/гейт/over10
    raw_is_anomaly = (
        (is_anomaly_z or corridor_flag) and not flow_gate and abs_delta_ok
    ) or bool(row.get("flag_over10", False))
    # Итог внутри часа с учётом severe
    is_anomaly = raw_is_anomaly or is_severe

    # Байесовский скоринг причин
    logp = {k: float(np.log(v + 1e-12)) for k, v in cfg_.prior.items()}
    logp["normal"] += _gauss_ll(abs(z_delta)) + _gauss_ll(abs(z_ratio))
    logp["leak"] += _gauss_ll((max(0.0, z_delta) - 2.0)) + _gauss_ll(
        (min(0.0, z_ratio) + 2.0)
    )
    logp["leak"] += _bern_ll(not flags["itp_zero_run"], 0.9) + _bern_ll(
        not flags["odpu_zero_run"], 0.9
    )
    logp["odpu_fault"] += _bern_ll(
        flags["odpu_flat"] or flags["odpu_zero_run"], 0.95
    ) + _gauss_ll(max(0.0, z_delta) - 1.5)
    logp["itp_fault"] += _bern_ll(
        flags["itp_flat"] or flags["itp_zero_run"], 0.95
    ) + _gauss_ll(max(0.0, -z_delta) - 1.5)
    logp["tamper"] += _gauss_ll(max(0.0, z_ratio) - 2.0) + _bern_ll(
        (not flags["itp_flat"]) and (not flags["odpu_flat"]), 0.6
    )

    # Техпроцесс: рециркуляция/подпитка и/или низкая ΔT
    recirc_proxy = None
    if (
        COL_SUPPLY in row
        and COL_RETURN in row
        and pd.notna(row[COL_SUPPLY])
        and pd.notna(row[COL_RETURN])
    ):
        recirc_proxy = max(0.0, float(row[COL_SUPPLY]) - float(row[COL_RETURN]))
    low_dT = (
        (row.get(COL_T1, np.nan) - row.get(COL_T2, np.nan))
        if (COL_T1 in row and COL_T2 in row)
        else np.nan
    )
    likely_tech = (recirc_proxy is not None and recirc_proxy > 0.1) or (
        pd.notna(low_dT) and low_dT < 5
    )
    logp["tech"] += _bern_ll(likely_tech, 0.8) + _bern_ll(
        flags["itp_spike"] and not flags["odpu_spike"], 0.7
    )

    # Нормировка лог-вероятностей
    mx = max(logp.values())
    probs = {k: float(np.exp(v - mx)) for k, v in logp.items()}
    ssum = sum(probs.values()) or 1.0
    probs = {k: v / ssum for k, v in probs.items()}
    # Постобработка вероятностей:
    #  - если аномалии НЕТ — всё в «Норму»;
    #  - если аномалия ЕСТЬ — исключаем «Норму» и пере-нормируем по ненормальным причинам.
    if not is_anomaly:
        probs = {k: (1.0 if k == "normal" else 0.0) for k in probs}
    else:
        non_norm = [k for k in probs.keys() if k != "normal"]
        s = sum(probs[k] for k in non_norm) or 1.0
        probs = {k: (probs[k] / s if k in non_norm else 0.0) for k in probs}

    best = max(probs.items(), key=lambda kv: kv[1])

    return {
        "ts": ts,
        "itp_xvs_m3": xvs,
        "odpu_gvs_m3": gvs,
        "delta_m3": delta,
        "ratio": ratio,
        "z_delta": float(z_delta),
        "z_ratio": float(z_ratio),
        "raw_is_anomaly": bool(raw_is_anomaly),
        "severe": bool(is_severe),
        "is_anomaly": bool(is_anomaly),
        "is_anomaly_final": bool(is_anomaly),
        "top_cause": best[0],
        "top_cause_prob": float(best[1]),
        **{f"p_{k}": v for k, v in probs.items()},
    }
