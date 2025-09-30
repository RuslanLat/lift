from __future__ import annotations

"""
Конфигурация пайплайна прогнозирования аномалий.

Файл задаёт пути к данным/артефактам, параметры формирования признаков,
горизонты прогнозирования и настройки бэктеста/моделей.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional


@dataclass
class Config:
    """Глобальная конфигурация для обучения и инференса.

    Атрибуты:
        hourly_path: Путь к почасовым данным после предобработки
            (ожидается CSV с колонкой ``ts``).
        online_decisions_path: Путь к таблице онлайн-решений детектора
            (CSV с колонкой ``ts`` и опциональной ``is_anomaly_final``).
        daily_report_path: Путь к суточному отчёту (опционально, для аналитики).

        outdir: Директория для сохранения артефактов обучения/бэктеста/моделей.

        use_online_labels: Если True, используем онлайн-метки ``is_anomaly_final``.
            Иначе формируем метку из порога по ``rel_diff``.
        rel_diff_limit: Порог относительного расхождения для суррогатной метки.

        hourly_horizons_h: Горизонты прогноза (в часах) для часовой модели.
        daily_horizons_d: Горизонты прогноза (в сутках) для дневной модели.

        lags_hours: Набор лагов (в часах) для построения почасовых признаков.
        roll_windows_h: Окна (в часах) для скользящих агрегатов почасовых признаков.

        lags_days: Набор лагов (в сутках) для построения дневных признаков.
        roll_windows_d: Окна (в сутках) для скользящих агрегатов дневных признаков.

        model_name: Имя модели. Поддерживаются "catboost"/"cat"/"cb"
            (CatBoostClassifier) или fallback на модель sklearn.
        class_weight: Веса классов для борьбы с дисбалансом ({0: w0, 1: w1}).

        n_splits: Число разбиений в TimeSeriesSplit (expanding backtest).
        min_train_days: Минимальная длина трейна в днях для часовой модели.
        min_train_days_daily: Минимальная длина трейна в днях для дневной модели.
    """

    # Пути к данным/отчётам
    hourly_path: str = "./output/processed_hourly.csv"
    online_decisions_path: str = "./output/online_decisions_full.csv"
    daily_report_path: str = "./output/daily_report.csv"

    # Артефакты
    outdir: str = "./output"

    # Источники разметки
    use_online_labels: bool = True
    rel_diff_limit: float = 0.10

    # Мульти-горизонты
    hourly_horizons_h: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    daily_horizons_d: Tuple[int, ...] = (1, 2, 3)

    # Признаки (короткие окна под 30 дней истории)
    lags_hours: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    roll_windows_h: Tuple[int, ...] = (2, 3, 4, 6)

    lags_days: Tuple[int, ...] = (1, 2)
    roll_windows_d: Tuple[int, ...] = (2, 3)

    # Модель
    model_name: str = "catboost"
    class_weight: Optional[Dict[int, float]] = field(
        default_factory=lambda: {0: 1.0, 1: 3.0}
    )

    # Backtest (expanding)
    n_splits: int = 2
    min_train_days: int = 5
    min_train_days_daily: int = 5
