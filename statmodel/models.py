from __future__ import annotations
"""
Обёртка над моделями классификации (CatBoost / Sklearn) для пайплайна.

Содержит:
  * SkModel — тонкая обёртка, унифицирующая .fit/.predict_proba и расчёт метрик;
  * build_model — фабрика моделей по имени с разумными дефолтами.
"""

from dataclasses import dataclass
from typing import Optional, List, Sequence, Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None  # type: ignore
    _HAS_CATBOOST = False


@dataclass
class SkModel:
    """Единая обёртка для моделей бинарной классификации.

    Предоставляет совместимый интерфейс для моделей CatBoost и sklearn,
    а также вспомогательные методы для обучения, инференса и метрик.

    Attributes:
        model: Нативный объект модели (CatBoostClassifier или модель sklearn).
    """
    model: object  # CatBoostClassifier или любой совместимый .fit/.predict_proba

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        class_weight: Optional[Dict[int, float]] = None,
        cat_features: Optional[Sequence[Union[int, str]]] = None,
    ) -> None:
        """Обучает модель на данных.

        Поддерживает взвешивание классов и передачу категориальных признаков
        (для CatBoost через индексы или имена колонок).

        Args:
            X: Матрица признаков (обучение).
            y: Вектор бинарных меток (0/1).
            class_weight: Веса классов {0: w0, 1: w1}. Если None — без весов.
            cat_features: Список имён или индексов категориальных признаков.
                Используется только CatBoost; для sklearn будет проигнорировано.

        Returns:
            None
        """
        sample_weight: Optional[np.ndarray] = None
        if class_weight is not None:
            cw0 = float(class_weight.get(0, 1.0))
            cw1 = float(class_weight.get(1, 1.0))
            sample_weight = np.where(y.values.astype(int) == 1, cw1, cw0)

        # Нормализуем cat_features (поддержка имён и индексов)
        cat_idx: Optional[List[int]] = None
        if cat_features:
            if isinstance(next(iter(cat_features)), str):
                cols = list(X.columns)
                cat_idx = [cols.index(c) for c in cat_features if c in cols]
            else:
                cat_idx = [int(i) for i in cat_features]
            if len(cat_idx) == 0:
                cat_idx = None

        # Обучение
        if cat_idx is not None and hasattr(self.model, "fit"):
            self.model.fit(
                X,
                y.astype(int),
                sample_weight=sample_weight,
                cat_features=cat_idx,  # применимо для CatBoost
                verbose=False,         # чтобы не засорять лог
            )
        else:
            self.model.fit(
                X,
                y.astype(int),
                sample_weight=sample_weight,
                verbose=False if hasattr(self.model, "get_params") else None,
            )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Возвращает вероятности класса 1.

        Args:
            X: Матрица признаков для инференса.

        Returns:
            Numpy-массив формы (n_samples,) или (n_samples, 2) — здесь возвращаем
            столбец вероятностей положительного класса как (n_samples,).
        """
        proba = self.model.predict_proba(X)
        # CatBoost/Sklearn возвращают (n,2); берём столбец вероятностей класса 1
        if proba.ndim == 2:
            return proba[:, 1]
        return proba  # на случай пользовательской модели

    def metrics(self, y_true: pd.Series, proba: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
        """Считает набор метрик по вероятностям.

        Args:
            y_true: Истинные метки (0/1).
            proba: Предсказанные вероятности класса 1.
            thr: Порог бинаризации для F1/precision/recall (по умолчанию 0.5).

        Returns:
            Словарь с ключами:
              - "f1": F1-score;
              - "roc_auc": ROC-AUC (NaN, если один класс в y_true).
        """
        y_true_arr = y_true.astype(int).values
        y_pred = (proba >= float(thr)).astype(int)

        f1 = f1_score(y_true_arr, y_pred, zero_division=0)
        if np.unique(y_true_arr).size > 1:
            roc = roc_auc_score(y_true_arr, proba)
        else:
            roc = float("nan")

        return {"f1": float(f1), "roc_auc": float(roc)}


def build_model(name: str) -> SkModel:
    """Фабрика моделей по имени.

    Поддерживаемые варианты:
      * "catboost" / "cat" / "cb" — CatBoostClassifier (если установлен);
      * любой другой — HistGradientBoostingClassifier (sklearn) как fallback.

    Параметры CatBoost подобраны консервативно для коротких рядов и дисбаланса.

    Args:
        name: Имя модели.

    Returns:
        SkModel: Обёртка над готовым классификатором.
    """
    if name.lower() in ("catboost", "cat", "cb") and _HAS_CATBOOST:
        clf = CatBoostClassifier(
            depth=6,
            learning_rate=0.05,
            iterations=500,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        return SkModel(model=clf)

    # fallback: бустинг из sklearn, если catboost недоступен
    from sklearn.ensemble import HistGradientBoostingClassifier
    print("CatBoost недоступен — используется HistGradientBoostingClassifier")
    clf = HistGradientBoostingClassifier(random_state=42)
    return SkModel(model=clf)
