"""Загрузка датасетов варианта, очистка и диагностика геометрии данных.

Из методички: «столбец корректных меток y не должен использоваться в анализе и
служит только для визуализации результатов». Поэтому Dataset хранит X и y
раздельно, а все функции этого модуля, кроме самой загрузки, принимают только
X — метки достаются наружу исключительно для подсчёта метрик качества в
пункте 2.5.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples)
from sklearn.neighbors import NearestNeighbors

from src.io_utils import read_variant


@dataclass
class Dataset:
    """Числовой датасет одного варианта."""

    name: str
    X: np.ndarray                      # (n, p) признаки
    y: np.ndarray                      # (n,) эталонные метки, ТОЛЬКО для оценки
    feature_names: list
    target_names: list
    report: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_targets(self) -> int:
        return int(len(np.unique(self.y)))


def load_dataset(path: str, variant: int, name: str,
                 feature_names: list | None = None) -> Dataset:
    """Читает вариант из xlsx, чистит и возвращает Dataset.

    Пропуски и нечисловые ячейки: строки, где не заполнен хотя бы один
    признак или метка, удаляются целиком. В датасетах лабораторной пропуски
    всегда «строчные» (число непустых ячеек у всех столбцов варианта
    одинаковое), поэтому построчное удаление не выбрасывает половину
    наблюдений и не ломает пропорции классов.
    """
    raw = read_variant(path, variant)
    report = {
        "файл": path,
        "вариант": variant,
        "исходная форма": raw.shape,
        "пропуски": raw.isna().sum().to_dict(),
    }

    df = raw.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    report["нечисловых ячеек"] = int(
        (df.isna().sum().sum() - raw.isna().sum().sum())
    )

    if "y" not in df.columns:
        raise KeyError(f"В варианте {variant} нет столбца y: {list(df.columns)}")

    feature_names = feature_names or [c for c in df.columns if c != "y"]
    missing = [c for c in feature_names + ["y"] if c not in df.columns]
    if missing:
        raise KeyError(f"Нет столбцов: {missing}")

    before = len(df)
    df = df.dropna(subset=feature_names + ["y"])
    report["строк до/после очистки"] = (before, len(df))
    report["удалено строк"] = before - len(df)
    if df.empty:
        raise ValueError(f"После очистки не осталось ни одной строки (вариант {variant})")

    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.int64)
    report["признаки"] = feature_names
    report["распределение меток"] = {
        str(int(v)): int(c) for v, c in zip(*np.unique(y, return_counts=True))
    }

    return Dataset(
        name=name,
        X=X,
        y=y,
        feature_names=list(feature_names),
        target_names=[str(int(v)) for v in np.unique(y)],
        report=report,
    )


def print_report(ds: Dataset) -> None:
    """Печатает отчёт о загрузке датасета."""
    r = ds.report
    print(f"      файл            : {r['файл']} (вариант {r['вариант']})")
    print(f"      в файле         : {r['исходная форма'][0]} строк x "
          f"{r['исходная форма'][1]} столбцов (все варианты сразу)")
    print(f"      строк отброшено : {r['удалено строк']} "
          f"({before_after(r)}), осталось {ds.n_samples}")
    print(f"      признаки        : {', '.join(ds.feature_names)}")
    print(f"      метка y         : {len(ds.target_names)} класса, "
          f"распределение {r['распределение меток']}")
    print(f"      y в анализе     : НЕ используется, только для визуализации и "
          f"итоговой оценки (2.5)")


def before_after(report: dict) -> str:
    b, a = report["строк до/после очистки"]
    return f"{b} -> {a}"


# ----------------------------------------------------------------------
# Диагностика геометрии. Эти функции работают ТОЛЬКО на признаках X: именно по
# ним в пункте 2.1 выносится вердикт, пригоден ли алгоритм для данных.
# ----------------------------------------------------------------------
def k_distance(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Расстояние от каждой точки до k-го ближайшего соседа (k-dist профиль).

    Ключевая величина для методов на основе плотности: если между кластерами
    есть «разрыв», у точек на границе k-dist заметно выше, чем у точек в ядре
    кластера, и на профиле виден локоть.
    """
    k = int(min(max(k, 1), X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    return dist[:, k]


def density_probe(X: np.ndarray, k: int = 5, min_samples: int = 5,
                  n_eps: int = 60, noise_limit: float = 0.05) -> dict:
    """Проверяет, есть ли в данных разрыв в плотности, пригодный для DBSCAN.

    Идея: методы на основе плотности работают, если существует такое eps, при
    котором DBSCAN находит несколько кластеров и почти не отбрасывает точки
    как шум. Поэтому перебирается широкий геометрический диапазон eps (от
    0.05 до 8 максимальных k-мерных расстояний) и считается доля значений,
    дающих >= 2 кластеров при доле шума не выше noise_limit.

    Почему не «локоть на k-dist профиле». У плотного кластера из 36 точек
    пятый ближайший сосед лежит ВНУТРИ кластера, поэтому k-dist всех точек
    остаётся мал — разрыв между кластерами на этом профиле не виден вовсе,
    и его «локоть» находится в хвосте от одиночных точек. Ложное срабатывание
    здесь гарантировано. Мера плотности, в отличие от k-dist, напрямую
    отвечает на вопрос, волнует который и есть: можно ли одним eps описать
    несколько кластеров.
    """
    from sklearn.cluster import DBSCAN

    kd = k_distance(X, k=k)
    hi = float(kd.max())
    if hi <= 0:
        return {"fraction_good": 0.0, "min_noise": 1.0, "best": None,
                "eps_range": (0.0, 0.0), "curve": []}

    eps_grid = np.geomspace(hi * 0.05, hi * 8.0, n_eps)
    curve, n_good, min_noise = [], 0, 1.0
    best = None
    for eps in eps_grid:
        labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(X)
        n_clusters = int(len(np.unique(labels[labels != -1])))
        noise = float((labels == -1).mean())
        curve.append({"eps": float(eps), "n_clusters": n_clusters,
                      "noise_share": noise})
        if n_clusters >= 2:
            min_noise = min(min_noise, noise)
            if noise <= noise_limit:
                n_good += 1
                if best is None or n_clusters > best[1]:
                    best = (float(eps), n_clusters, noise)

    return {
        "fraction_good": n_good / len(eps_grid),
        "min_noise": min_noise,
        "best": best,
        "eps_range": (float(eps_grid[0]), float(eps_grid[-1])),
        "curve": curve,
    }


def knn_graph_components(X: np.ndarray, k: int = 5) -> int:
    """Число связных компонент графа k ближайших соседей.

    Если при малом k граф уже связен, данные — единый плотный сгусток, и
    методы, ищущие в плотности «дыры» (DBSCAN, HDBSCAN, OPTICS), не найдут
    границ между кластерами.
    """
    k = int(min(max(k, 1), X.shape[0] - 1))
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    vals = dist[:, 1:].ravel()
    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    # симметризуем: соседство в sklearn асимметрично
    graph = graph.maximum(graph.T)
    n_comp, _ = connected_components(graph, directed=False)
    return int(n_comp)


def estimate_k(X: np.ndarray, k_min: int = 2, k_max: int = 8) \
        -> tuple[int, pd.DataFrame, int]:
    """Оценка числа кластеров по трём индексам качества (без меток y).

    Метки y по условию задания в анализе не участвуют, поэтому k выбирается
    по силуэту, Калински-Гарабаш и Дэвису-Боулдину.

    Основной результат — k по правилу 1-SE: берётся НАИМЕНЬШЕЕ k, силуэт
    которого не более чем на одно стандартное отклонение силуэта отличается
    от максимума. На перекрывающихся данных кривая силуэта плоская, и k по
    строгому максимуму скачет на единицу от шума в данных; правило 1-SE
    (стандартная практика, например в fviz::nbclust) от этого страдает.
    Максимум по всем индексам при этом тоже сохраняется в таблице.
    """
    from sklearn.cluster import KMeans

    upper = int(min(k_max, X.shape[0] - 1))
    rows = []
    for k in range(k_min, upper + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        per_point = silhouette_samples(X, labels)
        sil = float(per_point.mean())
        rows.append({
            "k": k,
            "силуэт": sil,
            "SE": float(per_point.std() / np.sqrt(len(per_point))),
            "Калински-Гарабаш": float(calinski_harabasz_score(X, labels)),
            "Дэвис-Боулдин": float(davies_bouldin_score(X, labels)),
        })
    table = pd.DataFrame(rows).set_index("k")

    best_k = int(table["силуэт"].idxmax())
    threshold = table.loc[best_k, "силуэт"] - table.loc[best_k, "SE"]
    k_1se = int(table[table["силуэт"] >= threshold].index.min())

    table["1-SE порог"] = threshold
    return k_1se, table, best_k


def geometry_report(X: np.ndarray) -> dict:
    """Сводная диагностика геометрии выборки для обоснования выбора алгоритмов.

    Возвращаемые величины:
        estimated_k      — число кластеров по правилу 1-SE;
        k_argmax         — число кластеров по строгому максимуму силуэта;
        density          — результат зондирования разрыва в плотности (см.
                           density_probe): доля eps, дающих >= 2 кластеров
                           почти без шума, и минимально достижимая доля шума;
        knn_components   — связность графа 5 ближайших соседей. Единица означает,
                           что данные образуют связный сгусток без «дыр»
                           в плотности;
        overlap          — оценка перекрытия кластеров: во сколько раз
                           расстояние до k-го соседа больше расстояния до
                           ближайшего. Большие значения означают, что
                           «свои» и «чужие» точки неразличимы;
        skew, kurtosis   — нелинейность маргиналов (важно для Gaussian mixtures).
    """
    from scipy.stats import kurtosis as _kurt, skew as _skew

    k, k_table, k_argmax = estimate_k(X)
    kd = k_distance(X, k=5)
    density = density_probe(X)

    # Перекрытие: для каждой точки сравниваем расстояние до ближайшего и до
    # k-го соседа. Если они сопоставимы, «свои» и «чужие» точки неразличимы.
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    nearest = dist[:, 1]
    overlap = float(np.mean(dist[:, -1] / np.maximum(nearest, 1e-12)))

    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "estimated_k": k,
        "k_argmax": k_argmax,
        "k_table": k_table,
        "kdist_median": float(np.median(kd)),
        "kdist_max": float(kd.max()),
        "density": density,
        "knn_components": knn_graph_components(X, k=5),
        "overlap": overlap,
        "skew": _skew(X, axis=0).round(3).tolist(),
        "kurtosis": _kurt(X, axis=0).round(3).tolist(),
        "std": X.std(axis=0).round(3).tolist(),
    }


def print_geometry(geo: dict) -> None:
    """Печатает результаты geometry_report."""
    d = geo["density"]
    print(f"      точек/признаков        : {geo['n_samples']} / {geo['n_features']}")
    print(f"      оценка числа кластеров : k = {geo['estimated_k']} "
          f"(правило 1-SE), по строгому максимуму силуэта k = {geo['k_argmax']}")
    table = geo["k_table"]
    print("      индексы качества по k  : "
          + "  ".join(f"k={k}:{row['силуэт']:.3f}" for k, row in table.iterrows()))
    print(f"      разрыв в плотности     : "
          f"{d['fraction_good'] * 100:.0f} % значений eps дают >= 2 кластеров "
          f"при доле шума <= 5 %; минимально достижимая доля шума при "
          f">= 2 кластерах = {d['min_noise'] * 100:.1f} %")
    print(f"      k-dist профиль         : медиана = {geo['kdist_median']:.3f}, "
          f"максимум = {geo['kdist_max']:.3f}")
    print(f"      компонент графа 5-NN   : {geo['knn_components']}")
    print(f"      перекрытие (kNN/1NN)   : {geo['overlap']:.2f}")
    print(f"      СКО по признакам       : {geo['std']}")
    print(f"      асимметрия             : {geo['skew']}")
    print(f"      эксцесс                : {geo['kurtosis']}")


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Матрица попарных евклидовых расстояний (используется в разделах 2.3 и 2.4)."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def all_pairs_above(d: np.ndarray, threshold: float) -> int:
    """Сколько пар точек дальше threshold (оценка разрежённости данных)."""
    n = d.shape[0]
    iu = np.triu_indices(n, k=1)
    return int((d[iu] > threshold).sum())


def n_pairs(d: np.ndarray) -> int:
    n = d.shape[0]
    return n * (n - 1) // 2


def feature_correlation(X: np.ndarray) -> pd.DataFrame:
    """Матрица корреляций Пирсона между признаками."""
    return pd.DataFrame(X).corr()


def pairs_iterable(n: int):
    """Все пары индексов без повторов — для отчётов о корреляциях."""
    return combinations(range(n), 2)
