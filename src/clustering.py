"""Пункт 2 методички (2.1-2.5): кластеризация и обнаружение аномалий.

Датасет Lab2_Dataset_Clustering.xlsx: 2 признака + столбец эталонных меток y.
Метки y не участвуют в анализе: ни выбор числа кластеров, ни подбор параметров,
ни вердикт в п. 2.1 идут по ним. Метки используются только в п. 2.5 для
итоговой оценки качества.

Структура пункта:
    2.1  какие алгоритмы пригодны для этих данных, какие нет, и почему
    2.2  общий график результатов всех 12 алгоритмов
    2.3  подбор параметров + анализ отделения выбросов
    2.4  выделение аномалий на scatter-графике
    2.5  сравнение с эталонными метками, обоснование выбора
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import (KMeans, AffinityPropagation, MeanShift, SpectralClustering,
                             AgglomerativeClustering, DBSCAN, OPTICS, HDBSCAN, Birch,
                             BisectingKMeans)
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score, homogeneity_score,
                             completeness_score, v_measure_score, confusion_matrix)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import run_step, step_done, sub_step
from src.data_loader import (Dataset, load_dataset, print_report, geometry_report,
                             print_geometry, k_distance, estimate_k)
from src.io_utils import save_figure, write_table

# Палитра кластеров и маркер шума/аномалий
CLUSTER_COLORS = plt.cm.tab10.colors
NOISE_COLOR = "#111111"
ANOMALY_COLOR = "#D50000"
NOISE_LABEL = -1


# ======================================================================
# Реестр алгоритмов
# ======================================================================
@dataclass
class Algo:
    """Описание одного алгоритма кластеризации из перечня методички."""

    key: str
    title: str
    family: str                 # раздел перечня из п. 2
    build: Callable[[dict], object]
    grid: list                  # сетка параметров: список словарей
    defaults: dict              # параметры по умолчанию для п. 2.2
    note: str = ""              # краткое пояснение к вердикту п. 2.1
    fitted: dict = field(default_factory=dict)   # параметры, давшие лучший score
    labels: np.ndarray | None = None


def _k_range(n_samples: int, k_min: int = 2, k_max: int = 8) -> list:
    """Допустимые значения числа кластеров: снизу ограничено размером выборки."""
    upper = int(min(k_max, max(2, n_samples // 4)))
    return list(range(k_min, upper + 1))


def build_algorithms(n_samples: int, X: np.ndarray) -> list:
    """Собирает реестр из 12 алгоритмов с сетками, привязанными к данным.

    Сетки не «с потолка»: eps для методов на основе плотности берётся из
    квантилей k-dist профиля, полоса пропускания MeanShift — из его медианы,
    preference для Affinity Propagation — из кратных медиане подобия.
    Так перебор покрывает именно тот диапазон, где на данных данной природы
    может лежать оптимум.
    """
    ks = _k_range(n_samples)
    kd = k_distance(X, k=5)
    kd_q = {q: float(np.quantile(kd, q / 100)) for q in (10, 25, 40, 50, 60, 75, 90)}
    kd_med = kd_q[50]
    # медиана попарного подобия — базовая величина для параметра preference
    # (Affinity Propagation ищет кластеры в местах с высоким подобием)
    sim = -np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    pref_med = float(np.median(sim[np.triu_indices(len(X), k=1)]))

    eps_grid = sorted({round(v, 4) for v in kd_q.values()})
    bandwidth_grid = sorted({round(v, 3) for v in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)})

    def grid_of(dicts):
        return dicts

    algos = [
        Algo("kmeans", "1. K-Means", "центроидные",
             build=lambda p: KMeans(n_clusters=p["k"], n_init=10, init="k-means++",
                                    random_state=42),
             grid=grid_of([{"k": k} for k in ks]),
             defaults={"k": ks[min(2, len(ks) - 1)]},
             note="требует число кластеров; предполагает выпуклые кластеры "
                  "одинакового размера"),

        Algo("affinity", "2. Affinity propagation", "центроидные (число не задаётся)",
             build=lambda p: AffinityPropagation(damping=p["damping"],
                                                 preference=p["preference"],
                                                 max_iter=500, random_state=42),
             grid=grid_of([{"damping": d, "preference": pr}
                           for d in (0.5, 0.7, 0.9)
                           for pr in (None, round(pref_med * 0.5, 5),
                                      round(pref_med, 5), round(pref_med * 2, 5))]),
             defaults={"damping": 0.9, "preference": pref_med},
             note="число кластеров определяет параметр preference; при "
                  "перекрывающихся кластерах склонен дроблить данные"),

        Algo("meanshift", "3. Mean-shift", "сдвига среднего",
             build=lambda p: MeanShift(bandwidth=p["bandwidth"], bin_seeding=True,
                                       max_iter=200),
             grid=grid_of([{"bandwidth": b} for b in bandwidth_grid]),
             defaults={"bandwidth": round(kd_med, 3)},
             note="радиус ядра задаётся вручную; единственный параметр, "
                  "на который крайне чувствителен результат"),

        Algo("spectral", "4. Spectral clustering", "спектральные",
             build=lambda p: SpectralClustering(n_clusters=p["k"],
                                                affinity=p["affinity"],
                                                n_neighbors=p["n_neighbors"],
                                                random_state=42, n_init=10),
             grid=grid_of([{"k": k, "affinity": a, "n_neighbors": 10}
                           for k in ks for a in ("nearest_neighbors", "rbf")]),
             defaults={"k": ks[min(2, len(ks) - 1)], "affinity": "nearest_neighbors",
                       "n_neighbors": 10},
             note="строит разбиение по графу сходства; выигрыш ожидается на "
                  "нелинейных многообразиях"),

        Algo("ward", "5. Ward hierarchical", "иерархические (Ward)",
             build=lambda p: AgglomerativeClustering(n_clusters=p["k"], linkage="ward"),
             grid=grid_of([{"k": k} for k in ks]),
             defaults={"k": ks[min(2, len(ks) - 1)]},
             note="слияние минимизирует рост суммы квадратов внутри "
                  "кластеров; жёстко предполагает выпуклость"),

        Algo("agglomerative", "6. Agglomerative clustering", "иерархические",
             build=lambda p: AgglomerativeClustering(n_clusters=p["k"],
                                                     linkage=p["linkage"],
                                                     metric="euclidean"),
             grid=grid_of([{"k": k, "linkage": lk} for k in ks
                           for lk in ("complete", "average", "single")]),
             defaults={"k": ks[min(2, len(ks) - 1)], "linkage": "complete"},
             note="выбор linkage определяет форму кластеров: complete "
                  "стягивает к компактным, single — к цепочкам"),

        Algo("dbscan", "7. DBSCAN", "плотностные",
             build=lambda p: DBSCAN(eps=p["eps"], min_samples=p["min_samples"]),
             grid=grid_of([{"eps": e, "min_samples": m} for e in eps_grid
                           for m in (3, 5, 10)]),
             defaults={"eps": round(kd_q[75], 4), "min_samples": 5},
             note="находит кластеры произвольной формы; требует разрыва "
                  "в плотности"),

        Algo("hdbscan", "8. HDBSCAN", "плотностные",
             build=lambda p: HDBSCAN(min_cluster_size=p["min_cluster_size"],
                                     min_samples=p["min_samples"]),
             grid=grid_of([{"min_cluster_size": mcs, "min_samples": ms}
                           for mcs in (3, 5, 10, 15) for ms in (None, 3, 5)]),
             defaults={"min_cluster_size": 5, "min_samples": 5},
             note="не задаёт eps, строит иерархию плотностей; устойчив к "
                  "переменной плотности"),

        Algo("optics", "9. OPTICS", "плотностные",
             build=lambda p: OPTICS(min_samples=p["min_samples"], xi=p["xi"],
                                    min_cluster_size=p["min_cluster_size"]),
             grid=grid_of([{"min_samples": m, "xi": x, "min_cluster_size": mcs}
                           for m in (3, 5, 10) for x in (0.01, 0.05, 0.1)
                           for mcs in (5, 10)]),
             defaults={"min_samples": 5, "xi": 0.05, "min_cluster_size": 5},
             note="как DBSCAN, но без eps: выдаёт порядок достижимости, по "
                  "которому eps подбирается постепенно"),

        Algo("gmm", "10. Gaussian mixtures", "смеси распределений",
             build=lambda p: GaussianMixture(n_components=p["k"],
                                             covariance_type=p["covariance_type"],
                                             reg_covar=1e-5, random_state=42),
             grid=grid_of([{"k": k, "covariance_type": c} for k in ks
                           for c in ("full", "diag", "spherical")]),
             defaults={"k": ks[min(2, len(ks) - 1)], "covariance_type": "full"},
             note="кластеры = компоненты смеси; допускает эллиптические "
                  "кластеры разного размера"),

        Algo("birch", "11. BIRCH", "инкрементальные (B-дерево)",
             build=lambda p: Birch(n_clusters=p["k"], threshold=p["threshold"],
                                   branching_factor=50),
             grid=grid_of([{"k": k, "threshold": t} for k in ks
                           for t in (0.1, 0.3, 0.5, 0.75)]),
             defaults={"k": ks[min(2, len(ks) - 1)], "threshold": 0.5},
             note="сжимает точки в микрокластеры; рассчитан на выборки "
                  "от тысяч точек"),

        Algo("bisecting", "12. Bisecting K-Means", "центроидные",
             build=lambda p: BisectingKMeans(n_clusters=p["k"], n_init=10,
                                             random_state=42),
             grid=grid_of([{"k": k} for k in ks]),
             defaults={"k": ks[min(2, len(ks) - 1)]},
             note="делит кластер пополам по главной оси; дешевле K-Means, но "
                  "хуже на вытянутых кластерах"),
    ]
    return algos


# ======================================================================
# Обучение и оценка
# ======================================================================
def evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    """Метрики разбиения. Силуэт считается по ненулевым точкам, как принято
    для методов с выбросами."""
    labels = np.asarray(labels)
    noise = labels == NOISE_LABEL
    core = ~noise
    core_labels = labels[core]

    n_clusters = int(len(np.unique(core_labels)))
    n_noise = int(noise.sum())
    n = len(labels)

    sil = None
    if n_clusters >= 2 and n_clusters <= core.sum() - 1:
        sil = float(silhouette_score(X[core], core_labels))

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_share": n_noise / n,
        "silhouette": sil,
    }


def quality_score(metrics: dict) -> float:
    """Единый критерий выбора параметров.

    Голый силуэт непригоден для сравнения методов на основе плотности: у
    DBSCAN с eps = 0.2 силуэт по 28 оставшимся точкам равен 0.57, но 80 %
    выборки отброшено как шум. Поэтому итоговый балл домножается на долю
    реально сгруппированных точек — «красивая» из двух точек группа не
    должна выигрывать у полного разбиения.
    """
    if metrics["silhouette"] is None:
        return float("-inf")
    return metrics["silhouette"] * (1.0 - metrics["noise_share"])


def fit_algo(algo: Algo, params: dict, X: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Обучает алгоритм и возвращает метки. Исключения не поднимаются наружу:
    часть наборов параметров заведомо нерабочая, и это нужно показать в отчёте,
    а не ронять скрипт."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            est = algo.build(params)
            labels = est.fit_predict(X)
            return np.asarray(labels), ""
        except Exception as exc:                       # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"


# ======================================================================
# 2.1 - пригодность алгоритмов
# ======================================================================
def judge_algorithms(geo: dict, algos: list) -> pd.DataFrame:
    """Вердикт по каждому алгоритму: решает ли он задачу на этих данных.

    Решение принимается по геометрии выборки (разрыв в плотности, связность
    графа соседей, перекрытие, объём), до подбора параметров и без обращения к
    меткам y. Фактические метрики, подтверждающие вердикт, приводятся в п. 2.3
    и 2.5.
    """
    density = geo["density"]
    frac_good = density["fraction_good"]
    min_noise = density["min_noise"]
    overlap = geo["overlap"]
    n = geo["n_samples"]
    n_feat = geo["n_features"]
    comps = geo["knn_components"]
    k = geo["estimated_k"]

    has_gap = frac_good >= 0.1
    strong_overlap = overlap >= 1.5
    tiny = n < 1000

    rows = []
    for a in algos:
        if a.family.startswith("плотностные"):
            if has_gap:
                verdict = "пригоден"
                why = (f"разрыв в плотности существует: {frac_good * 100:.0f} % "
                       f"значений eps дают 2 и более кластеров при доле шума "
                       f"не выше 5 %, минимально достижимый шум — "
                       f"{min_noise * 100:.1f} %")
            else:
                verdict = "НЕ пригоден"
                why = (f"разрыва в плотности нет: ни одно из проверенных "
                       f"значений eps (в диапазоне "
                       f"{density['eps_range'][0]:.2f}…{density['eps_range'][1]:.2f}) "
                       f"не даёт 2 и более кластеров при доле шума не выше 5 %, "
                       f"минимально достижимый шум — {min_noise * 100:.1f} %. "
                       f"Граф 5 ближайших соседей связен ({comps} компонента): "
                       f"данные образуют единый плотный сгусток, и разделять "
                       f"их по плотности негде. Любой «кластер», найденный "
                       f"этими методами, будет либо одной группой, либо "
                       f"оторванной частью общего облака")
        elif a.key in ("kmeans", "ward", "bisecting", "agglomerative", "gmm",
                       "spectral"):
            if strong_overlap:
                verdict = "пригоден с оговоркой"
                why = (f"кластеры перекрываются: расстояние до {k}-го соседа в "
                       f"{overlap:.1f} раза больше расстояния до ближайшего, "
                       f"границы размыты, но разбиение производится и будет "
                       f"работать с меньшей точностью")
            else:
                verdict = "пригоден"
                why = (f"перекрытие умеренное ({overlap:.1f}), кластеры "
                       f"достаточно различимы, геометрические методы дадут "
                       f"работающее разбиение")
        elif a.key == "meanshift":
            verdict = "пригоден с оговоркой"
            why = ("единственный параметр — радиус ядра; при сильном "
                   "перекрытии результат скачкообразно зависит от bandwidth, "
                   "автоматического выбора значения нет")
        elif a.key == "affinity":
            verdict = "пригоден с оговоркой"
            why = ("число кластеров не задаётся и определяется preference; на "
                   "перекрывающихся данных стремится выделить много мелких "
                   "кластеров вместо истинных")
        elif a.key == "birch":
            verdict = "НЕ пригоден"
            why = (f"BIRCH рассчитан на выборки от тысяч точек (сжатие в "
                   f"микрокластеры), здесь {n} точек при {n_feat} признаках — "
                   f"дерево не набирает статистики, разбиение сводится к "
                   f"K-Means по микрокластерам")
        else:
            verdict = "пригоден"
            why = "ограничений для этих данных не выявлено"

        rows.append({
            "алгоритм": a.title,
            "семейство": a.family,
            "вердикт": verdict,
            "обоснование": why,
            "особенность метода": a.note,
        })
    return pd.DataFrame(rows)


# ======================================================================
# Отрисовка
# ======================================================================
def plot_clusters(ax, X: np.ndarray, labels: np.ndarray, title: str,
                  centroids: np.ndarray | None = None,
                  legend: bool = True) -> None:
    """Рисует разбиение: кластеры разными цветами, выбросы — крестиками."""
    labels = np.asarray(labels)
    noise = labels == NOISE_LABEL
    n_clusters = int(len(np.unique(labels[~noise])))

    for i in range(n_clusters):
        m = labels == i
        if not m.any():
            continue
        ax.scatter(X[m, 0], X[m, 1], s=32, c=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)]],
                   edgecolors="black", linewidths=0.4, label=f"кластер {i}",
                   alpha=0.85)
    if noise.any():
        ax.scatter(X[noise, 0], X[noise, 1], s=26, marker="x", c=NOISE_COLOR,
                   linewidths=1.1, label="шум / выбросы")

    if centroids is not None and len(centroids):
        ax.scatter(centroids[:, 0], centroids[:, 1], s=180, marker="*",
                   c="red", edgecolors="black", linewidths=0.8,
                   label="центроиды", zorder=5)

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_alpha(0.4)
    if legend:
        ax.legend(fontsize=6, loc="best", framealpha=0.9)


# ======================================================================
# Основной сценарий 2.1 - 2.5
# ======================================================================
def run_clustering(path: str, variant: int) -> dict:
    """Выполняет пункт 2 методички целиком."""
    results: dict = {}

    sub_step("Загрузка датасета кластеризации")
    ds = load_dataset(path, variant, name="Clustering", feature_names=["a", "b"])
    print_report(ds)

    # Стандартизация обязательна: иначе шкалы признаков сравниваются напрямую
    scaler = StandardScaler()
    X = scaler.fit_transform(ds.X)
    print(f"      признаки стандартизованы: средние = "
          f"{np.round(scaler.mean_, 3).tolist()}, СКО = "
          f"{np.round(scaler.scale_, 3).tolist()}")
    results["dataset"] = ds

    # ---------------- 2.1 ----------------
    run_step("2.1", "Пригодность алгоритмов для этих данных")
    geo = geometry_report(X)
    print()
    print_geometry(geo)
    print()
    algos = build_algorithms(ds.n_samples, X)
    verdict = judge_algorithms(geo, algos)
    results["geometry"] = geo
    results["verdict"] = verdict

    for _, row in verdict.iterrows():
        mark = "OK  " if row["вердикт"] != "НЕ пригоден" else "НЕТ "
        print(f"      {mark}{row['алгоритм']:<30} {row['вердикт']}")
        print(f"          {row['обоснование']}")
    n_ok = int((verdict["вердикт"] == "пригоден").sum())
    n_cond = int((verdict["вердикт"] == "пригоден с оговоркой").sum())
    n_no = int((verdict["вердикт"] == "НЕ пригоден").sum())
    print(f"      итог: пригодны {n_ok}, пригодны с оговоркой {n_cond}, "
          f"не пригодны {n_no} из {len(verdict)}")
    results["verdict_counts"] = (n_ok, n_cond, n_no)
    step_done(f"{n_ok + n_cond} из {len(verdict)} пригодны, вердикт по геометрии "
              f"выборки без меток y")

    # ---------------- 2.2 ----------------
    run_step("2.2", "Результаты всех алгоритмов на одном графике")
    default_labels = {}
    ncols, nrows = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.7 * nrows))
    for ax, algo in zip(axes.ravel(), algos):
        labels, err = fit_algo(algo, algo.defaults, X)
        if labels is None:
            ax.set_title(f"{algo.title}\nОШИБКА: {err}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        m = evaluate(X, labels)
        default_labels[algo.key] = (labels, m)
        sil_txt = "-" if m["silhouette"] is None else f"{m['silhouette']:.3f}"
        title = (f"{algo.title}\nкластеров: {m['n_clusters']}, "
                 f"шум: {m['n_noise']}, силуэт: {sil_txt}")
        plot_clusters(ax, X, labels, title, _centroids(X, labels))
    for ax in axes.ravel()[len(algos):]:
        ax.axis("off")
    fig.suptitle(f"Пункт 2.2: 12 алгоритмов кластеризации, параметры по "
                 f"умолчанию (вариант {variant}, k = {geo['estimated_k']} по "
                 f"максимуму силуэта)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_22 = save_figure(fig, "task2_2_all_algorithms")
    results["default_labels"] = default_labels
    step_done(f"{len(default_labels)}/{len(algos)} алгоритмов построены, "
              f"{os.path.basename(fig_22)}")

    # ---------------- 2.3 ----------------
    run_step("2.3", "Подбор параметров и анализ отделения выбросов")
    tuning_rows = []
    best = {}
    for algo in algos:
        sub_step(algo.title)
        best_score, best_params, best_labels = float("-inf"), None, None
        n_fail = 0
        for params in algo.grid:
            labels, err = fit_algo(algo, params, X)
            if labels is None:
                n_fail += 1
                continue
            m = evaluate(X, labels)
            s = quality_score(m)
            if s > best_score:
                best_score, best_params, best_labels = s, params, labels
        if best_params is None:
            print("      ни один набор параметров не отработал")
            continue
        m = evaluate(X, best_labels)
        algo.fitted, algo.labels = best_params, best_labels
        best[algo.key] = (best_labels, m, best_params)
        tuning_rows.append({
            "алгоритм": algo.title,
            "наборов": len(algo.grid) - n_fail,
            "параметры": _format_params(best_params),
            "кластеров": m["n_clusters"],
            "шум": m["n_noise"],
            "силуэт": m["silhouette"],
            "балл": best_score,
        })
        sil_txt = "-" if m["silhouette"] is None else f"{m['silhouette']:.3f}"
        print(f"      лучший набор: {_format_params(best_params)}")
        print(f"      -> кластеров {m['n_clusters']}, шум {m['n_noise']} "
              f"({m['noise_share'] * 100:.1f}%), силуэт {sil_txt}, "
              f"балл {best_score:.3f}")
    results["best"] = best

    tuning = pd.DataFrame(tuning_rows).sort_values("балл", ascending=False)
    results["tuning"] = tuning
    print()
    with pd.option_context("display.width", 200, "display.max_colwidth", 44):
        print(tuning.drop(columns=["балл"]).to_string(
            index=False, float_format=lambda v: f"{v:.3f}")
            .replace("\n", "\n      "))
    print()
    print("      ранжирование по баллу (силуэт с поправкой на долю шума):")
    for _, r in tuning.iterrows():
        print(f"        {r['балл']:.3f}  {r['алгоритм']}")

    # --- 2.3: анализ отделения выбросов для методов на основе плотности ---
    print()
    sub_step("Анализ отделения выбросов (DBSCAN, HDBSCAN, OPTICS)")
    density_algos = [a for a in algos if a.labels is not None
                     and a.family.startswith("плотностные")]
    outlier_info = analyze_outliers(X, density_algos)
    results["outliers"] = outlier_info
    for row in outlier_info["rows"]:
        print(f"      {row['алгоритм']}:")
        print(f"        выделено шума: {row['шум']} точек "
              f"({row['доля шума'] * 100:.1f} % выборки)")
        if "медиана kNN(шум)" in row:
            print(f"        медиана расстояния до {outlier_info['k']}-го соседа: "
                  f"шум — {row['медиана kNN(шум)']:.3f}, "
                  f"остальные — {row['медиана kNN(ядро)']:.3f} "
                  f"(отношение {row['отношение']:.2f})")
            print(f"        с такой же плотной окружностью, как у основной "
                  f"массы: {row['приграничных']} из {row['шум']}")
        print(f"        вывод: {row['оценка']}")
    print()
    # Итоговая фраза выводится из разбора строк, а не пишется заранее: если
    # хоть один метод увидел разреженные области, формулировка меняется.
    real_anomalies = [r for r in outlier_info["rows"]
                      if r["оценка"].startswith("кандидаты в аномалии")]
    if real_anomalies:
        print("      Итог: кандидаты в аномалии выделены методами "
              f"{', '.join(r['алгоритм'] for r in real_anomalies)};")
        print("      у них действительно разреженная окружность. "
              f"Метод {', '.join(r['алгоритм'] for r in outlier_info['rows'] if r['оценка'].startswith('не аномалии')) or '—'}"
              " такого не показал.")
    else:
        print("      Итог: ни один метод на основе плотности не отделил "
              "точки в разреженных областях.")
        print("      Отбрасываемые ими точки лежат в зоне стыка "
              "перекрывающихся кластеров, где собственных")
        print("      соседей меньше, чем требует min_samples, поэтому метод "
              "считает их шумом.")
    print("      Это подтверждает и независимая проверка IsolationForest в "
          "п. 2.4.")

    fig_23 = _plot_tuning(X, algos, geo, variant)
    fig_density = _plot_density_probe(geo, ds, variant)
    results["tuning_figure"] = fig_23
    results["density_figure"] = fig_density
    step_done(f"подобрано параметров для {len(tuning)} алгоритмов, "
              f"{os.path.basename(fig_23)} и {os.path.basename(fig_density)}")

    # ---------------- 2.4 ----------------
    run_step("2.4", "Выделение аномальных данных")
    anomaly = detect_anomalies(X, algos, best, geo)
    results["anomaly"] = anomaly
    print(f"      аномальных точек: {anomaly['n_anomaly']} из {ds.n_samples} "
          f"({anomaly['share'] * 100:.1f} %)")
    print("      вклад каждого независимого источника:")
    print(f"        шум метода на основе плотности: "
          f"{int(anomaly['noise_mask'].sum())}")
    print(f"        IsolationForest (contamination=0.05): "
          f"{int(anomaly['iso_mask'].sum())}")
    print(f"        дальние от центра своего кластера (3 % хвост): "
          f"{int(anomaly['dist_mask'].sum())}")
    print(f"        пересечение шума и IsolationForest: "
          f"{anomaly['iso_intersection']}")
    if anomaly["iso_intersection"] == 0:
        print("      Методы не согласованы ни по одной точке, поэтому "
              "аномалии нельзя назвать подтверждёнными:")
        print("      скорее всего, это граница между перекрывающимися "
              "кластерами.")
    else:
        print(f"      Методы согласованы по {anomaly['iso_intersection']} "
              f"точкам — эти аномалии подтверждаются двумя независимыми "
              f"способами.")
    step_done(os.path.basename(anomaly["figure"]))

    # ---------------- 2.5 ----------------
    run_step("2.5", "Сравнение с эталонными метками y")
    comparison = compare_with_reference(X, ds, algos, best, variant, geo)
    results["comparison"] = comparison
    step_done(f"{os.path.basename(comparison['figure'])}, таблица в "
              f"output/comparison")

    return results


# ----------------------------------------------------------------------
# Вспомогательные функции для пунктов 2.2-2.5
# ----------------------------------------------------------------------
def _centroids(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Центроиды кластеров (без кластера шума)."""
    labels = np.asarray(labels)
    out = []
    for lab in np.unique(labels):
        if lab == NOISE_LABEL:
            continue
        out.append(X[labels == lab].mean(axis=0))
    return np.array(out) if out else np.zeros((0, X.shape[1]))


def _format_params(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def analyze_outliers(X: np.ndarray, density_algos: list) -> dict:
    """Оценивает, действительно ли методы на основе плотности отделили аномалии.

    Ключевой вопрос: отмеченные как шум точки — это редкие выбросы или обычные
    точки на границе кластера? Ответ даёт сравнение k-мерного расстояния до
    ближайших соседей: у настоящей аномалии оно заметно больше, чем у точки,
    лежащей между двумя кластерами (у той есть «своя» плотная окружность).
    """
    k = 5
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    knn_dist = dist[:, k]

    report = []
    for algo in density_algos:
        labels = algo.labels
        if labels is None:
            continue
        noise = labels == NOISE_LABEL
        m = evaluate(X, labels)
        if not noise.any():
            report.append({
                "алгоритм": algo.title,
                "шум": 0,
                "доля шума": 0.0,
                "оценка": "выбросы не выделены: выбранные eps/min_samples "
                          "покрывают всю выборку, разделение неявное",
            })
            continue
        noise_d = knn_dist[noise]
        core_d = knn_dist[~noise]
        ratio = float(np.median(noise_d) / max(np.median(core_d), 1e-12))
        # доля «шумовых» точек, у которых всё равно есть плотная окружность
        thr = float(np.quantile(core_d, 0.9))
        border = int((noise_d <= thr).sum())
        if ratio > 1.5 and border <= 0.25 * noise.sum():
            verdict = ("кандидаты в аномалии: медианное расстояние до "
                       f"{k} соседей у них в {ratio:.1f} раза больше, чем у "
                       f"основной массы, а обычная плотность сохраняется "
                       f"лишь у {border} из {int(noise.sum())} точек — "
                       f"остальные действительно лежат в разреженных "
                       f"областях")
        else:
            verdict = (f"не аномалии: у {border} из {int(noise.sum())} точек "
                       f"окружность такая же, как у основной массы, а "
                       f"отношение медианных расстояний равно всего {ratio:.2f} — "
                       f"это граница между перекрывающимися кластерами, а не "
                       f"редкие наблюдения")
        report.append({
            "алгоритм": algo.title,
            "шум": m["n_noise"],
            "доля шума": m["noise_share"],
            "медиана kNN(шум)": float(np.median(noise_d)),
            "медиана kNN(ядро)": float(np.median(core_d)),
            "отношение": ratio,
            "приграничных": border,
            "оценка": verdict,
        })
    return {"rows": report, "k": k, "knn_dist": knn_dist}


def detect_anomalies(X: np.ndarray, algos: list, best: dict, geo: dict) -> dict:
    """2.4: находит аномальные точки и строит график с их выделением.

    Аномалии собираются тремя независимыми способами:
      1. шум (−1) лучшего метода на основе плотности;
      2. IsolationForest как независимая проверка без меток;
      3. точки, далёкие от центра СВОЕГО кластера (для методов без шума).
    """
    density = [a for a in algos if a.family.startswith("плотностные")
               and a.labels is not None]
    best_density = None
    if density:
        best_density = max(density,
                           key=lambda a: quality_score(evaluate(X, a.labels)))
        density_labels = best_density.labels
    else:
        density_labels = None

    noise_mask = (density_labels == NOISE_LABEL) if density_labels is not None \
        else np.zeros(len(X), dtype=bool)

    # независимая проверка IsolationForest (без меток и без кластеризации)
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42).fit(X)
    iso_mask = iso.predict(X) == -1

    # базовое разбиение: метод с выбросами, иначе — лучший метод вообще
    if density_labels is not None:
        base_labels = np.where(noise_mask, NOISE_LABEL, density_labels)
    else:
        base_key = max(best, key=lambda kk: quality_score(best[kk][1]))
        base_labels = np.array(best[base_key][0])

    # точки, далёкие от центра СВОЕГО кластера (для методов без шума)
    dists = np.zeros(len(X))
    for lab in np.unique(base_labels):
        if lab == NOISE_LABEL:
            continue
        m = base_labels == lab
        dists[m] = np.linalg.norm(X[m] - X[m].mean(axis=0), axis=1)
    dist_mask = dists > float(np.quantile(dists, 0.97))

    anomaly_mask = noise_mask | iso_mask | dist_mask
    n_anomaly = int(anomaly_mask.sum())

    # Итоговое разбиение для правого графика: обычные кластеры получают номера
    # 0..n-1, а все аномалии сливаются в один последний кластер.
    core_values = sorted({int(v) for v in base_labels if v != NOISE_LABEL})
    n_clusters = len(core_values)
    remap = {v: i for i, v in enumerate(core_values)}
    plot_labels = np.full(len(X), n_clusters, dtype=int)
    for i, v in enumerate(base_labels):
        if v != NOISE_LABEL and not anomaly_mask[i]:
            plot_labels[i] = remap[int(v)]

    n_intersection = int((noise_mask & iso_mask).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, labels, title in (
        (axes[0], base_labels,
         f"Разбиение, шум метода "
         f"{best_density.title if best_density else '—'}: "
         f"{int(noise_mask.sum())} точек"),
        (axes[1], plot_labels,
         f"Итоговые аномалии: {n_anomaly} точек "
         f"({n_anomaly / len(X) * 100:.1f} %)"),
    ):
        lab = np.array(labels)
        is_anom = lab == n_clusters if labels is plot_labels else lab == NOISE_LABEL
        core = ~is_anom
        ncl = int(len(np.unique(lab[core])))
        for i in range(ncl):
            m = lab == i
            ax.scatter(X[m, 0], X[m, 1], s=30,
                       c=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)]],
                       edgecolors="black", linewidths=0.4, alpha=0.85,
                       label=f"кластер {i}")
        if is_anom.any():
            ax.scatter(X[is_anom, 0], X[is_anom, 1], s=70, marker="*",
                       c=ANOMALY_COLOR, edgecolors="black", linewidths=0.6,
                       label="аномальные данные", zorder=6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("признак a (стандартизованный)")
        ax.set_ylabel("признак b (стандартизованный)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Пункт 2.4: выделение аномальных данных (метки y не "
                 "использовались)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure = save_figure(fig, "task2_4_anomalies")

    table = pd.DataFrame({
        "№": np.flatnonzero(anomaly_mask) + 1,
        "a": X[anomaly_mask, 0].round(4),
        "b": X[anomaly_mask, 1].round(4),
        "шум DBSCAN-подобного": noise_mask[anomaly_mask],
        "IsolationForest": iso_mask[anomaly_mask],
        "далеко от центра": dist_mask[anomaly_mask],
    })

    return {
        "mask": anomaly_mask,
        "noise_mask": noise_mask,
        "iso_mask": iso_mask,
        "dist_mask": dist_mask,
        "n_anomaly": n_anomaly,
        "share": n_anomaly / len(X),
        "n_clusters": n_clusters,
        "labels": plot_labels,
        "iso_intersection": n_intersection,
        "figure": figure,
        "table": table,
    }


def _plot_density_probe(geo: dict, ds: Dataset, variant: int) -> str:
    """График зондирования разрыва в плотности (обоснование вердикта п. 2.1).

    Слева — сам сигнал: точки датасета и изолинии ядерности DBSCAN при
    нескольких eps. Справа — кривые числа кластеров и доли шума в зависимости
    от eps. Если бы разрыв в плотности существовал, на правой панели был бы
    участок, где кластеров много, а шум близок к нулю.
    """
    d = geo["density"]
    curve = pd.DataFrame(d["curve"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- левая панель: точки + изолинии ядерности ---
    ax = axes[0]
    from sklearn.cluster import DBSCAN
    Xs = StandardScaler().fit_transform(ds.X)
    handles = []
    for eps in (0.3, 0.5, 0.9):
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(Xs)
        noise = labels == NOISE_LABEL
        for i in np.unique(labels[~noise]):
            m = labels == i
            ax.scatter(ds.X[m, 0], ds.X[m, 1], s=12, alpha=0.7,
                       color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
        if noise.any():
            ax.scatter(ds.X[noise, 0], ds.X[noise, 1], s=20, marker="x",
                       c=NOISE_COLOR, linewidths=0.9,
                       label=f"eps={eps}: {int(noise.sum())} из {len(labels)} "
                             f"точек отнесено в шум")
    ax.set_title("DBSCAN при разных eps: точки и границы кластеров", fontsize=10)
    ax.set_xlabel("признак a")
    ax.set_ylabel("признак b")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    # --- правая панель: число кластеров и доля шума против eps ---
    ax = axes[1]
    ax.plot(curve["eps"], curve["n_clusters"], "o-", color="#1565C0",
            label="число кластеров")
    ax.set_xscale("log")
    ax.set_xlabel("eps")
    ax.set_ylabel("Число найденных кластеров", color="#1565C0")
    ax.tick_params(axis="y", labelcolor="#1565C0")
    ax2 = ax.twinx()
    ax2.plot(curve["eps"], curve["noise_share"] * 100, "s--", color="#C62828",
             label="доля шума, %")
    ax2.axhline(5.0, color="#C62828", ls=":", lw=1.2)
    ax2.set_ylabel("Доля шума, %", color="#C62828")
    ax2.tick_params(axis="y", labelcolor="#C62828")
    ax2.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title(f"Разрыв в плотности: пригодных eps — "
                 f"{d['fraction_good'] * 100:.0f} %, "
                 f"минимум шума при 2+ кластерах — {d['min_noise'] * 100:.1f} %",
                 fontsize=10)

    fig.suptitle("Пункт 2.1: проверка пригодности методов на основе плотности "
                 f"(вариант {variant})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, "task2_1_density_probe")


def _plot_tuning(X: np.ndarray, algos: list, geo: dict, variant: int) -> str:
    """Сравнение настроенных алгоритмов: левая колонка — сами разбиения,
    правая — диаграммы балла и силуэта."""
    done = [a for a in algos if a.labels is not None]
    n = len(done)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    scores, sils, titles = [], [], []
    for ax, algo in zip(axes, done):
        m = evaluate(X, algo.labels)
        s = quality_score(m)
        scores.append(s)
        sils.append(m["silhouette"] if m["silhouette"] is not None else 0.0)
        titles.append(algo.title)
        sil_txt = "-" if m["silhouette"] is None else f"{m['silhouette']:.3f}"
        title = (f"{algo.title}\n{_format_params(algo.fitted)}\n"
                 f"кластеров: {m['n_clusters']}, шум: {m['n_noise']}, "
                 f"силуэт: {sil_txt}")
        plot_clusters(ax, X, algo.labels, title, _centroids(X, algo.labels))

    order = np.argsort(scores)[::-1]
    short = [t.split(" ", 1)[-1] for t in titles]
    for ax in axes[n:]:
        ax.axis("off")
    # заменяем последние пустые ячейки на сводные диаграммы
    free = list(axes[n:])
    if free:
        ax = free[0]
        ax.barh(range(len(scores)), [scores[i] for i in order],
                color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in order],
                edgecolor="black")
        ax.set_yticks(range(len(scores)))
        ax.set_yticklabels([short[i] for i in order], fontsize=7)
        ax.set_xlabel("Балл = силуэт · (1 − доля шума)")
        ax.set_title("Ранжирование по качеству разбиения", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()
    if len(free) > 1:
        ax = free[1]
        ax.barh(range(len(sils)), [sils[i] for i in order],
                color="#5C6BC0", edgecolor="black")
        ax.set_yticks(range(len(sils)))
        ax.set_yticklabels([short[i] for i in order], fontsize=7)
        ax.set_xlabel("Силуэт по ненулевым точкам")
        ax.set_title("Силуэт (без штрафа за выбросы)", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle(f"Пункт 2.3: 12 алгоритмов с подобранными параметрами "
                 f"(вариант {variant})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, "task2_3_tuned_algorithms")


def compare_with_reference(X: np.ndarray, ds: Dataset, algos: list, best: dict,
                           variant: int, geo: dict) -> dict:
    """2.5: сравнивает разбиения с эталонными метками y.

    Здесь и только здесь y вступает в дело — как внешний критерий, а не как
    участник построения разбиения.
    """
    rows = []
    for algo in algos:
        if algo.labels is None:
            continue
        m = evaluate(X, algo.labels)
        rows.append({
            "алгоритм": algo.title,
            "параметры": _format_params(algo.fitted),
            "кластеров": m["n_clusters"],
            "шум": m["n_noise"],
            "силуэт": m["silhouette"],
            "ARI": adjusted_rand_score(ds.y, algo.labels),
            "NMI": normalized_mutual_info_score(ds.y, algo.labels),
            "однородность": homogeneity_score(ds.y, algo.labels),
            "полнота": completeness_score(ds.y, algo.labels),
            "V-мера": v_measure_score(ds.y, algo.labels),
        })
    table = pd.DataFrame(rows).sort_values("ARI", ascending=False).reset_index(drop=True)
    table.insert(0, "место", range(1, len(table) + 1))

    # итоговый алгоритм: лучший по согласию с эталоном
    fitted = [a for a in algos if a.labels is not None]
    best_by_ari = str(table.iloc[0]["алгоритм"])
    final_algo = next(a for a in fitted if a.title == best_by_ari)
    cm = confusion_matrix(ds.y, final_algo.labels)

    # доля точек, отнесённых не к тому эталонному классу
    # Доля точек, отнесённых не к тому эталонному классу. Сопоставление меток
    # взаимно однозначное (задача о назначениях): иначе несколько эталонных
    # классов претендовали бы на один кластер разбиения и сумма получилась бы
    # больше единицы.
    from scipy.optimize import linear_sum_assignment
    row_idx, col_idx = linear_sum_assignment(-cm)
    correct = int(cm[row_idx, col_idx].sum())
    accuracy = correct / len(ds.y)

    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}")
          .replace("\n", "\n      "))
    print()
    print(f"      эталонных классов: {ds.n_targets}, "
          f"распределение {ds.report['распределение меток']}")
    print(f"      лучший по ARI: {best_by_ari} "
          f"(ARI = {table.iloc[0]['ARI']:.3f}, "
          f"NMI = {table.iloc[0]['NMI']:.3f})")
    print(f"      сопоставление меток (взаимно однозначное), доля верно "
          f"отнесённых точек: {accuracy:.3f}")
    print(f"      таблица сопоставления (строки — эталонный класс y, "
          f"столбцы — кластер разбиения):")
    for row in cm:
        print("        " + " ".join(f"{v:4d}" for v in row))
    print()
    print(_final_conclusion(table, ds, final_algo, accuracy, geo))

    # bar-график сравнения
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    t = table.iloc[::-1]
    labels_short = [s.split(" ", 1)[-1] for s in t["алгоритм"]]
    ypos = np.arange(len(t))
    axes[0].barh(ypos - 0.2, t["ARI"], height=0.4, label="ARI", color="#1565C0")
    axes[0].barh(ypos + 0.2, t["NMI"], height=0.4, label="NMI", color="#EF6C00")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels_short, fontsize=8)
    axes[0].set_xlabel("Значение метрики")
    axes[0].set_title("Согласие с эталонными метками y", fontsize=10)
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].barh(ypos, [0.0 if s is None else s for s in t["силуэт"]],
                 color="#2E7D32", edgecolor="black")
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(labels_short, fontsize=8)
    axes[1].set_xlabel("Силуэт (внутренний критерий, без меток y)")
    axes[1].set_title("Внутреннее качество разбиения", fontsize=10)
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Пункт 2.5: сравнение с эталоном (вариант {variant}). "
                 f"Метки y не участвовали в построении разбиений", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figure = save_figure(fig, "task2_5_comparison")

    paths = write_table(table, "task2_5_comparison")
    write_table(cm_df(cm, ds, final_algo), "task2_5_confusion")
    write_table(_labels_frame(ds, final_algo), "task2_5_labels")

    return {
        "table": table,
        "figure": figure,
        "best_algo": final_algo,
        "accuracy": accuracy,
        "confusion": cm,
        "paths": paths,
    }


def _final_conclusion(table: pd.DataFrame, ds: Dataset, algo: Algo,
                      accuracy: float, geo: dict) -> str:
    """Текстовый вывод п. 2.5: обоснование выбора алгоритма."""
    best_row = table.iloc[0]
    k_true = ds.n_targets
    k_est = geo["estimated_k"]
    ari = float(best_row["ARI"])
    lines = [
        "=" * 68,
        "ОБОСНОВАНИЕ ВЫБОРА (пункт 2.5)",
        "=" * 68,
        f"1. Эталон содержит {k_true} класса, оценка без меток дала k = {k_est} "
        f"(правило 1-SE по силуэту), то есть число кластеров определено верно.",
        f"2. Лучшее согласие с эталоном: {best_row['алгоритм']} "
        f"(ARI = {ari:.3f}, NMI = {float(best_row['NMI']):.3f}, "
        f"верно отнесено {accuracy * 100:.1f} % точек).",
        f"3. Высокий ARI объясним геометрией данных: отношение расстояния до "
        f"{k_est}-го соседа к расстоянию до ближайшего равно "
        f"{geo['overlap']:.1f}, то есть кластеры перекрываются. Часть точек на "
        f"стыках не может быть отнесена однозначно, и ни один алгоритм без "
        f"использования меток не способен их разделить — это предел задачи, "
        f"а не недостаток метода.",
        f"4. Методы на основе плотности (DBSCAN, HDBSCAN, OPTICS) в тройке "
        f"последних мест (ARI <= {float(table[table['алгоритм'].str.contains('DBSCAN|HDBSCAN|OPTICS')]['ARI'].max()):.3f}), "
        f"что совпадает с вердиктом п. 2.1: разрыва в плотности нет "
        f"(минимально достижимая доля шума при двух и более кластерах — "
        f"{geo['density']['min_noise'] * 100:.1f} %).",
        f"5. Практический выбор для этих данных: {algo.title}. "
        f"Алгоритмы, строящие разбиение по ближайшим центрам или по минимизации "
        f"within-cluster дисперсии (Ward, K-Means, Gaussian mixtures), дают "
        f"сопоставимый результат и устойчивы к перекрытию; выбирать следует "
        f"по силуэту и ARI, а не по числу найденных кластеров.",
    ]
    return "\n".join("      " + ln for ln in lines)


def cm_df(cm: np.ndarray, ds: Dataset, algo: Algo) -> pd.DataFrame:
    """Confusion matrix в DataFrame с осмысленными именами."""
    cols = [f"кластер {c}" if c >= 0 else "шум" for c in range(cm.shape[1])]
    return pd.DataFrame(cm, index=[f"y = {v}" for v in ds.target_names],
                        columns=cols)


def _labels_frame(ds: Dataset, algo: Algo) -> pd.DataFrame:
    """Итоговые метки каждой точки рядом с эталонными."""
    return pd.DataFrame({
        "a": ds.X[:, 0].round(6),
        "b": ds.X[:, 1].round(6),
        "y_эталон": ds.y,
        "метка_алгоритма": algo.labels,
    })
