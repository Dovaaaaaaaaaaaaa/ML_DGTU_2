"""Пункт 1 методички (1.1-1.5): отбор признаков через PCA, SVD и ICA.

Датасет Lab2_Dataset_PCA.xlsx: 5 признаков + столбец эталонных меток y.
Метки y не участвуют в анализе (см. data_loader) и используются только для
раскраски точек на графиках.

Структура пункта:
    1.1  матрица взаимных корреляций
    1.2  матрица разброса (pd.plotting.scatter_matrix)
    1.3  оценка значимости: PCA, SVD, ICA
    1.4  деление компонент на главные / второстепенные / шумовые + графики
    1.5  проекции на 2 главные компоненты с направлениями (биплоты)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

from src.config import run_step, step_done, sub_step
from src.data_loader import Dataset, load_dataset, print_report
from src.io_utils import save_figure

# ----------------------------------------------------------------------
# Пороги классификации компонент. Зафиксированы явно, потому что критерий
# «на глаз» в отчёте не воспроизводится.
#
# Для PCA/SVD объяснённая дисперсия нормирована на сумму (сумма = 1):
#   >= MAIN_EVR      — компонента несёт существенную долю дисперсии;
#   >= NOISE_EVR     — компонента не вырождена, её можно интерпретировать;
#   <  NOISE_EVR     — вырожденная (нулевая) компонента = шум.
# Для ICA score — модуль эксцесса: у гауссовой компоненты он равен нулю,
# поэтому чем он дальше от 0, тем компонента «информативнее».
# ----------------------------------------------------------------------
MAIN_EVR = 0.15
NOISE_EVR = 0.01
MAIN_KURT = 1.0
SECONDARY_KURT = 0.3

# Цвета уровней значимости на графиках
COLOR_MAIN = "#2E7D32"
COLOR_SECONDARY = "#F9A825"
COLOR_NOISE = "#C62828"
LEVEL_COLORS = {"главная": COLOR_MAIN,
                "второстепенная": COLOR_SECONDARY,
                "шумовая": COLOR_NOISE}


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def numeric_rank(X: np.ndarray, tol: float = 1e-10) -> int:
    """Числовой ранг матрицы по сингулярным значениям.

    Нужен, чтобы не обучать ICA на вырожденном направлении: FastICA делит на
    сингулярные значения при отбеливании, и на почти нулевом значении
    (здесь s_min = 4e-15) матрица смешивания схлопывается в нули, а
    восстановление данных перестаёт работать.
    """
    if X.size == 0:
        return 0
    s = np.linalg.svd(X, compute_uv=False)
    if s[0] == 0:
        return 0
    return int(np.sum(s > s[0] * tol))


def marchenko_pastur_edge(n: int, p: int) -> tuple[float, float]:
    """Границы распределения Марченко-Пастура для шумовых сингулярных значений.

    Для матрицы корреляций (след = p) шумовой спектр при n >> p сосредоточен
    в [(1-sqrt(q))^2, (1+sqrt(q))^2], где q = p/n. Собственные значения выше
    верхней границы статистически значимы, ниже — неотличимы от шума.
    """
    q = p / n
    return (1.0 - np.sqrt(q)) ** 2, (1.0 + np.sqrt(q)) ** 2


def classify_by_variance(ratios: np.ndarray) -> list[str]:
    """Размечает компоненты по доле объяснённой дисперсии."""
    out = []
    for r in ratios:
        if r >= MAIN_EVR:
            out.append("главная")
        elif r >= NOISE_EVR:
            out.append("второстепенная")
        else:
            out.append("шумовая")
    return out


def classify_by_kurtosis(kurt: np.ndarray) -> list[str]:
    """Размечает ICA-компоненты по модулю эксцесса (несгуассовость)."""
    out = []
    for k in np.abs(kurt):
        if k >= MAIN_KURT:
            out.append("главная")
        elif k >= SECONDARY_KURT:
            out.append("второстепенная")
        else:
            out.append("шумовая")
    return out


def biplot(scores: np.ndarray, loadings: np.ndarray, ax=None,
           labels: list | None = None, targets: np.ndarray | None = None,
           target_labels: list | None = None,
           xlabel: str = "PC1", ylabel: str = "PC2",
           title: str = "Biplot") -> None:
    """Совмещает проекции точек (scores) и направления признаков (loadings).

    Близко к коду из методички. Оси scores нормируются в [0, 1], а нагрузки
    лежат в [-1, 1], поэтому без масштабирования стрелки не читались бы.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    scalex = 1.0 / (scores[:, 0].max() - scores[:, 0].min())
    scaley = 1.0 / (scores[:, 1].max() - scores[:, 1].min())

    if targets is not None:
        scatter = ax.scatter(scores[:, 0] * scalex, scores[:, 1] * scaley,
                             c=targets, cmap="viridis", alpha=0.7,
                             edgecolors="k", linewidths=0.3, s=25)
        handles, _ = scatter.legend_elements()
        ax.legend(handles, target_labels or [], title="Classes",
                  loc="upper right", fontsize=8)
    else:
        ax.scatter(scores[:, 0] * scalex, scores[:, 1] * scaley, alpha=0.7)

    for i in range(loadings.shape[0]):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], color="r",
                 alpha=0.7, head_width=0.03, head_length=0.03, linewidth=1.5)
        if labels is not None:
            ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, labels[i],
                    color="darkred", ha="center", va="center", fontweight="bold")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)


def _annotate_levels(ax, levels: list[str], label: str) -> None:
    """Подкрашивает подписи оси X уровнями значимости."""
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{i + 1}\n{lev}" for i, lev in enumerate(levels)],
                       fontsize=7)
    ax.set_xlabel(label)


# ----------------------------------------------------------------------
# 1.1 - 1.5
# ----------------------------------------------------------------------
def run_pca_analysis(path: str, variant: int) -> dict:
    """Выполняет пункт 1 методички целиком. Возвращает словарь с результатами."""
    results: dict = {}

    # ---------------- Загрузка ----------------
    sub_step("Загрузка датасета PCA")
    ds = load_dataset(path, variant, name="PCA",
                      feature_names=["a", "b", "c", "d", "e"])
    print_report(ds)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(ds.X)
    rank = numeric_rank(Xs)
    n, p = Xs.shape
    results["rank"] = rank
    print(f"      признаки стандартизованы (среднее 0, СКО 1), "
          f"числовой ранг матрицы = {rank} из {p}")
    print(f"      => {p - rank} из {p} направлений не несут информации "
          f"(сингулярное значение ~0)")

    # ================= 1.1. Матрица корреляций =================
    run_step("1.1", "Матрица взаимных корреляций")
    corr = pd.DataFrame(ds.X, columns=ds.feature_names).corr()
    # максимальная по модулю корреляция не на главной диагонали
    mask = np.eye(len(corr), dtype=bool)
    off_diag = corr.to_numpy(dtype=float).copy()
    off_diag[mask] = np.nan
    max_r = float(np.nanmax(np.abs(off_diag)))
    ij = np.unravel_index(np.nanargmax(np.abs(off_diag)), off_diag.shape)
    pair = f"{ds.feature_names[ij[0]]}~{ds.feature_names[ij[1]]}"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr.iloc[i, j]) > 0.55 else "black")
    ax.set_title(f"Матрица корреляций Пирсона (вариант {variant})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    fig.tight_layout()
    path_fig = save_figure(fig, "task1_1_correlation")
    results["corr"] = corr

    print()
    print(corr.round(3).to_string().replace("\n", "\n      "))
    print(f"      максимальная по модулю корреляция: {max_r:.3f} между "
          f"{pair}")
    if max_r > 0.95:
        print(f"      => признаки {pair} практически линейно зависимы "
              f"(|r| > 0.95), один из них избыточен")
    else:
        print(f"      => линейно зависимых пар нет (|r| < 0.95)")
    step_done(f"{corr.shape[0]}x{corr.shape[1]}, max|r| = {max_r:.3f} ({pair}), "
              f"{os.path.basename(path_fig)}")

    # ================= 1.2. Матрица разброса =================
    run_step("1.2", "Матрица разброса (pd.plotting.scatter_matrix)")
    frame = pd.DataFrame(ds.X, columns=ds.feature_names)
    # Именно pd.plotting.scatter_matrix, как требует методичка. Точки
    # раскрашены по эталонным меткам y — это единственное применение y.
    # Свою фигуру pandas создаёт сам: передавать ax нельзя, он очищается.
    axes_sm = pd.plotting.scatter_matrix(
        frame, c=ds.y, diagonal="hist",
        figsize=(2.7 * p, 2.7 * p), marker="o", s=6, alpha=0.6)
    fig = axes_sm[0][0].get_figure()
    handles = [plt.Line2D([], [], marker="o", ls="", color="gray",
                          label=f"y = {v}") for v in ds.target_names]
    fig.legend(handles=handles, loc="upper right", fontsize=8, title="Классы y")
    fig.suptitle(f"Матрица разброса, точки раскрашены по y (вариант {variant})",
                 y=0.995)
    path_sm = save_figure(fig, "task1_2_scatter_matrix")
    step_done(f"{p * p} ячеек, раскраска по y, {os.path.basename(path_sm)}")

    # ================= 1.3. Значимость: PCA / SVD / ICA =================
    run_step("1.3", "Оценка значимости и зашумлённости: PCA, SVD, ICA")

    # --- PCA ---
    pca = PCA().fit(Xs)
    evr = pca.explained_variance_ratio_
    ev = pca.explained_variance_
    # log-likelihood по числу компонент: sklearn PCA это считает, но при
    # вырожденном спектре даёт -inf, поэтому используем только k < rank
    loglik = {}
    for k in range(1, rank + 1):
        loglik[k] = float(PCA(n_components=k).fit(Xs).score_samples(Xs).sum())
    results["pca"] = {"evr": evr, "ev": ev, "loglik": loglik}

    # --- SVD ---
    U, sv, Vt = np.linalg.svd(Xs, full_matrices=False)
    sv_ratio = sv ** 2 / (sv ** 2).sum()
    results["svd"] = {"sv": sv, "ratio": sv_ratio, "U": U, "Vt": Vt}

    # проверка эквивалентности SVD и PCA на центрированных данных
    diff_scores = float(np.abs(U[:, :2] * sv[:2] - pca.transform(Xs)[:, :2]).max())
    diff_load = float(np.abs(Vt[:2].T - pca.components_[:2].T).max())
    results["svd_pca_diff"] = (diff_scores, diff_load)

    # --- ICA (только на невырожденных направлениях) ---
    ica = FastICA(n_components=rank, random_state=42, max_iter=2000,
                  tol=1e-6, whiten="unit-variance")
    S = ica.fit_transform(Xs)
    mixing = ica.mixing_                      # (rank, p) — разложение по признакам
    recon_err = float(np.abs(S @ mixing.T + ica.mean_ - Xs).max())
    kurt = pd.DataFrame(S).kurtosis().to_numpy()      # эксцесс Фишера
    # Сортировка по убыванию значимости: для ICA «главная компонента» — это
    # компонента с максимальной несгуассовостью, а не первая по счёту.
    order = np.argsort(-np.abs(kurt))
    S, mixing, kurt = S[:, order], mixing[order], kurt[order]
    results["ica"] = {"sources": S, "mixing": mixing, "kurtosis": kurt,
                      "recon_err": recon_err, "order": order}

    # контрольная попытка обучить ICA на всех p компонентах
    # Предупреждение о сходимости здесь ожидаемо и само по себе является
    # диагностикой: на вырожденном направлении FastICA не сходится.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica_full = FastICA(n_components=p, random_state=42, max_iter=2000, tol=1e-6)
        S_full = ica_full.fit_transform(Xs)
    recon_err_full = float(np.abs(S_full @ ica_full.mixing_.T + ica_full.mean_ - Xs).max())
    results["ica_full"] = {"recon_err": recon_err_full,
                           "mixing_absmax": float(np.abs(ica_full.mixing_).max())}
    print(f"      PCA : доли дисперсии {np.round(evr, 4).tolist()}, "
          f"кумулятивно {np.round(np.cumsum(evr), 4).tolist()}")
    print(f"      SVD : сингулярные значения {np.round(sv, 3).tolist()}, "
          f"доли {np.round(sv_ratio, 4).tolist()}")
    print(f"      ICA : обучен на {rank} компонентах (ранг данных), "
          f"ошибка восстановления = {recon_err:.2e}, "
          f"модули эксцесса {np.round(kurt, 3).tolist()}")
    print(f"            контроль: ICA на всех {p} компонентах даёт ошибку "
          f"восстановления {recon_err_full:.3f} при max|mixing| = "
          f"{results['ica_full']['mixing_absmax']:.2e} — модель вырождена")
    step_done(f"ранг {rank}/{p}, SVD≡PCA с точностью {diff_scores:.1e}, "
              f"ICA восстанавливает данные с {recon_err:.1e}")

    # ================= 1.4. Классификация компонент =================
    run_step("1.4", "Главные / второстепенные / шумовые компоненты")
    mp_low, mp_high = marchenko_pastur_edge(n, p)
    results["mp"] = (mp_low, mp_high)

    # eigenval = доля * p, так как след матрицы корреляций равен p
    eig = evr * p
    levels_pca = classify_by_variance(evr)
    levels_svd = classify_by_variance(sv_ratio)
    levels_ica = classify_by_kurtosis(kurt)

    results["levels"] = {"PCA": levels_pca, "SVD": levels_svd, "ICA": levels_ica}

    # Для вырожденных компонент log-likelihood не определён (-inf из-за
    # нулевого собственного значения), поэтому такие ячейки помечаются текстом,
    # а не None: иначе в таблице остаётся молчаливый NaN.
    ll_text = []
    for i in range(p):
        if (i + 1) in loglik:
            ll_text.append(f"{loglik[i + 1]:.1f}")
        else:
            ll_text.append("-inf (вырождена)")

    table_pca = pd.DataFrame({
        "компонента": [f"PC{i + 1}" for i in range(p)],
        "собств. знач.": eig.round(4),
        "доля дисп.": evr.round(4),
        "кумулятивно": np.cumsum(evr).round(4),
        "log-lik": ll_text,
        "уровень": levels_pca,
    })
    table_svd = pd.DataFrame({
        "компонента": [f"SV{i + 1}" for i in range(p)],
        "сингулярное значение": sv.round(4),
        "доля дисперсии": sv_ratio.round(4),
        "кумулятивно": np.cumsum(sv_ratio).round(4),
        "уровень": levels_svd,
    })
    table_ica = pd.DataFrame({
        "компонента": [f"IC{i + 1}" for i in range(rank)],
        "эксцесс": kurt.round(3),
        "модуль эксцесса": np.abs(kurt).round(3),
        "уровень": levels_ica,
    })
    results["tables"] = {"pca": table_pca, "svd": table_svd, "ica": table_ica}

    sub_step("PCA")
    print(table_pca.to_string(index=False).replace("\n", "\n      "))
    sub_step("SVD")
    print(table_svd.to_string(index=False).replace("\n", "\n      "))
    sub_step(f"ICA (ранг {rank}, сортировка по модулю эксцесса)")
    print(table_ica.to_string(index=False).replace("\n", "\n      "))

    for name, lev in (("PCA", levels_pca), ("SVD", levels_svd), ("ICA", levels_ica)):
        cnt = {lvl: lev.count(lvl) for lvl in ("главная", "второстепенная", "шумовая")
               if lvl in lev}
        print(f"      {name}: {cnt}")

    # --- графики значимости ---
    path_fig = _plot_significance(pca, sv, kurt, evr, sv_ratio, eig,
                                  levels_pca, levels_svd, levels_ica,
                                  mp_high, n, variant)
    results["significance_figure"] = path_fig
    step_done(f"пороги: доля >= {MAIN_EVR} главная, >= {NOISE_EVR} "
              f"второстепенная, < {NOISE_EVR} шумовая; "
              f"граница шума Марченко-Пастура = {mp_high:.3f}")

    # ================= 1.5. Бipartные проекции =================
    run_step("1.5", "Проекции на 2 наиболее значимые компоненты (биплоты)")
    figs = _plot_biplots(ds, pca, U, sv, Vt, S, mixing, evr, results, variant)
    results["biplots"] = figs
    step_done(f"{len(figs)} биплота (PCA, SVD, ICA) по 2 главным компонентам")

    return results


def _plot_significance(pca, sv, kurt, evr, sv_ratio, eig, levels_pca,
                       levels_svd, levels_ica, mp_high, n, variant) -> str:
    """Совмещённый график значимости компонент по трём методам (п. 1.4)."""
    p = len(evr)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # --- PCA: scree + кумулятивная дисперсия ---
    ax = axes[0]
    bars = ax.bar(range(1, p + 1), evr, color=[LEVEL_COLORS[l] for l in levels_pca],
                  edgecolor="black", alpha=0.85, label="доля дисперсии")
    ax.axhline(MAIN_EVR, color=COLOR_MAIN, ls="--", lw=1,
               label=f"порог главной ({MAIN_EVR})")
    ax.axhline(NOISE_EVR, color=COLOR_NOISE, ls=":", lw=1.2,
               label=f"порог шума ({NOISE_EVR})")
    ax.set_yscale("log")
    ax.set_xticks(range(1, p + 1))
    ax.set_xlabel("Номер главной компоненты")
    ax.set_ylabel("Доля объяснённой дисперсии (log)")
    ax.set_title(f"PCA: скри-график (ранг {p - int(np.sum(evr < NOISE_EVR))})")
    ax2 = ax.twinx()
    ax2.plot(range(1, p + 1), np.cumsum(evr), "o-", color="#1565C0", lw=1.5,
             label="кумулятивно")
    ax2.set_ylabel("Кумулятивная дисперсия")
    ax2.set_ylim(0, 1.05)
    _annotate_levels(ax, levels_pca, "Номер компоненты")
    ax.grid(True, axis="y", alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")

    # --- SVD: сингулярные значения + граница Марченко-Пастура ---
    # Граница MP задана для собственных значений матрицы корреляций, а на
    # графике — сингулярные значения. Связь: lambda = s^2 / (n - 1), значит
    # s_MP = sqrt(lambda_max * (n - 1)).
    ax = axes[1]
    ax.bar(range(1, p + 1), sv, color=[LEVEL_COLORS[l] for l in levels_svd],
           edgecolor="black", alpha=0.85)
    ax.set_yscale("log")
    s_mp = float(np.sqrt(mp_high * (n - 1)))
    ax.axhline(s_mp, color="#6A1B9A", ls="--", lw=1.2,
               label=f"верхняя граница шума\n(Марченко-Пастур, "
                     f"$\\lambda_+$={mp_high:.2f} → s={s_mp:.1f})")
    ax.set_xticks(range(1, p + 1))
    ax.set_xlabel("Номер сингулярного значения")
    ax.set_ylabel("Сингулярное значение (log)")
    ax.set_title("SVD: спектр сингулярных значений")
    _annotate_levels(ax, levels_svd, "Номер компоненты")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)

    # --- ICA: модуль эксцесса ---
    ax = axes[2]
    n_ic = len(kurt)
    ax.bar(range(1, n_ic + 1), np.abs(kurt),
           color=[LEVEL_COLORS[l] for l in levels_ica],
           edgecolor="black", alpha=0.85)
    ax.axhline(MAIN_KURT, color=COLOR_MAIN, ls="--", lw=1,
               label=f"порог главной (|эксцесс| >= {MAIN_KURT})")
    ax.axhline(SECONDARY_KURT, color=COLOR_SECONDARY, ls=":", lw=1.2,
               label=f"порог второстепенной (>= {SECONDARY_KURT})")
    ax.set_xticks(range(1, n_ic + 1))
    ax.set_xlabel("Номер независимой компоненты (по убыванию |эксцесса|)")
    ax.set_ylabel("Модуль эксцесса")
    ax.set_title("ICA: несгуассовость компонент\n(score = |эксцесс|)")
    _annotate_levels(ax, levels_ica, "Номер компоненты")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)

    handles = [Patch(facecolor=c, edgecolor="black", label=lvl)
               for lvl, c in LEVEL_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Значимость компонент по трём методам (вариант {variant})")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return save_figure(fig, "task1_4_component_significance")


def _plot_biplots(ds, pca, U, sv, Vt, S, mixing, evr, results, variant) -> list:
    """Три биплота: PCA, SVD, ICA (п. 1.5)."""
    figs = []

    # --- PCA ---
    scores = pca.transform(StandardScaler().fit_transform(ds.X))
    loadings = pca.components_[:2].T
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    biplot(scores[:, :2], loadings, ax=ax, labels=ds.feature_names,
           targets=ds.y, target_labels=ds.target_names,
           xlabel=f"PC1 ({evr[0] * 100:.1f}% дисперсии)",
           ylabel=f"PC2 ({evr[1] * 100:.1f}% дисперсии)",
           title="PCA: проекция на 2 главные компоненты")
    figs.append(save_figure(fig, "task1_5_biplot_pca"))

    # --- SVD ---
    scores_svd = U[:, :2] * sv[:2]
    loadings_svd = Vt[:2].T
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    biplot(scores_svd, loadings_svd, ax=ax, labels=ds.feature_names,
           targets=ds.y, target_labels=ds.target_names,
           xlabel=f"SV1 ({results['svd']['ratio'][0] * 100:.1f}% дисперсии)",
           ylabel=f"SV2 ({results['svd']['ratio'][1] * 100:.1f}% дисперсии)",
           title="SVD: проекция на 2 ведущих сингулярных направления")
    figs.append(save_figure(fig, "task1_5_biplot_svd"))

    # --- ICA ---
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    biplot(S[:, :2], mixing[:2].T, ax=ax, labels=ds.feature_names,
           targets=ds.y, target_labels=ds.target_names,
           xlabel=f"IC1 (|эксцесс| = {abs(results['ica']['kurtosis'][0]):.2f})",
           ylabel=f"IC2 (|эксцесс| = {abs(results['ica']['kurtosis'][1]):.2f})",
           title="ICA: проекция на 2 наиболее несгуассовые компоненты")
    figs.append(save_figure(fig, "task1_5_biplot_ica"))
    return figs
