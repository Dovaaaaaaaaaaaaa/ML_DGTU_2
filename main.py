"""Точка входа лабораторной работы №2.

«Обучение без учителя. Понижение размерности. Обнаружение аномалий.
Кластеризация.»

Запускает оба пункта методички:
    Пункт 1 (1.1-1.5)  отбор признаков: корреляции, матрица разброса,
                      оценка значимости через PCA, SVD и ICA, деление
                      компонент на главные/второстепенные/шумовые, биплоты
    Пункт 2 (2.1-2.5)  кластеризация: 12 алгоритмов sklearn, вердикт о
                      пригодности каждого, подбор параметров, анализ выбросов,
                      выделение аномалий, сравнение с эталонными метками

Запуск:
    python main.py                          # оба пункта, вариант 7
    python main.py --variant 3              # другой вариант
    python main.py --sections pca           # только пункт 1
    python main.py --sections clustering    # только пункт 2
    python main.py --list-sections          # что во что входит
    python main.py --list-variants          # какие варианты есть в датасетах

Структура проекта повторяет лабораторную №1: main.py только связывает
модули из src/, вся логика разделов лежит в src/pca_analysis.py и
src/clustering.py.

Метки y не участвуют в анализе нигде, кроме п. 2.5, где они выступают
внешним критерием оценки. Это требование методички.
"""

import argparse
import importlib
import os
import sys
import time

# src.config первым: он выставляет MPLBACKEND=Agg до первого импорта
# matplotlib, иначе каждое сохранение графика открывает окно GUI.
from src.config import (
    SEED,
    VARIANT,
    PCA_DATASET_PATH,
    CLUSTER_DATASET_PATH,
    OUTPUT_DIR,
    banner,
    force_utf8_stdio,
    set_seeds,
)

# Модули разделов импортируются по требованию: один из них тянет за собой
# sklearn целиком, и для запуска только пункта 1 грузить кластеризацию незачем.
SECTIONS = (
    ("pca", "pca_analysis", "пункт 1 (1.1-1.5), отбор признаков PCA/SVD/ICA"),
    ("clustering", "clustering", "пункт 2 (2.1-2.5), кластеризация и аномалии"),
)
SECTION_MODULES = {key: module for key, module, _ in SECTIONS}


def get_module(name: str):
    """Импортирует модуль раздела при первом обращении к нему.

    Принимает и ключ раздела ("pca"), и имя модуля ("pca_analysis").
    """
    if name in SECTION_MODULES:
        module_name = SECTION_MODULES[name]
    elif name in SECTION_MODULES.values():
        module_name = name
    else:
        raise ValueError(f"Неизвестный раздел: {name}")
    return importlib.import_module(f"src.{module_name}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Лабораторная работа №2: обучение без учителя, "
                    "понижение размерности, кластеризация")
    ap.add_argument("--variant", type=int, default=VARIANT,
                    help=f"номер варианта в датасетах (по умолчанию {VARIANT})")
    ap.add_argument("--pca-data", default=PCA_DATASET_PATH,
                    help="путь к датасету для пункта 1")
    ap.add_argument("--cluster-data", default=CLUSTER_DATASET_PATH,
                    help="путь к датасету для пункта 2")
    ap.add_argument("--sections", nargs="*", default=None,
                    help="разделы для выполнения: pca clustering "
                         "(по умолчанию оба)")
    ap.add_argument("--list-sections", action="store_true",
                    help="показать список разделов и выйти")
    ap.add_argument("--list-variants", action="store_true",
                    help="показать доступные варианты в датасетах и выйти")
    args = ap.parse_args()

    if args.list_sections:
        for key, _, desc in SECTIONS:
            print(f"  {key:<12} {desc}")
        return 0

    if args.list_variants:
        _list_variants(args.pca_data, args.cluster_data)
        return 0

    if args.sections:
        unknown = [s for s in args.sections if s not in SECTION_MODULES]
        if unknown:
            print(f"ОШИБКА: неизвестные разделы: {', '.join(unknown)}")
            print(f"Доступны: {', '.join(SECTION_MODULES)}")
            return 2

    # В консоли Windows по умолчанию cp1251, и печать Δ, ≈, ✓ роняет скрипт
    force_utf8_stdio()

    if not os.path.exists(args.pca_data):
        print(f"ОШИБКА: не найден датасет {args.pca_data}")
        return 1
    if (args.sections is None or "clustering" in args.sections) \
            and not os.path.exists(args.cluster_data):
        print(f"ОШИБКА: не найден датасет {args.cluster_data}")
        return 1

    t0 = time.time()
    set_seeds(SEED)

    print("=" * 78)
    print("Лабораторная работа №2")
    print("«Обучение без учителя. Понижение размерности. Обнаружение аномалий.")
    print(" Кластеризация»")
    print("=" * 78)
    print(f"Вариант : {args.variant}")
    print(f"Вывод   : {os.path.abspath(OUTPUT_DIR)}")
    print(f"Seed    : {SEED}")

    sections = args.sections

    def want(name: str) -> bool:
        return sections is None or name in sections

    state: dict = {}

    if want("pca"):
        banner("Пункт 1. Отбор признаков: корреляции, PCA, SVD, ICA")
        state["pca"] = get_module("pca_analysis").run_pca_analysis(
            args.pca_data, args.variant)

    if want("clustering"):
        banner("Пункт 2. Кластеризация и обнаружение аномалий")
        state["clustering"] = get_module("clustering").run_clustering(
            args.cluster_data, args.variant)

    dt = time.time() - t0
    from src.io_utils import output_files
    n_png, n_other = output_files()
    banner(f"Готово за {dt:.1f} с")
    print(f"Файлов в {OUTPUT_DIR}/: {n_png} png, {n_other} прочих")
    return 0


def _list_variants(pca_path: str, cluster_path: str) -> None:
    """Показывает, какие варианты есть в каждом датасете."""
    from src.io_utils import read_workbook
    for label, path in (("PCA", pca_path), ("Clustering", cluster_path)):
        if not os.path.exists(path):
            print(f"  {label:<12} файл не найден: {path}")
            continue
        df = read_workbook(path)
        found = sorted({c.split(" /")[0] for c in df.columns if " /" in c})
        print(f"  {label:<12} {path}: {len(found)} вариантов -> "
              f"{', '.join(v.replace('Вариант ', '') for v in found)}")
        counts = {v: int(df[[c for c in df.columns
                             if c.startswith(v + " /")]].notna().all(axis=1).sum())
                  for v in found}
        print(f"  {'':<12} заполненных строк по вариантам: "
              + ", ".join(f"{k.replace('Вариант ', '')}:{v}"
                          for k, v in counts.items()))


if __name__ == "__main__":
    sys.exit(main())
