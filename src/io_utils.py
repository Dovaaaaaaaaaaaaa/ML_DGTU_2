"""Чтение/запись таблиц и сохранение фигур.

Главная особенность датасетов лабораторной №2 — ДВУХУРОВНЕВЫЙ заголовок Excel:
в первой строке объединённые ячейки с названием варианта («Вариант 7»), во
второй — имена признаков (a, b, c, d, e, y). Поэтому `header=[0, 1]`, а
`pandas.read_excel(path)` без указания header читает только первую строку и
получает столбцы «Unnamed: 1» … «Unnamed: 143».
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import timestamped_filename


def read_workbook(path: str) -> pd.DataFrame:
    """Читает лист вариантов с двухуровневым заголовком и «расплющивает» его.

    Возвращает DataFrame с однозначными именами столбцов вида
    «Вариант 7 / a», «Вариант 7 / y» — иначе в DataFrame попадут 144 столбца,
    у которых в первой строке стоит «Вариант N», а во второй — имя признака.
    """
    p = Path(path)
    if p.suffix.lower() != ".xlsx":
        raise ValueError(f"Ожидается .xlsx, получено: {p.suffix}")

    raw = pd.read_excel(p, header=[0, 1])
    if not isinstance(raw.columns, pd.MultiIndex):
        # Страховка: если в файле одна строка заголовка, level_0 может быть
        # единственным уровнем — тогда подставляем его в оба уровня.
        top = pd.Series([str(c) for c in raw.columns])
        bottom = pd.Series([""] * len(raw.columns))
    else:
        top = pd.Series([str(a) for a, _ in raw.columns])
        bottom = pd.Series([str(b) for _, b in raw.columns])

    # Объединённые ячейки Excel при чтении дают NaN в первой строке заголовка:
    # они относятся к тому же варианту, что и соседний заполненный столбец.
    # Заполнять нужно ТОЛЬКО заголовок. Раньше здесь стоял raw.ffill(axis=1),
    # который заполнял пропуски по всей таблице и подменял NaN в данных
    # значением соседнего столбца.
    top = top.ffill()
    raw.columns = [f"{a} / {b}".strip() for a, b in zip(top, bottom)]
    return raw


def extract_variant(df: pd.DataFrame, variant: int) -> pd.DataFrame:
    """Оставляет только столбцы одного варианта, имена становятся a, b, c, y."""
    prefix = f"Вариант {variant}"
    cols = [c for c in df.columns if c.startswith(prefix + " /") or c == prefix]
    if not cols:
        available = sorted({c.split(" /")[0] for c in df.columns if " /" in c})
        raise KeyError(f"Вариант {variant} не найден. Доступны: {available}")

    out = df[cols].copy()
    out.columns = [c.split(" / ", 1)[-1].strip() or prefix for c in cols]
    # 'Вариант 7 / y' -> 'y'
    out.columns = [f"y" if c in ("y", prefix) else c for c in out.columns]
    return out


def read_variant(path: str, variant: int) -> pd.DataFrame:
    """Полный путь: файл -> нужный вариант -> плоские имена столбцов."""
    return extract_variant(read_workbook(path), variant)


def save_figure(fig, base_name: str, close: bool = True) -> str:
    """Сохраняет matplotlib-фигуру в output/ и возвращает путь к файлу."""
    path = timestamped_filename(base_name, "png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return path


def write_table(df: pd.DataFrame, base_name: str) -> dict:
    """Сохраняет DataFrame в csv / xlsx / txt с timestamp в имени файла."""
    paths = {}
    paths["csv"] = timestamped_filename(base_name, "csv")
    df.to_csv(paths["csv"], index=False, encoding="utf-8-sig")
    paths["xlsx"] = timestamped_filename(base_name, "xlsx")
    df.to_excel(paths["xlsx"], index=False, engine="openpyxl")
    return paths


def output_files() -> tuple[int, int]:
    """Возвращает (число png, число прочих файлов) в output/."""
    if not os.path.isdir("output"):
        return 0, 0
    files = os.listdir("output")
    return len([f for f in files if f.endswith(".png")]), len([f for f in files if not f.endswith(".png")])
