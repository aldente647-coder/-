"""Запись оперативного пакета в файлы (ТЗ §13.2).

Массив отдаётся и в XLSX, и в CSV — как требует ТЗ. CSV пишется с BOM:
без него Excel открывает кириллицу кракозябрами, и половина получателей
решит, что выгрузка битая.

Служебные коды в массиве остаются кодами, а не превращаются в пустые ячейки:
различие между «отказался», «затруднился» и «не распознали» — это данные,
и терять его при экспорте нельзя.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from socmon.export.package import WavePackage

HEADER_FILL = PatternFill("solid", fgColor="1F2A33")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TITLE_FONT = Font(bold=True, size=12)


def validate_wave_code(code: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", code):
        raise ValueError("код волны: 1–32 латинские буквы, цифры, дефис или подчёркивание")
    return code


def csv_value(value: Any) -> Any:
    """Текст респондента не должен становиться формулой при открытии CSV."""
    if (isinstance(value, str) and value
            and (value[0] in "\t\r\n" or value.lstrip().startswith(("=", "+", "-", "@")))):
        return "'" + value
    return value


def write_package(directory: Path, package: WavePackage) -> list[Path]:
    """Разложить пакет по файлам. Возвращает список записанного."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"wave-{validate_wave_code(package.wave_code)}"
    written = [
        write_dataset_csv(directory / f"{stem}-массив.csv", package),
        write_codebook_csv(directory / f"{stem}-кодбук.csv", package),
        write_workbook(directory / f"{stem}.xlsx", package),
    ]
    return written


# ------------------------------------------------------------------------ CSV


def _dataset_header(package: WavePackage) -> list[str]:
    return ["interview_id", "weight", *package.codebook.variables]


def _dataset_rows(package: WavePackage) -> list[list[Any]]:
    variables = package.codebook.variables
    return [
        [record.id, round(record.weight, 6), *(record.get(v) for v in variables)]
        for record in package.records
    ]


def write_dataset_csv(path: Path, package: WavePackage) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", quoting=csv.QUOTE_ALL)
        writer.writerow([csv_value(v) for v in _dataset_header(package)])
        writer.writerows([csv_value(v) for v in row] for row in _dataset_rows(package))
    return path


def write_codebook_csv(path: Path, package: WavePackage) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", quoting=csv.QUOTE_ALL)
        writer.writerow([
            "переменная", "узел анкеты", "раздел", "тип",
            "формулировка", "код", "значение", "служебный",
        ])
        for entry in package.codebook:
            values = list(entry.values.items()) + list(entry.service_values.items())
            if not values:
                writer.writerow([
                    *map(csv_value, [entry.variable, entry.node_id, entry.section, entry.kind_label,
                                    entry.question, "", "текст ответа", ""]),
                ])
                continue
            service_codes = set(entry.service_values)
            for code, label in values:
                writer.writerow([
                    *map(csv_value, [entry.variable, entry.node_id, entry.section, entry.kind_label,
                                    entry.question, code, label]),
                    "да" if code in service_codes else "",
                ])
    return path


# ----------------------------------------------------------------------- XLSX


def write_workbook(path: Path, package: WavePackage) -> Path:
    book = Workbook()
    if book.active is not None:
        book.remove(book.active)

    _sheet_dataset(book, package)
    _sheet_codebook(book, package)
    _sheet_frequencies(book, package)
    _sheet_crosstabs(book, package)
    _sheet_weighting(book, package)
    _sheet_index(book, package)
    _sheet_outcomes(book, package)
    _sheet_limitations(book, package)

    # В книге нет вычисляемых формул: расчёты уже сделаны в analysis.
    # Сохраняем произвольный текст дословно, но явно с типом строки.
    for sheet in book:
        for row in sheet:
            for cell in row:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
    book.save(path)
    return path


def _write_header(sheet: Any, row: int, values: list[Any]) -> None:
    for column, value in enumerate(values, start=1):
        cell = sheet.cell(row=row, column=column, value=value)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def _autosize(sheet: Any, widths: dict[int, int]) -> None:
    for column, width in widths.items():
        sheet.column_dimensions[get_column_letter(column)].width = width


def _sheet_dataset(book: Workbook, package: WavePackage) -> None:
    sheet = book.create_sheet("Массив")
    header = _dataset_header(package)
    _write_header(sheet, 1, header)
    for row in _dataset_rows(package):
        sheet.append(row)
    sheet.freeze_panes = "C2"
    _autosize(sheet, {1: 38, 2: 10, **{i: 16 for i in range(3, len(header) + 1)}})


def _sheet_codebook(book: Workbook, package: WavePackage) -> None:
    sheet = book.create_sheet("Кодбук")
    _write_header(sheet, 1, [
        "переменная", "раздел", "тип", "формулировка", "код", "значение", "служебный",
    ])
    for entry in package.codebook:
        values = list(entry.values.items()) + list(entry.service_values.items())
        service_codes = set(entry.service_values)
        if not values:
            sheet.append([entry.variable, entry.section, entry.kind_label,
                          entry.question, None, "текст ответа", None])
            continue
        for code, label in values:
            sheet.append([
                entry.variable, entry.section, entry.kind_label, entry.question,
                code, label, "да" if code in service_codes else None,
            ])
    sheet.freeze_panes = "A2"
    _autosize(sheet, {1: 22, 2: 14, 3: 22, 4: 60, 5: 8, 6: 30, 7: 12})


def _sheet_frequencies(book: Workbook, package: WavePackage) -> None:
    sheet = book.create_sheet("Распределения")
    row = 1
    for table in package.frequencies:
        sheet.cell(row=row, column=1, value=f"{table.variable} — {table.question}").font = (
            TITLE_FONT
        )
        row += 1
        _write_header(sheet, row, [
            "код", "значение", "n", "невзвеш., %", "взвеш., %",
            "95% ДИ низ", "95% ДИ верх", "служебный",
        ])
        row += 1
        for item in table.rows:
            sheet.cell(row=row, column=1, value=item.code)
            sheet.cell(row=row, column=2, value=item.label)
            sheet.cell(row=row, column=3, value=item.count)
            sheet.cell(row=row, column=4, value=round(item.unweighted_share * 100, 1))
            sheet.cell(row=row, column=5, value=round(item.weighted_share * 100, 1))
            if item.estimate is not None:
                sheet.cell(row=row, column=6, value=round(item.estimate.ci_low * 100, 1))
                sheet.cell(row=row, column=7, value=round(item.estimate.ci_high * 100, 1))
            sheet.cell(row=row, column=8, value="да" if item.is_service else None)
            row += 1
        sheet.cell(row=row, column=2,
                   value=f"база содержательных ответов: {table.substantive_n} из {table.total_n}")
        row += 2
    _autosize(sheet, {1: 8, 2: 34, 3: 8, 4: 14, 5: 14, 6: 12, 7: 12, 8: 12})


def _sheet_crosstabs(book: Workbook, package: WavePackage) -> None:
    sheet = book.create_sheet("Кросс-табуляции")
    row = 1
    for table in package.crosstabs:
        sheet.cell(row=row, column=1,
                   value=f"{table.column_variable} × {table.row_variable}, "
                         f"взвешенные % по строке").font = TITLE_FONT
        row += 1
        _write_header(sheet, row, ["группа", "n", *table.column_labels])
        row += 1
        for label, total, shares in zip(
            table.row_labels, table.row_totals, table.shares, strict=True
        ):
            sheet.cell(row=row, column=1, value=label)
            sheet.cell(row=row, column=2, value=total)
            for offset, share in enumerate(shares, start=3):
                sheet.cell(row=row, column=offset, value=round(share * 100, 1))
            row += 1
        if table.unreliable_rows:
            sheet.cell(row=row, column=1,
                       value="малая подвыборка, справочно: "
                             + ", ".join(table.unreliable_rows))
            row += 1
        row += 1
    _autosize(sheet, {1: 26, 2: 8, **{i: 18 for i in range(3, 12)}})


def _sheet_weighting(book: Workbook, package: WavePackage) -> None:
    sheet = book.create_sheet("Взвешивание")
    diagnostics = package.summary.weights

    sheet.cell(row=1, column=1, value="Диагностика весов").font = TITLE_FONT
    rows: list[tuple[str, Any]] = [
        ("Метод", "итеративное пропорциональное взвешивание (raking, IPF)"),
        ("Наблюдений", diagnostics.n),
        ("Минимальный вес", round(diagnostics.min_weight, 4)),
        ("Максимальный вес", round(diagnostics.max_weight, 4)),
        ("Средний вес", round(diagnostics.mean_weight, 4)),
        ("Доля экстремальных весов, %", round(diagnostics.extreme_share * 100, 2)),
        ("Эффект дизайна", round(diagnostics.design_effect, 4)),
        ("Эффективный объём выборки", round(diagnostics.effective_n, 1)),
        ("Итераций", diagnostics.iterations),
        ("Сошлось", "да" if diagnostics.converged else "нет"),
        ("Максимальное отклонение от эталона",
         round(diagnostics.max_margin_deviation, 8)),
    ]
    row = 2
    for name, value in rows:
        sheet.cell(row=row, column=1, value=name)
        sheet.cell(row=row, column=2, value=value)
        row += 1

    row += 1
    sheet.cell(row=row, column=1, value="Структура выборки").font = TITLE_FONT
    row += 1
    _write_header(sheet, row, ["признак", "категория", "невзвеш., %", "взвеш., %", "эталон, %"])
    row += 1
    for variable, targets in diagnostics.reference.items():
        for category in sorted(targets):
            sheet.cell(row=row, column=1, value=variable)
            sheet.cell(row=row, column=2, value=category)
            sheet.cell(row=row, column=3,
                       value=round(diagnostics.unweighted[variable].get(category, 0) * 100, 1))
            sheet.cell(row=row, column=4,
                       value=round(diagnostics.achieved[variable].get(category, 0) * 100, 1))
            sheet.cell(row=row, column=5, value=round(targets[category] * 100, 1))
            row += 1
    _autosize(sheet, {1: 36, 2: 26, 3: 16, 4: 14, 5: 14})


def _sheet_index(book: Workbook, package: WavePackage) -> None:
    sheet = book.create_sheet("Индекс")
    index = package.summary.index

    sheet.cell(row=1, column=1, value="Индекс социального напряжения").font = TITLE_FONT
    rows: list[tuple[str, Any]] = [
        ("Значение (агрегатное)", round(index.aggregate, 3)),
        ("Шкала", "0-10, выше — напряжённее"),
        ("Формула", "I = (E+F+A+J+D+C+P) / 7"),
        ("Обработка пропусков", index.missing_policy.value),
        ("Отпечаток методики", index.recipe_hash),
        ("Полных наблюдений", f"{index.complete_cases} из {index.n}"),
    ]
    if index.individual is not None:
        rows.insert(1, ("Значение (по индивидуальным данным)",
                        round(index.individual.value, 3)))
        rows.insert(2, ("95% доверительный интервал",
                        f"{index.individual.ci_low:.2f}–{index.individual.ci_high:.2f}"))
    row = 2
    for name, value in rows:
        sheet.cell(row=row, column=1, value=name)
        sheet.cell(row=row, column=2, value=value)
        row += 1

    row += 1
    sheet.cell(row=row, column=1, value="Компонентный профиль").font = TITLE_FONT
    row += 1
    _write_header(sheet, row, [
        "код", "компонент", "показатель ядра", "значение", "95% ДИ", "пропусков, %",
    ])
    row += 1
    for component in index.components:
        sheet.cell(row=row, column=1, value=component.key)
        sheet.cell(row=row, column=2, value=component.name)
        sheet.cell(row=row, column=3, value=component.core_key)
        sheet.cell(row=row, column=4, value=round(component.value, 3))
        sheet.cell(row=row, column=5,
                   value=f"{component.estimate.ci_low:.2f}–{component.estimate.ci_high:.2f}")
        sheet.cell(row=row, column=6, value=round(component.missing_share * 100, 1))
        row += 1

    for variable, breakdown in package.index_breakdowns.items():
        row += 1
        sheet.cell(row=row, column=1, value=f"Индекс в разрезе: {variable}").font = TITLE_FONT
        row += 1
        _write_header(sheet, row, ["группа", "n", "индекс"])
        row += 1
        for label, count, value in breakdown:
            sheet.cell(row=row, column=1, value=label)
            sheet.cell(row=row, column=2, value=count)
            sheet.cell(row=row, column=3,
                       value=round(value, 3) if value is not None
                       else "подвыборка мала, не публикуется")
            row += 1
    _autosize(sheet, {1: 36, 2: 40, 3: 22, 4: 14, 5: 16, 6: 14})


def _sheet_outcomes(book: Workbook, package: WavePackage) -> None:
    sheet = book.create_sheet("Достижимость")
    statistics = package.call_statistics
    if statistics is None:
        sheet.cell(row=1, column=1,
                   value="Данные обзвона не переданы: массив собран вне телефонного контура")
        _autosize(sheet, {1: 80})
        return

    d, r = statistics.dispositions, statistics.rates
    sheet.cell(row=1, column=1, value="Ход обзвона").font = TITLE_FONT
    rows: list[tuple[str, Any]] = [
        ("Номеров в работе", statistics.numbers_dialled),
        ("Попыток дозвона", statistics.attempts_made),
        ("Попыток на номер", round(statistics.attempts_per_number, 2)),
        ("Завершённых интервью (1.1)", d.interviews),
        ("Частичных интервью (1.2)", d.partials),
        ("Отказов и прерываний (2.11, 2.12)", d.refusals),
        ("Не удалось поговорить (2.20)", d.non_contacts),
        ("Прочие пригодные (2.30, 2.33)", d.other_eligible),
        ("Пригодность неизвестна (3.20)", d.unknown_eligibility),
        ("Не подошли по фильтрам (4.70)", d.not_eligible),
        ("Недействующие номера (4.10)", d.out_of_sample),
        ("", ""),
        ("Валидных интервью", statistics.valid_interviews),
        ("Невалидных интервью", statistics.invalid_interviews),
        ("", ""),
        ("RR1", round(r.rr1, 4)),
        ("RR2", round(r.rr2, 4)),
        ("RR3 (основной публикуемый)", round(r.rr3, 4)),
        ("RR4", round(r.rr4, 4)),
        ("RR5", round(r.rr5, 4)),
        ("RR6", round(r.rr6, 4)),
        ("Коэффициент кооперации COOP1", round(r.cooperation1, 4)),
        ("Коэффициент отказов REF1", round(r.refusal1, 4)),
        ("Коэффициент контакта CON1", round(r.contact1, 4)),
        ("Оценка доли пригодных e", round(r.eligibility_rate, 4)),
    ]
    row = 2
    for name, value in rows:
        sheet.cell(row=row, column=1, value=name or None)
        sheet.cell(row=row, column=2, value=value if value != "" else None)
        row += 1

    if statistics.exclusion_reasons:
        row += 1
        sheet.cell(row=row, column=1, value="Основания исключения интервью").font = TITLE_FONT
        row += 1
        _write_header(sheet, row, ["основание", "интервью"])
        row += 1
        for reason, count in sorted(
            statistics.exclusion_reasons.items(), key=lambda kv: -kv[1]
        ):
            sheet.cell(row=row, column=1, value=reason)
            sheet.cell(row=row, column=2, value=count)
            row += 1
    _autosize(sheet, {1: 42, 2: 18})


def _sheet_limitations(book: Workbook, package: WavePackage) -> None:
    sheet = book.create_sheet("Ограничения")
    sheet.cell(row=1, column=1, value="Методологические ограничения").font = TITLE_FONT
    row = 2
    for note in package.limitations:
        sheet.cell(row=row, column=1, value=note).alignment = Alignment(wrap_text=True)
        row += 1
    if row == 2:
        sheet.cell(row=2, column=1, value="Ограничений, требующих отдельной оговорки, нет")
    _autosize(sheet, {1: 110})
