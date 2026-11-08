"""Аналитический отчёт по волне в DOCX (ТЗ §13.3).

Структура разделов повторяет перечень §13.3 буквально, чтобы приёмка сводилась
к сверке оглавления, а не к вычитыванию.

Главное правило документа — из того же §13.3: **выводы чётко отделены от
первичных данных**. Поэтому генератор пишет фактическую часть целиком, а раздел
выводов оставляет за аналитиком. Машинно составленный текст, поданный как
экспертное суждение, — это ровно то смешение, которое §13.3 запрещает.

Оформление — фирменное: Times New Roman, графит и золото.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

from socmon.export.package import WavePackage

GRAPHITE = RGBColor(0x1F, 0x2A, 0x33)
GOLD = RGBColor(0xB8, 0x8C, 0x46)
BODY_FONT = "Times New Roman"

PENDING = (
    "Раздел заполняется по результатам ночного кодирования открытых ответов. "
    "Соответствующий контур подключается на следующем этапе работ; "
    "до этого раздел остаётся пустым намеренно — заполнять его "
    "предположениями нельзя."
)


class PdfConversionError(RuntimeError):
    pass


def write_report(path: Path, package: WavePackage) -> Path:
    """Собрать отчёт по волне."""
    document = Document()
    _setup_styles(document)

    _title_page(document, package)
    _design(document, package)
    _sample(document, package)
    _quotas_and_weighting(document, package)
    _statistical_limits(document, package)
    _index(document, package)
    _subgroups(document, package)
    _authorities(document, package)
    _open_answers(document)
    _crosstabs(document, package)
    _conclusions(document)
    _appendix(document, package)

    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    return path


def to_pdf(docx_path: Path, output_dir: Path | None = None) -> Path:
    """Конвертировать отчёт в PDF через LibreOffice.

    ТЗ §13.3 требует оба формата. Конвертация вынесена отдельным шагом и падает
    с внятным сообщением, если LibreOffice не установлен: молча отдать один
    DOCX вместо пары DOCX+PDF — значит не выполнить требование и не сказать
    об этом.
    """
    binary = shutil.which("soffice") or shutil.which("libreoffice")
    if binary is None:
        raise PdfConversionError(
            "не найден LibreOffice: PDF не собран. "
            "Установите его (`brew install --cask libreoffice`) "
            "или конвертируйте DOCX вручную"
        )
    target = output_dir or docx_path.parent
    subprocess.run(
        [binary, "--headless", "--convert-to", "pdf", "--outdir", str(target), str(docx_path)],
        check=True,
        capture_output=True,
        timeout=180,
    )
    produced = target / f"{docx_path.stem}.pdf"
    if not produced.exists():
        raise PdfConversionError(f"LibreOffice отработал, но {produced} не появился")
    return produced


# ------------------------------------------------------------------ оформление


def _setup_styles(document: Any) -> None:
    normal = document.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(12)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for level, size in ((1, 16), (2, 13)):
        style = document.styles[f"Heading {level}"]
        style.font.name = BODY_FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = GRAPHITE if level == 1 else GOLD

    for section in document.sections:
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(1.5)


def _heading(document: Any, text: str, level: int = 1, tor: str | None = None) -> None:
    heading = document.add_heading(text, level=level)
    if tor:
        run = heading.add_run(f"   ТЗ {tor}")
        run.font.size = Pt(9)
        run.font.color.rgb = GOLD
        run.bold = False


def _facts(document: Any, rows: Sequence[tuple[str, Any]]) -> None:
    """Двухколоночная таблица «показатель — значение»."""
    table = document.add_table(rows=0, cols=2)
    table.style = "Light Grid Accent 1"
    for name, value in rows:
        cells = table.add_row().cells
        cells[0].text = name
        cells[1].text = "" if value is None else str(value)
    document.add_paragraph()


def _grid(document: Any, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    for cell, title in zip(table.rows[0].cells, header, strict=True):
        cell.text = str(title)
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row, strict=True):
            cell.text = "" if value is None else str(value)
    document.add_paragraph()


def _note(document: Any, text: str) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = GRAPHITE


# -------------------------------------------------------------------- разделы


def _title_page(document: Any, package: WavePackage) -> None:
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Аналитический отчёт по итогам волны мониторинга")
    run.bold = True
    run.font.size = Pt(20)
    run.font.color.rgb = GRAPHITE

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = subtitle.add_run(
        f"Волна {package.wave_code} · сформирован "
        f"{package.generated_at.strftime('%d.%m.%Y')}"
    )
    sub.font.size = Pt(12)
    sub.font.color.rgb = GOLD

    index = package.summary.index
    headline = document.add_paragraph()
    headline.alignment = WD_ALIGN_PARAGRAPH.CENTER
    value = headline.add_run(f"\nИндекс социального напряжения: {index.aggregate:.2f} из 10")
    value.bold = True
    value.font.size = Pt(14)
    value.font.color.rgb = GRAPHITE
    document.add_page_break()


def _design(document: Any, package: WavePackage) -> None:
    _heading(document, "1. Дизайн исследования", tor="§3, §5")
    document.add_paragraph(
        "Непрерывный телефонный социологический мониторинг с применением "
        "голосового робота на основе искусственного интеллекта. Единица "
        "отчётности — двухнедельная волна. Интервью проводятся на русском языке "
        "по анкете, состоящей из постоянного измерительного ядра и переменного "
        "тематического блока."
    )
    _facts(document, [
        ("Метод сбора", "исходящий телефонный обзвон, голосовой робот"),
        ("Анкета", package.codebook.questionnaire_name),
        ("Версия анкеты", package.codebook.version_id),
        ("Переменных в кодбуке", len(package.codebook)),
        ("Валидных интервью", package.summary.valid_interviews),
        ("Целевой минимум волны", package.summary.target_min),
    ])
    _note(document, (
        "Версия анкеты — отпечаток её содержимого. Любая правка формулировки "
        "даёт новую версию, поэтому данные разных волн невозможно случайно "
        "смешать под видом одинаковых."
    ))


def _sample(document: Any, package: WavePackage) -> None:
    _heading(document, "2. База, выборка и достигнутая аудитория", tor="§4, §13.2")
    statistics = package.call_statistics
    if statistics is None:
        _note(document, (
            "Сведения о ходе обзвона не переданы: массив сформирован вне "
            "телефонного контура. Показатели достижимости в этой волне "
            "не рассчитываются."
        ))
        return

    d, r = statistics.dispositions, statistics.rates
    _facts(document, [
        ("Номеров в работе", d.total),
        ("Попыток дозвона", statistics.attempts_made),
        ("Попыток на номер", f"{statistics.attempts_per_number:.2f}"),
        ("Завершённых интервью", d.interviews),
        ("Отказов и прерываний", d.refusals),
        ("Пригодность не установлена", d.unknown_eligibility),
        ("Недействующих номеров", d.out_of_sample),
    ])

    document.add_paragraph(
        "Достижимость рассчитана по AAPOR Standard Definitions. Основным "
        "публикуемым показателем является RR3."
    )
    _grid(document, ["Коэффициент", "Значение", "Что означает"], [
        ["RR3", f"{r.rr3:.1%}", "доля ответивших с оценкой доли пригодных"],
        ["RR1", f"{r.rr1:.1%}", "строгая нижняя оценка"],
        ["RR5", f"{r.rr5:.1%}", "верхняя граница, номера неизвестной пригодности исключены"],
        ["COOP1", f"{r.cooperation1:.1%}", "доля согласившихся среди поговоривших"],
        ["CON1", f"{r.contact1:.1%}", "доля номеров, где удалось выйти на человека"],
        ["e", f"{r.eligibility_rate:.3f}", "оценка доли пригодных"],
    ])
    _note(document, (
        "RR5 систематически выше остальных коэффициентов, поскольку исключает "
        "номера неизвестной пригодности. Публиковать его как «долю ответивших» "
        "некорректно; он приводится для полноты."
    ))


def _quotas_and_weighting(document: Any, package: WavePackage) -> None:
    _heading(document, "3. Выполнение квот и методика взвешивания", tor="§5.4")
    diagnostics = package.summary.weights
    document.add_paragraph(
        "Структура выборки приведена к эталонным распределениям Заказчика "
        "методом итеративного пропорционального взвешивания (raking). "
        "В расчёт весов входят исключительно признаки структуры выборки; "
        "содержательные ответы на веса не влияют."
    )
    _grid(
        document,
        ["Признак", "Категория", "Невзвеш., %", "Взвеш., %", "Эталон, %"],
        [
            [
                variable, category,
                f"{diagnostics.unweighted[variable].get(category, 0) * 100:.1f}",
                f"{diagnostics.achieved[variable].get(category, 0) * 100:.1f}",
                f"{targets[category] * 100:.1f}",
            ]
            for variable, targets in diagnostics.reference.items()
            for category in sorted(targets)
        ],
    )
    _facts(document, [
        ("Минимальный вес", f"{diagnostics.min_weight:.3f}"),
        ("Максимальный вес", f"{diagnostics.max_weight:.3f}"),
        ("Доля наблюдений с экстремальными весами",
         f"{diagnostics.extreme_share:.1%}"),
        ("Эффект дизайна", f"{diagnostics.design_effect:.3f}"),
        ("Эффективный объём выборки", f"{diagnostics.effective_n:.0f}"),
        ("Сходимость", "достигнута" if diagnostics.converged else "не достигнута"),
    ])


def _statistical_limits(document: Any, package: WavePackage) -> None:
    _heading(document, "4. Статистические ограничения", tor="§5.2, §13.2")
    diagnostics = package.summary.weights
    document.add_paragraph(
        f"Доверительные интервалы рассчитаны при доверительной вероятности 95% "
        f"с учётом весов и эффекта дизайна. Эффективный объём выборки составляет "
        f"{diagnostics.effective_n:.0f} наблюдений против {diagnostics.n} фактических; "
        f"именно он определяет фактическую ширину интервалов."
    )
    if package.limitations:
        document.add_paragraph("Ограничения интерпретации по данной волне:")
        for note in package.limitations:
            document.add_paragraph(note, style="List Bullet")
    else:
        document.add_paragraph("Дополнительных ограничений по данной волне не выявлено.")


def _index(document: Any, package: WavePackage) -> None:
    _heading(document, "5. Индекс социального напряжения", tor="§12")
    index = package.summary.index
    document.add_paragraph(
        "Индекс рассчитывается как среднее семи компонентов, приведённых "
        "к шкале 0-10. Единая ориентация шкалы: выше значение — выше напряжение."
    )
    rows: list[tuple[str, Any]] = [
        ("Значение индекса (агрегатное)", f"{index.aggregate:.2f}"),
        ("Формула", "I = (E+F+A+J+D+C+P) / 7"),
        ("Обработка отказов и затруднений",
         "исключаются из базы компонента"
         if index.missing_policy.value == "exclude"
         else "заменяются серединой шкалы по согласованию"),
        ("Полных наблюдений", f"{index.complete_cases} из {index.n}"),
        ("Отпечаток методики расчёта", index.recipe_hash),
    ]
    if index.individual is not None:
        rows.insert(1, (
            "Значение по индивидуальным данным",
            f"{index.individual.value:.2f} "
            f"(95% ДИ {index.individual.ci_low:.2f}–{index.individual.ci_high:.2f})",
        ))
    _facts(document, rows)

    _heading(document, "5.1. Компонентный профиль", level=2)
    _grid(
        document,
        ["Код", "Компонент", "Значение", "95% ДИ", "Пропусков, %"],
        [
            [
                c.key, c.name, f"{c.value:.2f}",
                f"{c.estimate.ci_low:.2f}–{c.estimate.ci_high:.2f}",
                f"{c.missing_share * 100:.1f}",
            ]
            for c in index.components
        ],
    )
    _note(document, (
        "Отпечаток методики фиксирует состав компонентов, направление шкал "
        "и правило обработки пропусков. При изменении методики отпечаток "
        "меняется, и показатели разных волн не могут быть сопоставлены по ошибке."
    ))


def _subgroups(document: Any, package: WavePackage) -> None:
    _heading(document, "6. Сравнение территорий и социальных групп", tor="§12, §13.3")
    if not package.index_breakdowns:
        _note(document, "Признаки для разрезов в данной волне не заданы.")
        return
    for variable, rows in package.index_breakdowns.items():
        _heading(document, f"6.{variable}", level=2)
        _grid(
            document,
            ["Группа", "Наблюдений", "Индекс"],
            [
                [label, count,
                 f"{value:.2f}" if value is not None else "подвыборка мала, не публикуется"]
                for label, count, value in rows
            ],
        )


def _authorities(document: Any, package: WavePackage) -> None:
    _heading(document, "7. Оценка работы органов власти", tor="§7.2, §13.3")
    table = next(
        (t for t in package.frequencies if t.variable == "authorities_rating"), None
    )
    if table is None:
        _note(document, "Показатель в данной волне не измерялся.")
        return
    _grid(
        document,
        ["Оценка", "n", "Взвеш., %"],
        [
            [row.label or row.code, row.count, f"{row.weighted_share * 100:.1f}"]
            for row in table.rows
        ],
    )
    _note(document, (
        f"База содержательных ответов: {table.substantive_n} из {table.total_n}. "
        f"Отказы и затруднения в распределение оценок не включены."
    ))


def _open_answers(document: Any) -> None:
    _heading(document, "8. Открытые ответы, темы и лидеры мнений", tor="§7.5, §11.2")
    _note(document, PENDING)


def _crosstabs(document: Any, package: WavePackage) -> None:
    _heading(document, "9. Таблицы двойных связей", tor="§13.2, §13.3")
    if not package.crosstabs:
        _note(document, "Кросс-табуляции в данной волне не строились.")
        return
    for table in package.crosstabs:
        _heading(
            document,
            f"9.{table.column_variable} × {table.row_variable}",
            level=2,
        )
        _grid(
            document,
            ["Группа", "n", *table.column_labels],
            [
                [label, total, *(f"{share * 100:.1f}" for share in shares)]
                for label, total, shares in zip(
                    table.row_labels, table.row_totals, table.shares, strict=True
                )
            ],
        )
        if table.unreliable_rows:
            _note(document, (
                "Малая подвыборка, значения приводятся справочно: "
                + ", ".join(table.unreliable_rows)
            ))


def _conclusions(document: Any) -> None:
    _heading(document, "10. Выводы", tor="§13.3")
    _note(document, (
        "ТЗ §13.3 требует, чтобы выводы были чётко отделены от первичных "
        "данных. Разделы 1-9 содержат исключительно фактуру: измеренные "
        "величины, их доверительные интервалы и ограничения интерпретации. "
        "Настоящий раздел заполняется аналитиком и не формируется "
        "автоматически."
    ))
    document.add_paragraph()


def _appendix(document: Any, package: WavePackage) -> None:
    document.add_page_break()
    _heading(document, "Приложение А. Одномерные распределения", tor="§13.2")
    for table in package.frequencies:
        _heading(document, table.variable, level=2)
        document.add_paragraph(table.question)
        _grid(
            document,
            ["Код", "Значение", "n", "Невзвеш., %", "Взвеш., %", "95% ДИ"],
            [
                [
                    row.code, row.label, row.count,
                    f"{row.unweighted_share * 100:.1f}",
                    f"{row.weighted_share * 100:.1f}",
                    f"{row.estimate.ci_low * 100:.1f}–{row.estimate.ci_high * 100:.1f}"
                    if row.estimate else "—",
                ]
                for row in table.rows
            ],
        )

    document.add_page_break()
    _heading(document, "Приложение Б. Кодбук", tor="§13.2")
    _grid(
        document,
        ["Переменная", "Тип", "Формулировка", "Значения"],
        [
            [
                entry.variable,
                entry.kind_label,
                entry.question,
                "; ".join(f"{code} — {label}" for code, label in entry.values.items())
                or "текст ответа",
            ]
            for entry in package.codebook
        ],
    )
    _note(document, (
        "Служебные коды едины для всех закрытых вопросов: "
        + "; ".join(
            f"{code} — {label}"
            for code, label in next(iter(package.codebook)).service_values.items()
        )
    ))
