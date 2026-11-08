"""Кодбук, таблицы, выгрузки и отчёт."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest
from docx import Document
from openpyxl import load_workbook

from socmon.analysis.dataset import Record
from socmon.analysis.outcomes import CallStatistics, Dispositions, response_rates
from socmon.analysis.wave import summarize
from socmon.analysis.weighting import Margin
from socmon.core.codes import AaporCode, SpecialCode
from socmon.core.questionnaire import Questionnaire
from socmon.export import codebook, report, tables, writers
from socmon.export.package import WavePackage
from socmon.export.package import build as build_package

MARGINS = [
    Margin("gender", {"Мужской": 0.46, "Женский": 0.54}),
    Margin("age_group", {"18-34": 0.30, "35-54": 0.36, "55+": 0.34}),
    Margin("territory", {"Северный": 0.30, "Центральный": 0.42, "Южный": 0.28}),
]
GENDERS = ["Мужской", "Женский"]
AGES = ["18-34", "35-54", "55+"]
TERRITORIES = ["Северный", "Центральный", "Южный"]


@pytest.fixture(scope="module")
def book(core: Questionnaire) -> codebook.Codebook:
    return codebook.build(core)


def make_records(n: int = 600) -> list[Record]:
    records: list[Record] = []
    for i in range(n):
        gender, age, territory = i % 2, i % 3, (i // 5) % 3
        refused = i % 25 == 0
        records.append(Record(
            id=f"r{i}",
            strata={
                "gender": GENDERS[gender],
                "age_group": AGES[age],
                "territory": TERRITORIES[territory],
            },
            answers={
                "life_situation": 5, "material_stability": 4,
                "anxiety": int(SpecialCode.REFUSED) if refused else 7,
                "fairness": 5, "institutional_trust": 4,
                "perceived_conflict": 6, "collective_action": (i % 4) + 1,
                "authorities_rating": (i % 9) + 1,
                "local_problem": int(SpecialCode.DONT_KNOW) if refused else (i % 5) + 1,
                "gender": gender + 1, "age_group": age + 1, "territory": territory + 1,
            },
        ))
    return records


@pytest.fixture(scope="module")
def package(core: Questionnaire, book: codebook.Codebook) -> WavePackage:
    import uuid

    records = make_records()
    summary = summarize(records, MARGINS, wave_id=uuid.uuid4(), target_min=500)
    dispositions = Dispositions.from_codes(
        [AaporCode.COMPLETE] * 600 + [AaporCode.REFUSAL] * 1200
        + [AaporCode.UNKNOWN_ELIGIBILITY] * 20_000 + [AaporCode.NOT_ELIGIBLE] * 100
    )
    statistics = CallStatistics(
        numbers_dialled=dispositions.total,
        attempts_made=dispositions.total * 3,
        dispositions=dispositions,
        rates=response_rates(dispositions),
        valid_interviews=600,
        invalid_interviews=120,
        exclusion_reasons={"not_completed": 80, "no_consent": 40},
    )
    return build_package(
        summary, book, wave_code="2026-17", call_statistics=statistics,
        now=datetime(2026, 8, 20, tzinfo=UTC),
        methodology_notes=("Порог длительности не применялся: утверждается по пилоту.",),
    )


# ------------------------------------------------------------------- кодбук


def test_codebook_covers_every_closed_question(book: codebook.Codebook) -> None:
    assert "anxiety" in book.variables
    assert "local_problem" in book.variables
    entry = book.by_variable("local_problem")
    assert entry is not None
    assert entry.values[1] == "Цены и доходы"


def test_codebook_is_pinned_to_the_questionnaire_version(
    book: codebook.Codebook, core: Questionnaire
) -> None:
    assert book.version_id == core.version_id


def test_service_codes_are_spelled_out(book: codebook.Codebook) -> None:
    """Без этого «97» в выгрузке выглядит как оценка по шкале."""
    entry = book.by_variable("anxiety")
    assert entry is not None
    assert entry.service_values[int(SpecialCode.DONT_KNOW)] == "затрудняюсь ответить"
    assert entry.value_label(int(SpecialCode.REFUSED)) == "отказ от ответа на вопрос"


def test_open_questions_have_no_service_codes(book: codebook.Codebook) -> None:
    entry = book.by_variable("q_open_leader")
    assert entry is not None
    assert entry.kind == "open"
    assert entry.service_values == {}


# ------------------------------------------------------------------ таблицы


def test_frequency_separates_substantive_from_service(
    book: codebook.Codebook
) -> None:
    """Доля «затрудняюсь» не должна разбавлять распределение мнений:
    иначе все содержательные проценты окажутся занижены."""
    entry = book.by_variable("local_problem")
    assert entry is not None
    table = tables.frequency(make_records(), entry)

    substantive = [r for r in table.rows if not r.is_service]
    assert sum(r.unweighted_share for r in substantive) == pytest.approx(1.0)
    service = [r for r in table.rows if r.is_service]
    assert service and service[0].unweighted_share == pytest.approx(24 / 600, abs=0.01)


def test_frequency_reports_its_base(book: codebook.Codebook) -> None:
    entry = book.by_variable("local_problem")
    assert entry is not None
    table = tables.frequency(make_records(), entry)
    assert table.total_n == 600
    assert table.substantive_n == 576
    assert table.missing_share == pytest.approx(0.04)


def test_crosstab_rows_sum_to_one(book: codebook.Codebook) -> None:
    rows_entry, columns_entry = book.by_variable("territory"), book.by_variable("local_problem")
    assert rows_entry is not None and columns_entry is not None
    table = tables.crosstab(make_records(), rows_entry, columns_entry)
    for shares in table.shares:
        assert sum(shares) == pytest.approx(1.0)


def test_crosstab_flags_thin_rows(book: codebook.Codebook) -> None:
    """ТЗ §5.2: по малым подгруппам — ограничение на интерпретацию."""
    rows_entry, columns_entry = book.by_variable("gender"), book.by_variable("local_problem")
    assert rows_entry is not None and columns_entry is not None
    table = tables.crosstab(make_records(40), rows_entry, columns_entry)
    assert set(table.unreliable_rows) == {"Мужской", "Женский"}


# ----------------------------------------------------------------- выгрузки


def test_package_writes_all_required_files(package: WavePackage, tmp_path: Path) -> None:
    written = writers.write_package(tmp_path, package)
    names = {p.name for p in written}
    assert names == {
        "wave-2026-17-массив.csv",
        "wave-2026-17-кодбук.csv",
        "wave-2026-17.xlsx",
    }
    assert all(p.stat().st_size > 0 for p in written)


def test_csv_opens_in_excel_with_cyrillic(package: WavePackage, tmp_path: Path) -> None:
    """Без BOM Excel показывает кириллицу кракозябрами, и половина
    получателей решит, что выгрузка битая."""
    path = writers.write_dataset_csv(tmp_path / "d.csv", package)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_dataset_keeps_service_codes(package: WavePackage, tmp_path: Path) -> None:
    """Различие между «отказался», «затруднился» и «не распознали» —
    это данные, и терять его при экспорте нельзя."""
    path = writers.write_dataset_csv(tmp_path / "d.csv", package)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    header, body = rows[0], rows[1:]
    column = header.index("anxiety")
    assert str(int(SpecialCode.REFUSED)) in {row[column] for row in body}
    assert len(body) == 600


def test_dataset_carries_weights_not_just_answers(
    package: WavePackage, tmp_path: Path
) -> None:
    """ТЗ §13.2 требует и невзвешенный, и взвешенный массив. Вес в строке
    позволяет получить оба из одного файла."""
    path = writers.write_dataset_csv(tmp_path / "d.csv", package)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    assert rows[0][1] == "weight"
    weights = [float(r[1]) for r in rows[1:]]
    assert sum(weights) == pytest.approx(len(weights), abs=0.01)
    assert len(set(weights)) > 1


def test_workbook_has_every_required_sheet(package: WavePackage, tmp_path: Path) -> None:
    path = writers.write_workbook(tmp_path / "w.xlsx", package)
    book = load_workbook(path)
    assert book.sheetnames == [
        "Массив", "Кодбук", "Распределения", "Кросс-табуляции",
        "Взвешивание", "Индекс", "Достижимость", "Ограничения",
    ]
    assert book["Массив"].max_row == 601


def test_workbook_records_the_index_recipe(package: WavePackage, tmp_path: Path) -> None:
    """ТЗ §12: воспроизводимость расчёта в выгрузке."""
    path = writers.write_workbook(tmp_path / "w.xlsx", package)
    values = [
        row for row in load_workbook(path)["Индекс"].iter_rows(values_only=True)
        if row[0] == "Отпечаток методики"
    ]
    assert values and values[0][1] == package.summary.index.recipe_hash


def test_limitations_sheet_is_never_silently_empty(
    package: WavePackage, tmp_path: Path
) -> None:
    path = writers.write_workbook(tmp_path / "w.xlsx", package)
    rows = list(load_workbook(path)["Ограничения"].iter_rows(values_only=True))
    assert len(rows) > 1


# -------------------------------------------------------------------- отчёт


def test_report_follows_the_structure_of_the_tor(
    package: WavePackage, tmp_path: Path
) -> None:
    """Приёмка §13.3 должна сводиться к сверке оглавления."""
    path = report.write_report(tmp_path / "r.docx", package)
    headings = [
        p.text for p in Document(str(path)).paragraphs
        if p.style is not None and p.style.name.startswith("Heading 1")
    ]
    joined = " ".join(headings)
    for expected in (
        "Дизайн исследования", "База, выборка", "Выполнение квот",
        "Статистические ограничения", "Индекс социального напряжения",
        "Сравнение территорий", "органов власти", "Открытые ответы",
        "двойных связей", "Выводы", "Приложение А", "Приложение Б",
    ):
        assert expected in joined


def test_report_does_not_write_its_own_conclusions(
    package: WavePackage, tmp_path: Path
) -> None:
    """ТЗ §13.3: выводы чётко отделены от первичных данных. Машинный текст,
    поданный как экспертное суждение, — ровно то смешение, что запрещено."""
    path = report.write_report(tmp_path / "r.docx", package)
    document = Document(str(path))
    texts = [p.text for p in document.paragraphs]
    start = next(i for i, t in enumerate(texts) if t.startswith("10. Выводы"))
    tail = [t for t in texts[start + 1:] if t.strip()]
    assert tail
    assert "заполняется аналитиком" in tail[0]
    assert not any("напряжение выросло" in t or "снизилось" in t for t in tail[:3])


def test_report_carries_the_limitations(package: WavePackage, tmp_path: Path) -> None:
    path = report.write_report(tmp_path / "r.docx", package)
    text = "\n".join(p.text for p in Document(str(path)).paragraphs)
    assert "Порог длительности не применялся" in text


def test_report_warns_that_rr5_flatters(package: WavePackage, tmp_path: Path) -> None:
    path = report.write_report(tmp_path / "r.docx", package)
    text = "\n".join(p.text for p in Document(str(path)).paragraphs)
    assert "RR5" in text and "некорректно" in text


def test_pending_sections_say_so_instead_of_guessing(
    package: WavePackage, tmp_path: Path
) -> None:
    """Раздел без данных остаётся пустым намеренно — заполнять его
    предположениями нельзя."""
    path = report.write_report(tmp_path / "r.docx", package)
    text = "\n".join(p.text for p in Document(str(path)).paragraphs)
    assert "заполнять его предположениями нельзя" in text


def test_pdf_conversion_fails_loudly_when_libreoffice_is_absent(
    package: WavePackage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Молча отдать один DOCX вместо пары DOCX+PDF — значит не выполнить
    требование §13.3 и не сказать об этом."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    path = report.write_report(tmp_path / "r.docx", package)
    with pytest.raises(report.PdfConversionError, match="LibreOffice"):
        report.to_pdf(path)


def test_xlsx_preserves_untrusted_text_without_creating_formulas(
    package: WavePackage, tmp_path: Path,
) -> None:
    from dataclasses import replace

    from openpyxl import load_workbook

    from socmon.export.writers import write_workbook

    malicious = '=HYPERLINK("https://attacker.example","click")'
    record = replace(package.records[0], id=malicious)
    path = write_workbook(tmp_path / "safe.xlsx", replace(package, records=(record,)))
    book = load_workbook(path, data_only=False)
    assert book["Массив"]["A2"].value == malicious
    assert book["Массив"]["A2"].data_type == "s"
    assert all(cell.data_type != "f" for sheet in book for row in sheet for cell in row)
