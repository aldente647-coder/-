"""Сводка по волне: взвешивание, индекс и сборка массива из базы."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session as DbSession

from socmon.analysis.dataset import Record
from socmon.analysis.wave import load_valid_records, summarize
from socmon.analysis.weighting import Margin
from socmon.core.engine import InterviewSession, Utterance
from socmon.core.questionnaire import Questionnaire
from socmon.db.storage import CallContext, save_interview
from socmon.domain.validity import Validity
from tests.conftest import FakeClock, needs_db
from tests.test_engine import HAPPY_PATH, drive
from tests.test_storage import context  # noqa: F401  фикстура

MARGINS = [
    Margin("gender", {"Мужской": 0.46, "Женский": 0.54}),
    Margin("age_group", {"18-34": 0.30, "35-54": 0.36, "55+": 0.34}),
    Margin("territory", {"Северный": 0.3, "Центральный": 0.4, "Южный": 0.3}),
]

CORE = {
    "life_situation": 5, "material_stability": 4, "anxiety": 7,
    "fairness": 5, "institutional_trust": 4, "perceived_conflict": 6,
    "collective_action": 2,
}


def synthetic(n: int = 600) -> list[Record]:
    genders = ["Мужской", "Женский"]
    ages = ["18-34", "35-54", "55+"]
    territories = ["Северный", "Центральный", "Южный"]
    return [
        Record(
            id=str(i),
            strata={
                "gender": genders[i % 2],
                "age_group": ages[i % 3],
                # Другой шаг, чтобы территория не оказалась точной копией
                # возраста: на полностью коллинеарных признаках противоречивые
                # эталоны нерешаемы в принципе.
                "territory": territories[(i // 5) % 3],
            },
            answers=dict(CORE),
        )
        for i in range(n)
    ]


def test_summary_weights_then_scores() -> None:
    summary = summarize(synthetic(), MARGINS, wave_id=uuid.uuid4(), target_min=500)

    assert summary.meets_target
    assert summary.weights.converged
    # Число выводится из ответов, а не подсмотрено в выводе:
    #   E = 10-5 = 5, F = 10-4 = 6, A = 7, J = 10-5 = 5,
    #   D = 10-4 = 6, C = 6, P(код 2) = 20/3
    expected = (5 + 6 + 7 + 5 + 6 + 6 + 20 / 3) / 7
    assert summary.index.aggregate == pytest.approx(expected)
    assert set(summary.index.profile) == {"E", "F", "A", "J", "D", "C", "P"}


def test_conflicting_margins_on_collinear_variables_do_not_converge() -> None:
    """Если два признака делят выборку одинаково, а эталоны по ним разные,
    решения не существует. Система обязана это показать, а не изобразить
    сходимость."""
    collinear = [
        Record(
            id=str(i),
            strata={
                "gender": ["Мужской", "Женский"][i % 2],
                "age_group": ["18-34", "35-54", "55+"][i % 3],
                "territory": ["Северный", "Центральный", "Южный"][i % 3],
            },
            answers=dict(CORE),
        )
        for i in range(600)
    ]
    summary = summarize(collinear, MARGINS, wave_id=uuid.uuid4(), target_min=500)
    assert not summary.weights.converged
    assert any("не сошлось" in n for n in summary.notes)


def test_short_wave_is_flagged_first() -> None:
    """ТЗ §5.2: минимум 1500 валидных интервью. Недобор — первое, что должен
    увидеть читатель отчёта, а не сноска в конце."""
    summary = summarize(synthetic(300), MARGINS, wave_id=uuid.uuid4())
    assert not summary.meets_target
    assert "не добирает целевой объём" in summary.notes[0]


def test_weighting_never_looks_at_the_answers() -> None:
    """§5.1 запрещает подгонять результат. Гарантия структурная: два массива
    с одинаковой структурой и разными ответами получают одинаковые веса."""
    calm = synthetic()
    tense = [
        Record(id=r.id, strata=r.strata, answers={**CORE, "anxiety": 10})
        for r in calm
    ]
    left = summarize(calm, MARGINS, wave_id=uuid.uuid4(), target_min=500)
    right = summarize(tense, MARGINS, wave_id=uuid.uuid4(), target_min=500)

    assert [r.weight for r in left.records] == [r.weight for r in right.records]
    assert right.index.aggregate > left.index.aggregate


def test_notes_gather_every_limitation() -> None:
    summary = summarize(synthetic(300), MARGINS, wave_id=uuid.uuid4())
    assert summary.notes
    assert all(isinstance(n, str) for n in summary.notes)


# ------------------------------------------------------------ сборка из базы

pytestmark_db = pytest.mark.usefixtures("engine")


def _record_interview(
    session: DbSession, core: Questionnaire, ctx: CallContext, script: dict[str, str]
) -> None:
    interview = InterviewSession(core, clock=FakeClock(step_sec=8.0))
    drive(interview, script)
    save_interview(session, interview, ctx)


@needs_db
@pytest.mark.usefixtures("engine")
def test_dataset_is_assembled_from_valid_interviews(
    session: DbSession, core: Questionnaire, context: CallContext  # noqa: F811
) -> None:
    _record_interview(session, core, context, HAPPY_PATH)

    records = load_valid_records(
        session, context.wave_id, core, ["gender", "age_group", "territory"]
    )
    assert len(records) == 1
    record = records[0]
    assert record.strata == {
        "gender": "Женский", "age_group": "35-54", "territory": "Центральный"
    }
    assert record.answers["anxiety"] == 8
    assert record.id != str(context.contact_id)


@needs_db
@pytest.mark.usefixtures("engine")
def test_invalid_interviews_stay_out_of_the_dataset(
    session: DbSession, core: Questionnaire, context: CallContext  # noqa: F811
) -> None:
    interview = InterviewSession(core, clock=FakeClock())
    interview.begin()
    interview.submit(Utterance.said("нет, не хочу"))
    stored = save_interview(session, interview, context)
    assert stored.validity is Validity.INVALID

    assert load_valid_records(session, context.wave_id, core, ["gender"]) == []


def test_unknown_strata_key_is_refused(core: Questionnaire) -> None:
    from socmon.analysis.wave import _strata_labels

    with pytest.raises(ValueError, match="взвешивать по этим признакам нечем"):
        _strata_labels(core, ["income_bracket"])
