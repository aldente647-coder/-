"""Жизненный цикл волны и проверка готовности к запуску."""

from __future__ import annotations

from dataclasses import replace

import pytest

from socmon.domain.wave import (
    WaveError,
    WaveSetup,
    WaveStatus,
    can_transition,
    readiness,
    transition,
)

READY = WaveSetup(
    has_questionnaire=True,
    questionnaire_valid=True,
    voice_missing=0,
    voice_stale=0,
    has_reference_distributions=True,
    caller_ids_active=5,
    contacts_available=90_000,
    target_min=1500,
    calling_window_configured=True,
)


# ---------------------------------------------------------------- переходы


def test_normal_life_of_a_wave() -> None:
    status = WaveStatus.DRAFT
    for step in (WaveStatus.ACTIVE, WaveStatus.PAUSED, WaveStatus.ACTIVE,
                 WaveStatus.CLOSED):
        status = transition(status, step)
    assert status is WaveStatus.CLOSED


def test_closed_wave_never_reopens() -> None:
    """По закрытой волне уже сдан оперативный пакет (ТЗ §13.1). Дописанное
    в неё интервью сделает выгрузку Заказчика неверной задним числом."""
    for target in (WaveStatus.ACTIVE, WaveStatus.PAUSED, WaveStatus.DRAFT):
        with pytest.raises(WaveError):
            transition(WaveStatus.CLOSED, target)


def test_draft_cannot_be_paused() -> None:
    with pytest.raises(WaveError, match="не может перейти"):
        transition(WaveStatus.DRAFT, WaveStatus.PAUSED)


def test_active_wave_cannot_go_back_to_draft() -> None:
    """Иначе анкету можно было бы поправить посреди волны, и половина
    интервью оказалась бы по другой редакции вопросов."""
    with pytest.raises(WaveError):
        transition(WaveStatus.ACTIVE, WaveStatus.DRAFT)


def test_repeating_the_current_status_is_harmless() -> None:
    assert transition(WaveStatus.ACTIVE, WaveStatus.ACTIVE) is WaveStatus.ACTIVE


def test_draft_may_be_abandoned() -> None:
    assert transition(WaveStatus.DRAFT, WaveStatus.CLOSED) is WaveStatus.CLOSED


def test_can_transition_matches_transition() -> None:
    for current in WaveStatus:
        for target in WaveStatus:
            if current is target:
                continue
            allowed = can_transition(current, target)
            if allowed:
                assert transition(current, target) is target
            else:
                with pytest.raises(WaveError):
                    transition(current, target)


# --------------------------------------------------------------- готовность


def test_fully_prepared_wave_is_ready() -> None:
    check = readiness(READY)
    assert check.ready
    assert check.blockers == ()


def test_stale_voice_blocks_the_launch() -> None:
    """Самый опасный случай: сценарий выглядит озвученным, робот читает
    прошлую редакцию вопроса, а в массив пишется код по новой."""
    check = readiness(replace(READY, voice_stale=2))
    assert not check.ready
    assert any("формулировка изменилась" in c.detail for c in check.blockers)


GAPS: list[tuple[WaveSetup, str]] = [
    (replace(READY, has_questionnaire=False), "не привязана версия анкеты"),
    (replace(READY, questionnaire_valid=False), "не связен"),
    (replace(READY, voice_missing=3), "не озвучено реплик"),
    (replace(READY, has_reference_distributions=False), "взвешивание"),
    (replace(READY, caller_ids_active=0), "нет активных исходящих"),
    (replace(READY, calling_window_configured=False), "время звонков"),
]


@pytest.mark.parametrize(("setup", "expected"), GAPS)
def test_each_gap_blocks_the_launch_and_explains_itself(
    setup: WaveSetup, expected: str
) -> None:
    """Система обязана показать, чего не хватает, а не молча не поехать."""
    check = readiness(setup)
    assert not check.ready
    assert any(expected in c.detail for c in check.blockers)


def test_single_caller_id_warns_but_does_not_block() -> None:
    """С одного номера операторы связи блокируют быстро, но запускать
    волну это не мешает — решение за Заказчиком."""
    check = readiness(replace(READY, caller_ids_active=1))
    assert check.ready
    assert any("блокируют быстро" in c.detail for c in check.warnings)


def test_thin_contact_base_warns_but_does_not_block() -> None:
    """«Контактов больше целевого объёма» — наивный критерий: при достижимости
    робо-обзвона на 1500 интервью нужны десятки тысяч номеров."""
    check = readiness(replace(READY, contacts_available=2000))
    assert check.ready
    assert any("столько номеров нужно" in c.detail for c in check.warnings)


def test_contact_requirement_is_derived_from_reachability() -> None:
    from socmon.domain.wave import CONTACTS_PER_INTERVIEW

    barely = readiness(replace(
        READY, contacts_available=READY.target_min * CONTACTS_PER_INTERVIEW
    ))
    assert not barely.warnings or all(
        "столько номеров нужно" not in c.detail for c in barely.warnings
    )


def test_every_check_carries_an_explanation_when_it_fails() -> None:
    empty = WaveSetup(
        has_questionnaire=False, questionnaire_valid=False,
        voice_missing=10, voice_stale=5, has_reference_distributions=False,
        caller_ids_active=0, contacts_available=0, target_min=1500,
        calling_window_configured=False,
    )
    for check in readiness(empty).checks:
        if not check.passed:
            assert check.detail, f"пункт «{check.name}» не объясняет, что не так"
