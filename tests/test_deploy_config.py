"""Конфиги Asterisk против констант в коде.

Живой Asterisk для этого не нужен, а польза прямая: классическая поломка
таких систем — переименовали Stasis-приложение или переменную канала в одном
месте, и звонки молча перестали связываться с базой. Компилятор такое
не ловит, а эти тесты ловят.

Отдельно проверяется то, что ломает не сборку, а сам замысел: AMD после
Stasis, direct_media без записи, ARI наружу.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from socmon.telephony.asterisk import (
    ATTEMPT_VARIABLE,
    DEFAULT_APP,
    DEFAULT_OUTBOUND_CONTEXT,
    DEFAULT_USER,
)

CONFIGS = Path(__file__).resolve().parents[1] / "deploy" / "asterisk"


def read(name: str) -> str:
    return (CONFIGS / name).read_text(encoding="utf-8")


def section(text: str, name: str) -> str:
    """Вырезать секцию конфига по её заголовку."""
    match = re.search(rf"^\[{re.escape(name)}\](.*?)(?=^\[|\Z)", text, re.S | re.M)
    if match is None:
        raise AssertionError(f"нет секции [{name}]")
    return match.group(1)


def meaningful(text: str) -> list[str]:
    """Строки без комментариев и пустот."""
    return [
        line.strip() for line in text.splitlines()
        if line.strip() and not line.strip().startswith(";")
    ]


# ------------------------------------------------------------ связь с кодом


def test_stasis_application_name_matches_the_code() -> None:
    assert f"STASIS_APP = {DEFAULT_APP}" in read("extensions.conf")


def test_outbound_context_matches_the_code() -> None:
    """Диалер набирает именно в этот контекст. Разойдутся — звонки уйдут
    в никуда, и Asterisk честно скажет «extension not found»."""
    assert f"[{DEFAULT_OUTBOUND_CONTEXT}]" in read("extensions.conf")


def test_ari_user_matches_the_code() -> None:
    assert f"[{DEFAULT_USER}]" in read("ari.conf")


def test_attempt_variable_is_exported_in_events() -> None:
    """Без этого события приходят «голыми», и связать их с попыткой можно
    только по реестру каналов в памяти — а он не переживает перезапуск."""
    channelvars = re.search(r"channelvars\s*=\s*(.+)", read("ari.conf"))
    assert channelvars is not None
    exported = {v.strip() for v in channelvars.group(1).split(",")}
    assert ATTEMPT_VARIABLE in exported
    assert "AMDSTATUS" in exported


def test_dialplan_passes_the_attempt_and_the_amd_verdict_to_stasis() -> None:
    """Адаптер читает их из args[0] и args[1] — см. translate()."""
    outbound = section(read("extensions.conf"), DEFAULT_OUTBOUND_CONTEXT)
    stasis = [line for line in meaningful(outbound) if "Stasis(" in line]
    assert len(stasis) == 1
    assert f"${{{ATTEMPT_VARIABLE}}}" in stasis[0]
    assert "${AMDSTATUS}" in stasis[0]


# ------------------------------------------------------- порядок в дилплане


def test_amd_runs_before_stasis() -> None:
    """Главная ловушка. AMD — приложение дилплана: если завести звонок сразу
    в Stasis, детектор не выполнится вовсе, и робот начнёт зачитывать анкету
    голосовой почте, прожигая минуты связи."""
    lines = meaningful(section(read("extensions.conf"), DEFAULT_OUTBOUND_CONTEXT))
    amd = next(i for i, line in enumerate(lines) if "AMD()" in line)
    stasis = next(i for i, line in enumerate(lines) if "Stasis(" in line)
    assert amd < stasis


def test_channel_is_answered_before_detection() -> None:
    lines = meaningful(section(read("extensions.conf"), DEFAULT_OUTBOUND_CONTEXT))
    answer = next(i for i, line in enumerate(lines) if "Answer()" in line)
    amd = next(i for i, line in enumerate(lines) if "AMD()" in line)
    assert answer < amd


def test_uncertain_detection_is_treated_as_a_human() -> None:
    """Цена ошибки несимметрична: принять человека за автоответчик —
    потерянное интервью, обратное — одна испорченная запись."""
    outbound = section(read("extensions.conf"), DEFAULT_OUTBOUND_CONTEXT)
    assert "NOTSURE" in outbound
    assert "Set(AMDSTATUS=HUMAN)" in outbound


def test_every_context_ends_with_hangup() -> None:
    """Оставлять абонента в тишине нельзя ни в одной ветке."""
    text = read("extensions.conf")
    for name in ("socmon-outbound", "socmon-inbound", "socmon-operator"):
        assert "Hangup()" in section(text, name), name


def test_callback_number_informs_the_caller() -> None:
    """ТЗ §9.1: голосовое информирование при обратном звонке, сообщение
    о цели исследования и контактах."""
    inbound = section(read("extensions.conf"), "socmon-inbound")
    assert "Playback(socmon/callback-info)" in inbound
    assert "Stasis(" in inbound  # журнал обращений ведёт управляющий процесс


# ------------------------------------------------------------------ запись


def test_media_is_not_allowed_to_bypass_us() -> None:
    """При direct_media=yes голос пошёл бы мимо Asterisk, и записывать стало
    бы нечего — а запись это ТЗ §10.2 и основание для всей аналитики."""
    text = read("pjsip.conf")
    assert re.search(r"^\s*direct_media\s*=\s*no", text, re.M)
    assert not re.search(r"^\s*direct_media\s*=\s*yes", text, re.M)


def test_only_narrowband_codecs_are_allowed() -> None:
    """Телефонный канал 8 кГц — ровно то, подо что озвучена анкета.
    Широкополосные кодеки дали бы транскодинг на каждой линии без выигрыша."""
    trunk = section(read("pjsip.conf"), "trunk")
    assert "disallow = all" in trunk
    allowed = set(re.findall(r"allow\s*=\s*(\w+)", trunk)) - {"all"}
    assert allowed == {"alaw", "ulaw"}


def test_rtp_range_covers_the_planned_concurrency() -> None:
    """60-100 линий плюс по два снуп-канала записи на каждую."""
    text = read("rtp.conf")
    start = int(re.search(r"rtpstart\s*=\s*(\d+)", text).group(1))  # type: ignore[union-attr]
    end = int(re.search(r"rtpend\s*=\s*(\d+)", text).group(1))  # type: ignore[union-attr]
    assert end - start >= 2000


# --------------------------------------------------------------- безопасность


def test_ari_listens_only_on_the_loopback() -> None:
    """ARI даёт полный контроль над АТС, включая прослушивание каналов.
    Открытый наружу порт — это сразу и утечка ПДн, и чужие звонки за наш счёт."""
    assert re.search(r"^\s*bindaddr\s*=\s*127\.0\.0\.1", read("http.conf"), re.M)


def test_no_secret_is_written_into_the_configs() -> None:
    """Конфиги лежат в репозитории, секреты — в окружении (ТЗ §14.2, §14.3)."""
    for name in ("ari.conf", "pjsip.conf"):
        for line in meaningful(read(name)):
            if re.match(r"^(password|secret)\s*=", line):
                value = line.split("=", 1)[1].strip()
                assert value.startswith("${"), f"{name}: пароль открытым текстом"


@pytest.mark.parametrize("module", ["chan_sip.so", "chan_iax2.so", "app_voicemail.so"])
def test_unused_modules_are_not_loaded(module: str) -> None:
    """Каждый лишний модуль — лишняя поверхность атаки на машине,
    где лежат записи разговоров и телефонные номера."""
    assert f"noload = {module}" in read("modules.conf")


def test_builtin_modules_are_never_noloaded() -> None:
    """Начиная с Asterisk 22 manager.so встроенный, и `noload` для него
    валит запуск целиком: «is a built-in module, you cannot specify noload».
    Поймано живой АТС на первом же старте."""
    builtin = {"manager.so", "loader.so", "logger.so", "config.so"}
    listed = set(re.findall(r"noload\s*=\s*(\S+)", read("modules.conf")))
    assert not (listed & builtin)


def test_ami_is_disabled_the_supported_way() -> None:
    """Выгрузить нельзя — значит выключаем в manager.conf. AMI это второй
    управляющий интерфейс с полным доступом к АТС, и он нам не нужен."""
    text = read("manager.conf")
    assert re.search(r"^\s*enabled\s*=\s*no", text, re.M)
    assert re.search(r"^\s*webenabled\s*=\s*no", text, re.M)


def test_all_configs_are_present() -> None:
    expected = {
        "amd.conf", "ari.conf", "extensions.conf", "http.conf",
        "manager.conf", "modules.conf", "pjsip.conf", "rtp.conf",
    }
    assert {p.name for p in CONFIGS.glob("*.conf")} == expected
