"""Командная строка: проверка анкеты и прогон интервью вручную.

`check` — то, чем анкета проверяется до запуска волны (ТЗ §7.3: предпросмотр
и тестирование анкеты до запуска). `play` даёт прощёлкать сценарий голосом
своих рук, без телефонии.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from socmon.core.engine import Action, InterviewSession, Turn, Utterance
from socmon.core.questionnaire import (
    OpenNode,
    Questionnaire,
    QuestionnaireError,
    ScaleNode,
    ScreenerNode,
    SingleNode,
)
from socmon.voice.synth import TELEPHONY_SAMPLE_RATE

DEFAULT = Path(__file__).parent / "questionnaires" / "wave-base.yaml"


def _load(path: Path) -> Questionnaire:
    try:
        return Questionnaire.from_yaml(path)
    except QuestionnaireError as exc:
        print(f"Анкета не прошла проверку:\n{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def cmd_check(path: Path) -> int:
    q = _load(path)
    counts: dict[str, int] = {}
    for node in q.nodes:
        counts[node.kind] = counts.get(node.kind, 0) + 1

    questions = sum(
        1 for n in q.nodes if isinstance(n, SingleNode | ScaleNode | OpenNode | ScreenerNode)
    )
    missing_audio = [n.id for n in q.nodes if n.text and not n.audio_key]

    print(f"Анкета:      {q.name}")
    print(f"Волна:       {q.wave}")
    print(f"Версия:      {q.version_id}")
    breakdown = ", ".join(f"{kind}={n}" for kind, n in sorted(counts.items()))
    print(f"Узлов:       {len(q.nodes)}  ({breakdown})")
    print(f"Вопросов:    {questions}")
    print(f"Ядро:        {len(q.core_keys)} показателей — {', '.join(q.core_keys)}")
    print(f"Лимит:       {q.policy.target_duration_sec} с, переспросов {q.policy.max_repeats}")
    print("Граф:        связен, все пути приводят к завершению")

    if missing_audio:
        # Кампанию нельзя запускать, пока сценарий не озвучен полностью:
        # робот обязан произносить утверждённую формулировку дословно (ТЗ §8.1).
        print(f"\nНе озвучено ({len(missing_audio)}): {', '.join(missing_audio)}")
        print("Волну запускать нельзя, пока не назначены аудиофайлы.")
        return 1
    print("Озвучка:     все реплики имеют аудио")
    return 0


def _hint(q: Questionnaire, node_id: str) -> str:
    node = q.node(node_id)
    match node:
        case ScaleNode():
            return f"[число {node.low}-{node.high}]"
        case SingleNode() | ScreenerNode():
            return "[" + " / ".join(o.label for o in node.options) + "]"
        case OpenNode():
            return "[свободный ответ]"
        case _:
            return "[да / нет]"


def _player() -> str | None:
    """Чем проигрывать WAV в этой системе."""
    import shutil

    for binary in ("afplay", "paplay", "aplay"):
        found = shutil.which(binary)
        if found:
            return found
    return None


def _show(turn: Turn, media: Path | None = None, player: str | None = None) -> None:
    import subprocess

    for prompt in turn.prompts:
        text = " ".join(prompt.text.split())
        if not text:
            continue
        print(f"  РОБОТ: {text}")
        if media is None or player is None:
            continue
        audio = media / f"{prompt.audio_key or prompt.node_id}.wav"
        if audio.exists():
            subprocess.run([player, str(audio)], check=False, capture_output=True)


def cmd_play(path: Path, media: Path | None = None) -> int:
    q = _load(path)
    session = InterviewSession(q)
    print(f"{q.name} · версия {q.version_id}")

    player = None
    if media is not None:
        from socmon.voice.render import verify

        player = _player()
        check = verify(q, media)
        if not check.ready:
            print(f"Озвучка неполная ({media}/): " + "; ".join(check.problems))
            print("Реплики будут только напечатаны. Соберите: socmon voice render\n")
            media = None
        elif player is None:
            print("Не нашёл проигрыватель WAV — реплики будут только напечатаны.\n")
            media = None
        else:
            print(f"Голос из {media}/ через {Path(player).name}\n")

    print("Отвечайте как респондент. Ctrl+D — положить трубку.\n")

    turn = session.begin()
    _show(turn, media, player)
    while turn.awaiting is not None:
        try:
            reply = input(f"  ВЫ {_hint(q, turn.awaiting)}: ").strip()
        except EOFError:
            print()
            turn = session.submit(Utterance.dropped())
            _show(turn, media, player)
            break
        turn = session.submit(Utterance.said(reply) if reply else Utterance.quiet())
        if turn.action is Action.REPEAT:
            print("  (переспрашиваю)")
        _show(turn, media, player)

    assert session.outcome is not None
    print(f"\nИсход:       {session.outcome.value}")
    print(f"Длительность: {session.duration_sec:.0f} с")
    substantive = sum(1 for a in session.answers.values() if a.is_substantive)
    print(f"Ответов:     {len(session.answers)} (содержательных {substantive})")
    for answer in session.answers.values():
        value = answer.text if answer.code is None else answer.code
        print(f"  {answer.node_id:<20} {value}")
    return 0


def cmd_wave(size: int, seed: int) -> int:
    """Сводка по волне на синтетическом массиве.

    Показывает ровно тот блок, который уходит Заказчику по ТЗ §13.2:
    структура выборки против эталона, диагностика весов, индекс
    с компонентным профилем и ограничения интерпретации.
    """
    import random
    import uuid

    from socmon.analysis.dataset import Record
    from socmon.analysis.wave import summarize
    from socmon.analysis.weighting import Margin

    rng = random.Random(seed)
    genders = ["Мужской", "Женский"]
    ages = ["18-34", "35-54", "55+"]
    territories = ["Северный", "Центральный", "Южный"]
    records = [
        Record(
            id=str(i),
            strata={
                "gender": rng.choice(genders),
                "age_group": rng.choice(ages),
                "territory": rng.choice(territories),
            },
            answers={
                "life_situation": rng.randint(2, 8),
                "material_stability": rng.randint(1, 7),
                "anxiety": rng.randint(3, 10),
                "fairness": rng.randint(1, 8),
                "institutional_trust": rng.randint(1, 8),
                "perceived_conflict": rng.randint(2, 9),
                "collective_action": rng.randint(1, 4),
            },
        )
        for i in range(size)
    ]
    margins = [
        Margin("gender", {"Мужской": 0.46, "Женский": 0.54}),
        Margin("age_group", {"18-34": 0.30, "35-54": 0.36, "55+": 0.34}),
        Margin("territory", {"Северный": 0.30, "Центральный": 0.42, "Южный": 0.28}),
    ]

    summary = summarize(records, margins, wave_id=uuid.uuid4(), target_min=1500)
    diagnostics, index = summary.weights, summary.index

    print(f"Валидных интервью:   {summary.valid_interviews}"
          f"{'' if summary.meets_target else '  ← ниже минимума 1500'}")
    print(f"Эффективный объём:   {diagnostics.effective_n:.0f} "
          f"(эффект дизайна {diagnostics.design_effect:.3f})")
    print(f"Веса:                {diagnostics.min_weight:.3f}–{diagnostics.max_weight:.3f}, "
          f"экстремальных {diagnostics.extreme_share:.1%}")
    print(f"Сходимость:          {'да' if diagnostics.converged else 'нет'}, "
          f"итераций {diagnostics.iterations}, "
          f"отклонение от эталона {diagnostics.max_margin_deviation:.2e}")

    print("\nСтруктура выборки (невзвеш. → взвеш. → эталон):")
    for variable, targets in diagnostics.reference.items():
        for category in sorted(targets):
            print(f"  {variable:<12} {category:<14}"
                  f"{diagnostics.unweighted[variable].get(category, 0):>7.1%}"
                  f"{diagnostics.achieved[variable].get(category, 0):>9.1%}"
                  f"{targets[category]:>9.1%}")

    print(f"\nИндекс социального напряжения: {index.aggregate:.2f} из 10")
    if index.individual is not None:
        print(f"  по индивидуальным данным:    {index.individual.formatted()}")
    print(f"  полных наблюдений:           {index.complete_cases} из {index.n}")
    print(f"  методика расчёта:            {index.recipe_hash}")

    print("\nКомпонентный профиль:")
    for component in index.components:
        bar = "█" * round(component.value * 3)
        print(f"  {component.key}  {component.value:>5.2f}  {bar:<30} {component.name}")

    if summary.notes:
        print("\nОграничения интерпретации:")
        for note in summary.notes:
            print(f"  • {note}")
    return 0


def _demo_package(size: int, seed: int, wave_code: str):  # type: ignore[no-untyped-def]
    """Синтетическая волна для демонстрации выгрузок и отчёта."""
    import random
    import uuid
    from datetime import UTC, datetime

    from socmon.analysis.dataset import Record
    from socmon.analysis.outcomes import CallStatistics, Dispositions, response_rates
    from socmon.analysis.wave import summarize
    from socmon.analysis.weighting import Margin
    from socmon.core.codes import AaporCode, SpecialCode
    from socmon.export import codebook, package

    rng = random.Random(seed)
    questionnaire = _load(DEFAULT)
    book = codebook.build(questionnaire)

    def maybe(value: int, refusal_rate: float = 0.04) -> int:
        return int(SpecialCode.DONT_KNOW) if rng.random() < refusal_rate else value

    labels = {
        "gender": ["Мужской", "Женский"],
        "age_group": ["18-34", "35-54", "55+"],
        "territory": ["Северный", "Центральный", "Южный"],
    }

    def demography() -> tuple[dict[str, int], dict[str, str]]:
        """Соцдем задаётся один раз: код идёт в массив, подпись — в структуру
        выборки. Дублировать их порознь — верный способ однажды разойтись."""
        codes = {key: rng.randrange(len(names)) for key, names in labels.items()}
        return (
            {key: code + 1 for key, code in codes.items()},
            {key: labels[key][code] for key, code in codes.items()},
        )

    records = []
    for _ in range(size):
        codes, strata = demography()
        records.append(Record(
            id=str(uuid.uuid4()),
            strata=strata,
            answers={
                "life_situation": maybe(rng.randint(2, 8)),
                "material_stability": maybe(rng.randint(1, 7)),
                "anxiety": maybe(rng.randint(3, 10)),
                "fairness": maybe(rng.randint(1, 8)),
                "institutional_trust": maybe(rng.randint(1, 8)),
                "perceived_conflict": maybe(rng.randint(2, 9)),
                "authorities_rating": maybe(rng.randint(1, 9), 0.09),
                "collective_action": maybe(rng.randint(1, 4)),
                "local_problem": maybe(rng.randint(1, 5), 0.06),
                **codes,
            },
        ))

    margins = [
        Margin("gender", {"Мужской": 0.46, "Женский": 0.54}),
        Margin("age_group", {"18-34": 0.30, "35-54": 0.36, "55+": 0.34}),
        Margin("territory", {"Северный": 0.30, "Центральный": 0.42, "Южный": 0.28}),
    ]
    summary = summarize(records, margins, wave_id=uuid.uuid4(), target_min=1500)

    dispositions = Dispositions.from_codes(
        [AaporCode.COMPLETE] * size
        + [AaporCode.PARTIAL] * (size // 8)
        + [AaporCode.REFUSAL] * (size * 2)
        + [AaporCode.NON_CONTACT] * (size // 2)
        + [AaporCode.UNKNOWN_ELIGIBILITY] * (size * 45)
        + [AaporCode.NOT_ELIGIBLE] * (size // 6)
        + [AaporCode.OUT_OF_SAMPLE] * (size)
    )
    statistics = CallStatistics(
        numbers_dialled=dispositions.total,
        attempts_made=int(dispositions.total * 3.4),
        dispositions=dispositions,
        rates=response_rates(dispositions),
        valid_interviews=size,
        invalid_interviews=size // 5,
        exclusion_reasons={"not_completed": size // 8, "no_consent": size // 12},
    )
    return package.build(
        summary, book,
        wave_code=wave_code,
        call_statistics=statistics,
        now=datetime.now(UTC),
        methodology_notes=(
            "Порог минимальной длительности интервью не применялся: "
            "он утверждается по результатам пилота (ТЗ §5.3).",
        ),
    )


def cmd_export(out: Path, size: int, seed: int, wave_code: str, pdf: bool) -> int:
    """Собрать оперативный пакет и аналитический отчёт."""
    from socmon.export import report, writers

    package = _demo_package(size, seed, wave_code)
    written = writers.write_package(out, package)
    written.append(report.write_report(out / f"wave-{wave_code}-отчёт.docx", package))

    print(f"Волна {wave_code}: {package.summary.valid_interviews} валидных интервью, "
          f"индекс {package.summary.index.aggregate:.2f}")
    print(f"Записано в {out}/")
    for path in written:
        print(f"  {path.name:<44} {path.stat().st_size / 1024:>8.1f} КБ")

    if pdf:
        try:
            produced = report.to_pdf(written[-1])
        except report.PdfConversionError as exc:
            print(f"\nPDF не собран: {exc}", file=sys.stderr)
            return 1
        print(f"  {produced.name:<44} {produced.stat().st_size / 1024:>8.1f} КБ")
    return 0


#: Каталог озвучки. Тот же источник, что и у веб-приложения: на сервере
#: библиотека лежит не рядом с кодом, и «media» в текущем каталоге — это
#: устаревшая копия. Проверка по ней отвечает «волну запускать нельзя»
#: на полностью озвученной анкете, и наоборот.
MEDIA = Path(os.environ.get("SOCMON_MEDIA_DIR", "media"))


def cmd_voice_render(
    path: Path, out: Path, voice: str, rate: int, force: bool, *,
    model_id: str = "v4_ru", model_path: Path | None = None, natural: bool = False,
) -> int:
    """Озвучить анкету (ТЗ §8.1, §8.3)."""
    from socmon.voice.render import render
    from socmon.voice.synth import RUSSIAN_VOICES, SileroSynthesizer

    questionnaire = _load(path)
    print(f"Анкета {questionnaire.name} · версия {questionnaire.version_id}")
    print(f"Голос {voice} ({RUSSIAN_VOICES.get(voice, '?')}), {rate} Гц → {out}/\n")

    report = render(
        questionnaire, SileroSynthesizer(
            model_id=model_id, model_path=model_path, natural=natural,
        ), out,
        voice=voice, sample_rate=rate, force=force,
    )
    for key in report.rendered:
        entry = report.manifest.entries[key]
        print(f"  озвучено   {key:<28} {entry.duration_sec:>6.2f} с")
    if report.reused:
        print(f"  без изменений: {len(report.reused)} реплик")

    # Считаем только реплики этой анкеты. Манифест общий на все волны —
    # сумма по нему дала бы длительность всей библиотеки рядом с числом
    # реплик одной волны, и цифры перестали бы сходиться друг с другом.
    own = set(report.rendered) | set(report.reused)
    seconds = sum(
        e.duration_sec for key, e in report.manifest.entries.items() if key in own
    )
    print(f"\nВсего реплик: {report.total}, суммарно {seconds / 60:.1f} мин звучания")
    print(f"Манифест: {out}/voice.json")
    return 0


def cmd_voice_qa(media: Path, whisper: Path, prune: bool, gender: str) -> int:
    """Прогнать библиотеку через независимое распознавание (ТЗ §11.3:
    автоматическая проверка перед выпуском записей в звонок)."""
    from socmon.voice.delivery import Catalogue
    from socmon.voice.qa import audit, whisper_transcriber
    from socmon.voice.render import speakable

    texts: dict[str, str] = {}
    for path in sorted((DEFAULT.parent).glob("wave-*.yaml")):
        for node in speakable(_load(path)):
            texts[node.audio_key or node.id] = node.text
    for phrase in Catalogue.from_yaml().phrases():
        texts[phrase.key] = phrase.spoken(gender)  # type: ignore[arg-type]

    report = audit(media, texts, whisper_transcriber(whisper), prune=prune)
    report.save(media / "qa.json")
    for check in report.checks:
        if not check.exact:
            print(f"  ✗ {check.file:32} пропущено {list(check.missing)} лишнее {list(check.extra)}")
    print(f"\nПроверено файлов: {len(report.checks)}, совпало: {report.passed}")
    if report.pruned:
        print(f"Убрано дублей: {len(report.pruned)}")
    if report.failed_primary:
        print("НЕТ ГОДНОЙ ЗАПИСИ: " + ", ".join(report.failed_primary))
        return 1
    return 0


def cmd_voice_check(path: Path, media: Path) -> int:
    """Готова ли озвучка к запуску волны."""
    from socmon.voice.render import verify

    questionnaire = _load(path)
    check = verify(questionnaire, media)

    if check.ready:
        print(f"Озвучка полная. Анкета {questionnaire.version_id}, каталог {media}/")
        from socmon.voice.render import MANIFEST_NAME, VoiceManifest, estimate_duration

        estimate = estimate_duration(
            questionnaire, VoiceManifest.load(media / MANIFEST_NAME)
        )
        verdict = "укладывается" if estimate.within_limit else "НЕ УКЛАДЫВАЕТСЯ"
        print(f"\nОценка длительности интервью: {estimate.note}")
        print(f"Лимит ТЗ §3.3 — 10 минут: {verdict}")
        print("Ответ и пауза — допущения до пилота, их надо заменить замером.")
    else:
        print(f"Волну запускать нельзя. Анкета {questionnaire.version_id}:")
    for problem in check.problems:
        print(f"  • {problem}")
    if check.version_mismatch and not check.missing:
        print("  • манифест собран для другой версии анкеты — перепроверьте формулировки")
    return 0 if check.ready else 1


def cmd_voice_list() -> int:
    from socmon.voice.catalog import CLONED, VOICE_OPTIONS

    print("Голоса для русского языка:")
    for option in VOICE_OPTIONS:
        model = "Qwen3-TTS" if option.id in CLONED else "Silero"
        print(f"  {option.id:<10} {option.name} · {option.gender} · {model}")
    return 0


def cmd_simulate(path: Path, numbers: int, seed: int, dropout: float) -> int:
    """Прогнать волну целиком без Asterisk.

    Полный цикл: диалер набирает, движок телефонии отвечает по вероятностям,
    рантайм ведёт интервью, ответы сохраняются. Всё, что можно проверить
    до аренды транка.

    Доли ответивших и бросивших трубку — допущения, а не измерения.
    Заменить их обязан пилот (ТЗ §15).
    """
    import asyncio
    import random
    import uuid
    from collections import Counter
    from datetime import UTC, datetime, timedelta

    from socmon.core.engine import InterviewSession
    from socmon.runtime.call import CallRuntime
    from socmon.runtime.dialer import Dialer, DialerPolicy, RoundRobinCallerIds
    from socmon.runtime.simulation import (
        MemoryContactSource,
        MemoryJournal,
        SimulatedProvider,
        sample_claims,
    )
    from socmon.telephony.machine import CallEvent
    from socmon.telephony.provider import CallHandle, ProviderEvent, RecognizedSpeech

    questionnaire = _load(path)
    moment = datetime(2026, 8, 20, 18, 30, tzinfo=UTC)

    def clock() -> datetime:
        nonlocal moment
        moment += timedelta(seconds=4)
        return moment

    rng = random.Random(seed)
    provider = SimulatedProvider(seed=seed)
    source = MemoryContactSource(queue=sample_claims(numbers, rng))
    journal = MemoryJournal()

    runtime = CallRuntime(
        provider, questionnaire, journal,
        operator_endpoint="PJSIP/operator1", clock=clock,
    )
    handles: dict[uuid.UUID, CallHandle] = {}

    def remember(attempt_id: uuid.UUID, handle: CallHandle) -> None:
        handles[attempt_id] = handle
        runtime.register(attempt_id, handle)

    dialer = Dialer(
        provider, source,
        RoundRobinCallerIds([f"+7495000{i:04d}" for i in range(1, 6)]),
        wave_id=uuid.uuid4(), on_originated=remember,
        policy=DialerPolicy(
            max_concurrent=numbers, calls_per_second=1e6, burst=1e6
        ),
        clock=clock,
    )

    async def run() -> int:
        report = await dialer.tick()
        print(
            f"{questionnaire.name} · версия {questionnaire.version_id}\n"
            f"Набрано номеров: {report.originated}, номеров в карусели: 5, "
            f"предохранитель: {report.breaker.value}\n"
        )
        for attempt_id, handle in list(handles.items()):
            for event in provider.outcome_for(handle.attempt_id, clock()):
                await runtime.on_call_event(event)
            await _converse(attempt_id, handle)
            dialer.release(attempt_id, None)
        return report.originated

    async def _converse(attempt_id: uuid.UUID, handle: CallHandle) -> None:
        while True:
            live = runtime.call_of(attempt_id)
            if live is None or live.finished:
                return
            awaiting = live.interview_awaiting()
            if awaiting is None:
                return
            if provider.hangs_up(dropout):
                await runtime.on_call_event(ProviderEvent(
                    attempt_id=handle.attempt_id, event=CallEvent.HANGUP,
                    at=clock(), event_id=f"{handle.attempt_id}:hangup",
                ))
                return
            await runtime.on_speech(RecognizedSpeech(
                attempt_id=handle.attempt_id,
                text=provider.reply_for(awaiting), at=clock(),
            ))

    dialled = asyncio.run(run())

    outcomes: Counter[str] = Counter()
    durations: list[float] = []
    for session in journal.interviews.values():
        assert isinstance(session, InterviewSession)
        outcomes[session.outcome.value if session.outcome else "?"] += 1
        durations.append(session.duration_sec)

    reached = len(journal.interviews)
    print(f"Дозвонились и начали разговор: {reached} из {dialled}")
    print("\nИсходы интервью:")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome:<26} {count:>4}")

    if durations:
        ordered = sorted(durations)
        inside = sum(1 for d in durations if d <= 600)
        print(
            f"\nДлительность: медиана {ordered[len(ordered) // 2] / 60:.1f} мин, "
            f"в лимит 10 мин уложились {inside}/{len(durations)}"
        )
    print(
        f"Записей разговоров: {len(journal.recordings)} "
        f"(раздельные каналы «робот» и «респондент»)"
    )
    print(
        "\nДоли ответивших и бросивших трубку — допущения симуляции, "
        "а не измерения.\nИх обязан заменить пилот (ТЗ §15)."
    )
    return 0


def cmd_serve(host: str, port: int) -> int:
    """Поднять кабинет."""
    import os

    import uvicorn
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from socmon.db.session import DATABASE_URL_VAR
    from socmon.web.app import create_app

    url = os.environ.get(DATABASE_URL_VAR)
    secret = os.environ.get("SOCMON_SECRET_KEY")
    if not url:
        print(f"не задана {DATABASE_URL_VAR}")
        return 1
    if not secret or len(secret) < 32:
        print("не задана SOCMON_SECRET_KEY (не короче 32 символов)")
        print("сгенерировать:  openssl rand -hex 32")
        return 1

    app = create_app(
        sessionmaker(bind=create_engine(url)), secret_key=secret,
        https_only=os.environ.get("SOCMON_ALLOW_HTTP") != "1",
    )
    print(f"Кабинет: http://{host}:{port}")
    if host not in ("127.0.0.1", "localhost"):
        print("ВНИМАНИЕ: кабинет открыт наружу. В нём телефонные номера "
              "и записи разговоров — ходить надо через туннель (ТЗ §14.2).")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def cmd_useradd(login: str, name: str, role: str) -> int:
    """Завести учётную запись."""
    import getpass
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from socmon.db.models import User
    from socmon.db.session import DATABASE_URL_VAR
    from socmon.web.security import ROLE_TITLES, Role, hash_password

    url = os.environ.get(DATABASE_URL_VAR)
    if not url:
        print(f"не задана {DATABASE_URL_VAR}")
        return 1
    try:
        parsed = Role(role)
    except ValueError:
        print(f"неизвестная роль «{role}»; доступны: "
              + ", ".join(r.value for r in Role))
        return 1

    password = getpass.getpass("Пароль (не короче 12 символов): ")
    try:
        digest = hash_password(password)
    except ValueError as exc:
        print(exc)
        return 1

    with Session(create_engine(url)) as db:
        db.add(User(login=login, full_name=name, role=parsed.value, password_hash=digest))
        db.commit()
    print(f"Заведён {login} — {ROLE_TITLES[parsed]}")
    return 0


def cmd_passwd(login: str) -> int:
    """Сменить пароль локально, отозвав все сессии пользователя."""
    import getpass
    import os

    from sqlalchemy.orm import Session

    from socmon.db.session import DATABASE_URL_VAR, make_engine
    from socmon.db.users import change_password

    if not os.environ.get(DATABASE_URL_VAR):
        print(f"не задана {DATABASE_URL_VAR}")
        return 1
    password = getpass.getpass("Новый пароль: ")
    if password != getpass.getpass("Повторите пароль: "):
        print("пароли не совпали")
        return 1
    engine = make_engine()
    try:
        with Session(engine) as db:
            change_password(db, login, password)
            db.commit()
    except ValueError as exc:
        print(exc)
        return 1
    finally:
        engine.dispose()
    print("Пароль изменён, прежние сессии отозваны.")
    return 0


def cmd_contacts_import(path: Path, column: int | None, name: str, seed: str) -> int:
    """Загрузить базу номеров из файла (ТЗ §4.2)."""
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from socmon.db.contacts import import_numbers, read_column
    from socmon.db.session import DATABASE_URL_VAR
    from socmon.domain.identity import PhoneVault

    url = os.environ.get(DATABASE_URL_VAR)
    if not url:
        print(f"не задана {DATABASE_URL_VAR}")
        return 1
    try:
        vault = PhoneVault.from_env()
    except RuntimeError as exc:
        print(exc)
        print("сгенерировать ключи:  openssl rand -hex 32")
        return 1
    if not path.exists():
        print(f"нет файла {path}")
        return 1

    text = path.read_text(encoding="utf-8-sig")
    with Session(create_engine(url)) as db:
        report = import_numbers(
            db, vault, read_column(text, column),
            name=name or path.name, source=f"файл {path.name}", seed=seed or None,
        )
        db.commit()

    print(f"Строк в файле:        {report.total}")
    print(f"Загружено:            {report.loaded}")
    print(f"Дублей в файле:       {report.duplicate_in_file}")
    print(f"Уже были в базе:      {report.duplicate_in_base}")
    print(f"Отклонено:            {report.rejected}")
    print(f"Зерно порядка:        {report.seed}")
    if report.samples:
        print("\nПримеры отклонённых строк:")
        for sample in report.samples[:10]:
            print(f"  строка {sample.line}: {sample.value} — {sample.reason}")
    for note in report.notes:
        print(f"\n• {note}")
    return 0


def cmd_calltest(
    destination: str, caller_id: str, path: Path, media: Path, voice: str = "",
) -> int:
    """Позвонить и провести интервью по-настоящему.

    Проверяет то, что не проверяет ни один тест: узнаёт ли робот живую
    речь через телефонный канал, не говорит ли поверх себя, что делает
    при молчании и при перебивании.
    """
    import asyncio
    import os

    from socmon.asr.gigaam import ONLINE_MODEL, GigaamConfig, GigaamRecognizer
    from socmon.runtime.live import LiveCall
    from socmon.runtime.simulation import MemoryJournal
    from socmon.telephony.asterisk import AsteriskConfig, AsteriskProvider
    from socmon.telephony.provider import TelephonyError
    from socmon.voice.catalog import questionnaire_directory
    from socmon.voice.render import verify

    questionnaire = _load(path)
    if voice or media == MEDIA:
        media = questionnaire_directory(media, questionnaire.version_id, voice)
    check = verify(questionnaire, media)
    if not check.ready:
        print("Озвучка неполная: " + "; ".join(check.problems))
        print("Соберите: socmon voice render")
        return 1

    try:
        config = AsteriskConfig.from_env()
    except TelephonyError as exc:
        print(exc)
        return 1

    model = os.environ.get("SOCMON_ASR_MODEL", ONLINE_MODEL)
    print(f"Анкета:      {questionnaire.name} · {questionnaire.version_id}")
    print(f"Распознаю:   {model} на {os.environ.get('SOCMON_ASR_DEVICE', 'cpu')}")
    print("Загружаю модель…", flush=True)
    recognizer = GigaamRecognizer(GigaamConfig(
        model_name=model, device=os.environ.get("SOCMON_ASR_DEVICE", "cpu")
    ))
    recognizer.transcribe(b"\x00\x00" * 8000)  # прогрев: первый вызов долгий

    provider = AsteriskProvider(config)
    journal = MemoryJournal()
    call = LiveCall(
        provider, questionnaire, journal, recognizer,
        audio_host=os.environ.get("SOCMON_AUDIO_HOST", "127.0.0.1"),
        audio_port=int(os.environ.get("SOCMON_AUDIO_PORT", "0")),
        operator_endpoint=os.environ.get("SOCMON_OPERATOR_ENDPOINT"),
        on_event=lambda line: print(f"  {line}", flush=True),
    )

    async def run() -> None:
        attempt = await call.dial(destination, caller_id)
        session = journal.interviews.get(attempt)
        print()
        if session is None:
            print("Интервью не состоялось: разговор не начался.")
            return
        print(f"Исход:       {session.outcome.value if session.outcome else '—'}")
        print(f"Длительность: {session.duration_sec:.0f} с")
        answered = [a for a in session.answers.values() if a.is_substantive]
        print(f"Ответов:     {len(answered)} из {len(session.answers)}")
        for node_id, answer in session.answers.items():
            value = answer.text if answer.code is None else answer.code
            print(f"  {node_id:<20} {value}")

    async def managed_run() -> None:
        try:
            await run()
        finally:
            await provider.aclose()

    try:
        asyncio.run(managed_run())
    except TelephonyError as exc:
        print(f"Звонок завершён с ошибкой: {exc}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="socmon", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="проверить анкету перед запуском волны")
    check.add_argument("path", nargs="?", type=Path, default=DEFAULT)

    play = sub.add_parser("play", help="прогнать интервью вручную")
    play.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    play.add_argument(
        "--voice", nargs="?", type=Path, const=MEDIA, default=None,
        metavar="КАТАЛОГ", help="произносить реплики из озвученного каталога",
    )

    wave = sub.add_parser("wave", help="сводка по волне на синтетическом массиве")
    wave.add_argument("--size", type=int, default=1700)
    wave.add_argument("--seed", type=int, default=20260820)

    voice = sub.add_parser("voice", help="озвучка анкеты")
    voice_sub = voice.add_subparsers(dest="voice_command", required=True)

    voice_render = voice_sub.add_parser("render", help="озвучить анкету")
    voice_render.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    voice_render.add_argument("--out", type=Path, default=MEDIA)
    voice_render.add_argument("--voice", default="baya")
    voice_render.add_argument(
        "--rate", type=int, default=TELEPHONY_SAMPLE_RATE,
        help="частота: 8000 — как на линии, 24000 — для прослушивания",
    )
    voice_render.add_argument("--force", action="store_true")
    voice_render.add_argument("--model-id", default="v4_ru", help="версия модели Silero")
    voice_render.add_argument("--model-path", type=Path, help="локальный проверенный файл модели")
    voice_render.add_argument(
        "--natural", action="store_true", help="спокойная подача со смысловыми паузами",
    )

    voice_check = voice_sub.add_parser("check", help="проверить полноту озвучки")
    voice_check.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    voice_check.add_argument("--media", type=Path, default=MEDIA)

    voice_sub.add_parser("voices", help="список голосов")
    variants = voice_sub.add_parser("variants", help="пять голосов и примеры для сравнения")
    variants.add_argument("--out", type=Path, default=MEDIA / "voices")
    variants.add_argument("--model-path", type=Path, required=True)
    variants.add_argument("--model-id", default="v5_5_ru")

    realistic = voice_sub.add_parser(
        "realistic", help="мужская и женская озвучка Qwen3-TTS на Apple Silicon",
    )
    realistic.add_argument("path", nargs="?", type=Path, help="загруженная анкета YAML")
    realistic.add_argument("--out", type=Path, required=True)
    realistic.add_argument("--model-path", type=Path, required=True)
    realistic.add_argument("--references", type=Path, required=True)
    realistic.add_argument("--cache", type=Path, required=True)
    realistic.add_argument("--retakes", type=Path, help="карта повторных дублей после проверки")
    realistic.add_argument(
        "--takes", type=int, default=3,
        help="дублей на реплику; лучший по просодии становится основным",
    )
    realistic.add_argument("--temperature", type=float, default=0.7)
    realistic.add_argument(
        "--plain", action="store_true",
        help="без речевой разметки: читать формулировку как на бумаге",
    )
    realistic.add_argument(
        "--room-tone", type=float, default=None,
        help="комнатный тон на краях, дБ. Прослушивание 09.09 показало: слышен "
             "как помеха, по умолчанию выключен",
    )
    realistic.add_argument(
        "--breath", action="store_true",
        help="чуткая обрезка краёв ради вдоха. Оставляет шум модели перед речью — "
             "по умолчанию выключено",
    )
    realistic.add_argument(
        "--no-service", action="store_true", help="не озвучивать служебные реплики подачи",
    )
    realistic.add_argument(
        "--voice", choices=("denis", "anna", "alex"), help="только один голос; по умолчанию все",
    )
    realistic.add_argument(
        "--engine", choices=("qwen", "f5"), default="qwen",
        help="qwen — Qwen3-TTS на MLX; f5 — русская дообучка F5-TTS со словарём ударений",
    )
    realistic.add_argument("--steps", type=int, default=32, help="шагов диффузии F5")

    qa = voice_sub.add_parser("qa", help="проверить библиотеку распознаванием")
    qa.add_argument("--media", type=Path, required=True, help="каталог библиотеки одного голоса")
    qa.add_argument("--whisper", type=Path, required=True, help="каталог модели Whisper MLX")
    qa.add_argument("--prune", action="store_true", help="убрать провалившиеся дубли")
    qa.add_argument("--gender", choices=("f", "m"), default="f")

    simulate = sub.add_parser("simulate", help="прогнать волну без Asterisk")
    simulate.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    simulate.add_argument("--numbers", type=int, default=200)
    simulate.add_argument("--seed", type=int, default=20260820)
    simulate.add_argument(
        "--dropout", type=float, default=0.02,
        metavar="ДОЛЯ", help="вероятность бросить трубку на каждом вопросе",
    )

    contacts = sub.add_parser("contacts", help="база контактов")
    contacts_sub = contacts.add_subparsers(dest="contacts_command", required=True)
    contacts_import = contacts_sub.add_parser("import", help="загрузить файл с номерами")
    contacts_import.add_argument("path", type=Path)
    contacts_import.add_argument("--column", type=int, default=None)
    contacts_import.add_argument("--name", default="")
    contacts_import.add_argument("--seed", default="")

    calltest = sub.add_parser("calltest", help="позвонить и провести интервью")
    calltest.add_argument("destination", help="номер абонента; маршрут задаёт SOCMON_ARI_ENDPOINT")
    calltest.add_argument("--from", dest="caller_id", default="+74951234567")
    calltest.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    calltest.add_argument("--media", type=Path, default=MEDIA)
    calltest.add_argument(
        "--voice", choices=("denis", "anna", "alex"), default="",
        help="Анна или Алексей; запись выбирается по редакции анкеты",
    )

    serve = sub.add_parser("serve", help="поднять кабинет")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    useradd = sub.add_parser("useradd", help="завести учётную запись кабинета")
    useradd.add_argument("login")
    useradd.add_argument("name")
    useradd.add_argument("role")
    passwd = sub.add_parser("passwd", help="сменить пароль и отозвать сессии")
    passwd.add_argument("login")

    export = sub.add_parser("export", help="собрать оперативный пакет и отчёт")
    export.add_argument("--out", type=Path, default=Path("out"))
    export.add_argument("--size", type=int, default=1700)
    export.add_argument("--seed", type=int, default=20260820)
    export.add_argument("--wave", default="2026-17")
    export.add_argument("--pdf", action="store_true", help="конвертировать отчёт в PDF")

    args = parser.parse_args(argv)
    if args.command == "check":
        return cmd_check(args.path)
    if args.command == "wave":
        return cmd_wave(args.size, args.seed)
    if args.command == "export":
        return cmd_export(args.out, args.size, args.seed, args.wave, args.pdf)
    if args.command == "calltest":
        return cmd_calltest(args.destination, args.caller_id, args.path, args.media, args.voice)
    if args.command == "contacts":
        return cmd_contacts_import(args.path, args.column, args.name, args.seed)
    if args.command == "serve":
        return cmd_serve(args.host, args.port)
    if args.command == "useradd":
        return cmd_useradd(args.login, args.name, args.role)
    if args.command == "passwd":
        return cmd_passwd(args.login)
    if args.command == "simulate":
        return cmd_simulate(args.path, args.numbers, args.seed, args.dropout)
    if args.command == "voice":
        if args.voice_command == "realistic":
            from socmon.voice.catalog import NATURAL_OPTIONS
            from socmon.voice.delivery import VOICE_GENDER, Catalogue
            from socmon.voice.qwen import QwenMlxSynthesizer
            from socmon.voice.render import render, render_catalogue
            from socmon.voice.variants import build_variants

            synthesizer: Any
            if args.engine == "f5":
                from socmon.voice.f5 import F5Synthesizer

                synthesizer = F5Synthesizer(
                    args.model_path, args.references, args.cache,
                    takes=args.takes, steps=args.steps, shaped=not args.plain,
                )
            else:
                synthesizer = QwenMlxSynthesizer(
                    args.model_path, args.references, args.cache, args.retakes,
                    takes=args.takes, temperature=args.temperature, shaped=not args.plain,
                    keep_breath=args.breath, room_tone_db=args.room_tone,
                )
            print(f"Рецепт: {synthesizer.fingerprint}")
            voices = tuple(o for o in NATURAL_OPTIONS if not args.voice or o.id == args.voice)
            if args.path:
                questionnaire = _load(args.path)
                for option in voices:
                    report = render(
                        questionnaire, synthesizer, args.out / option.id, voice=option.id,
                    )
                    print(f"{option.name}: {report.total} реплик → {report.directory}")
            else:
                build_variants(args.out, synthesizer, voices=voices)
            if not args.no_service:
                catalogue = Catalogue.from_yaml()
                for option in voices:
                    report = render_catalogue(
                        catalogue, synthesizer, args.out / option.id,
                        voice=option.id, gender=VOICE_GENDER[option.id],
                    )
                    print(f"{option.name}: служебных реплик {report.total}")
            return 0
        if args.voice_command == "qa":
            return cmd_voice_qa(args.media, args.whisper, args.prune, args.gender)
        if args.voice_command == "variants":
            from socmon.voice.synth import SileroSynthesizer
            from socmon.voice.variants import build_variants

            build_variants(args.out, SileroSynthesizer(
                model_id=args.model_id, model_path=args.model_path, natural=True,
            ))
            return 0
        if args.voice_command == "render":
            return cmd_voice_render(
                args.path, args.out, args.voice, args.rate, args.force,
                model_id=args.model_id, model_path=args.model_path, natural=args.natural,
            )
        if args.voice_command == "check":
            return cmd_voice_check(args.path, args.media)
        return cmd_voice_list()
    return cmd_play(args.path, args.voice)


if __name__ == "__main__":
    raise SystemExit(main())
