"""Озвучка анкеты и её манифест.

Главная забота модуля — не дать роботу произносить одно, а записывать другое.

Каждая реплика привязана к хешу своего текста. Правка формулировки меняет хеш,
запись помечается устаревшей, и волну нельзя запустить, пока она не
переозвучена. Без этого сценарий выглядит озвученным, робот бодро читает
прошлую редакцию вопроса, а в массив пишется код по новой — и расхождение
всплывает уже после публикации результатов.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from socmon.core.questionnaire import EndNode, InfoNode, Node, Questionnaire
from socmon.voice.catalog import VOICE_BY_ID
from socmon.voice.delivery import Catalogue, Gender
from socmon.voice.synth import (
    TELEPHONY_SAMPLE_RATE,
    Speech,
    SpeechSynthesizer,
    prepare,
)

MANIFEST_NAME = "voice.json"


class RenderError(RuntimeError):
    pass


def text_hash(text: str) -> str:
    """Отпечаток произносимого текста — после нормализации пробелов.

    Считается от подготовленной строки, иначе переформатирование YAML
    объявляло бы устаревшей исправную запись.
    """
    return hashlib.sha256(prepare(text).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class VoiceEntry:
    audio_key: str
    node_id: str
    text_hash: str
    file: str
    duration_sec: float
    rendered_at: str
    takes: tuple[str, ...] = ()
    """Запасные дубли той же реплики, от лучшего к худшему по просодии.
    `file` — основной. Подача выбирает из всех, чтобы два звонка не звучали
    одинаково; звонок с одним дублем в библиотеке играет его всегда."""


@dataclass(frozen=True, slots=True)
class VoiceManifest:
    """Что и каким голосом озвучено."""

    questionnaire: str
    version_id: str
    voice: str
    sample_rate: int
    entries: dict[str, VoiceEntry] = field(default_factory=dict)
    synthesis_id: str = "legacy"

    def to_json(self) -> str:
        payload = {
            "questionnaire": self.questionnaire,
            "version_id": self.version_id,
            "voice": self.voice,
            "sample_rate": self.sample_rate,
            "synthesis_id": self.synthesis_id,
            "entries": {k: asdict(v) for k, v in sorted(self.entries.items())},
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> VoiceManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            questionnaire=raw["questionnaire"],
            version_id=raw["version_id"],
            voice=raw["voice"],
            sample_rate=raw["sample_rate"],
            synthesis_id=raw.get("synthesis_id", "legacy"),
            entries={
                k: VoiceEntry(**{**v, "takes": tuple(v.get("takes", ()))})
                for k, v in raw["entries"].items()
            },
        )

    def save(self, directory: Path) -> Path:
        path = directory / MANIFEST_NAME
        path.write_text(self.to_json(), encoding="utf-8")
        return path


@dataclass(frozen=True, slots=True)
class VoiceReport:
    rendered: tuple[str, ...]
    reused: tuple[str, ...]
    manifest: VoiceManifest
    directory: Path

    @property
    def total(self) -> int:
        return len(self.rendered) + len(self.reused)


@dataclass(frozen=True, slots=True)
class Verification:
    """Готова ли озвучка к запуску волны."""

    missing: tuple[str, ...]
    """Узлы без записи вовсе."""

    stale: tuple[str, ...]
    """Формулировка изменилась, запись осталась прежней. Самый опасный
    случай: сценарий выглядит озвученным."""

    orphaned: tuple[str, ...]
    """Записи от узлов, которых в анкете больше нет. Не мешают, но
    захламляют библиотеку и путают при разборе."""

    version_mismatch: bool

    @property
    def ready(self) -> bool:
        return not self.missing and not self.stale

    @property
    def problems(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.missing:
            notes.append("не озвучены: " + ", ".join(self.missing))
        if self.stale:
            notes.append(
                "формулировка изменилась после озвучки: " + ", ".join(self.stale)
            )
        if self.orphaned:
            notes.append("лишние записи: " + ", ".join(self.orphaned))
        return tuple(notes)


def speakable(questionnaire: Questionnaire) -> list[Node]:
    """Узлы, которые робот произносит вслух.

    Терминал с пустым текстом озвучивать нечего: он существует только чтобы
    зафиксировать исход интервью.
    """
    return [node for node in questionnaire.nodes if prepare(node.text)]


def render(
    questionnaire: Questionnaire,
    synthesizer: SpeechSynthesizer,
    directory: Path,
    *,
    voice: str,
    sample_rate: int = TELEPHONY_SAMPLE_RATE,
    force: bool = False,
    now: datetime | None = None,
) -> VoiceReport:
    """Озвучить анкету.

    Неизменившиеся реплики переиспользуются: полная переозвучка ради правки
    одного вопроса не только долгая, но и вредная — §8.3 требует стабильности
    голоса между волнами, а каждый прогон синтеза даёт чуть другое звучание.
    """
    items = [
        (node.audio_key or node.id, node.id, node.text)
        for node in speakable(questionnaire)
    ]
    return _render_items(
        items, synthesizer, directory, voice=voice, sample_rate=sample_rate,
        force=force, now=now, questionnaire=questionnaire.name,
        version_id=questionnaire.version_id,
    )


def render_catalogue(
    catalogue: Catalogue,
    synthesizer: SpeechSynthesizer,
    directory: Path,
    *,
    voice: str,
    gender: Gender,
    sample_rate: int = TELEPHONY_SAMPLE_RATE,
    force: bool = False,
    now: datetime | None = None,
) -> VoiceReport:
    """Озвучить служебные реплики подачи в ту же библиотеку.

    Отклики и префиксы живут рядом с вопросами под своими ключами: подача
    просит их у того же каталога, что и вопросы, и тем же голосом."""
    items = [(p.key, "service", p.spoken(gender)) for p in catalogue.phrases()]
    return _render_items(
        items, synthesizer, directory, voice=voice, sample_rate=sample_rate,
        force=force, now=now, questionnaire=catalogue.name,
        version_id=catalogue.version_id,
    )


def _render_items(
    items: list[tuple[str, str, str]],
    synthesizer: SpeechSynthesizer,
    directory: Path,
    *,
    voice: str,
    sample_rate: int,
    force: bool,
    now: datetime | None,
    questionnaire: str,
    version_id: str,
) -> VoiceReport:
    if voice not in VOICE_BY_ID:
        raise RenderError(
            f"неизвестный голос «{voice}»; доступны: {', '.join(sorted(VOICE_BY_ID))}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now(UTC)).isoformat()
    synthesis_id = str(getattr(synthesizer, "fingerprint", "legacy"))
    count = max(1, int(getattr(synthesizer, "takes", 1)))

    previous = _previous(directory, voice, sample_rate, synthesis_id)
    entries: dict[str, VoiceEntry] = {}
    rendered: list[str] = []
    reused: list[str] = []

    for key, node_id, text in items:
        digest = text_hash(text)
        target = directory / f"{key}.wav"

        known = previous.get(key)
        if (
            not force and known and known.text_hash == digest and target.exists()
            and len(known.takes) == count - 1
            and all((directory / t).exists() for t in known.takes)
        ):
            entries[key] = known
            reused.append(key)
            continue

        primary, others = _synthesize_takes(
            synthesizer, text, voice=voice, sample_rate=sample_rate, count=count,
        )
        target.write_bytes(primary.wav)
        take_files: list[str] = []
        for index, speech in enumerate(others, start=2):
            name = f"{key}-t{index}.wav"
            (directory / name).write_bytes(speech.wav)
            take_files.append(name)
        # Дубли прошлой сборки сверх нынешнего числа не должны остаться:
        # подача сочла бы их живыми.
        for stale in directory.glob(f"{key}-t*.wav"):
            if stale.name not in take_files:
                stale.unlink()
        entries[key] = VoiceEntry(
            audio_key=key,
            node_id=node_id,
            text_hash=digest,
            file=target.name,
            duration_sec=round(primary.duration_sec, 3),
            rendered_at=stamp,
            takes=tuple(take_files),
        )
        rendered.append(key)

    # Записи чужих анкет остаются в манифесте. Одна библиотека обслуживает
    # все волны сразу — ядро у них общее, и переозвучивать его на каждую
    # волну и незачем, и нельзя: §8.3 требует стабильности голоса. Если бы
    # манифест описывал только текущую анкету, озвучка предыдущей исчезала
    # бы при рендере следующей — файлы на диске, а в манифесте пусто.
    carried = {
        key: entry
        for key, entry in previous.items()
        if key not in entries and (directory / entry.file).exists()
    }
    manifest = VoiceManifest(
        questionnaire=questionnaire,
        version_id=version_id,
        voice=voice,
        sample_rate=sample_rate,
        entries=carried | entries,
        synthesis_id=synthesis_id,
    )
    manifest.save(directory)
    return VoiceReport(
        rendered=tuple(rendered),
        reused=tuple(reused),
        manifest=manifest,
        directory=directory,
    )


def _synthesize_takes(
    synthesizer: SpeechSynthesizer, text: str, *, voice: str, sample_rate: int, count: int,
) -> tuple[Speech, list[Speech]]:
    """Несколько дублей одной реплики; лучший по просодии — основной.

    Остальные упорядочены от лучшего к худшему. Просодия — грубая мера,
    она не отличит хорошую интонацию от плохой, но монотонный, скомканный
    или сбойный дубль отсеет надёжно."""
    if count <= 1:
        return synthesizer.synthesize(text, voice=voice, sample_rate=sample_rate), []

    from socmon.voice.prosody import best_take, measure

    speeches: list[Speech] = []
    for take in range(count):
        speeches.append(synthesizer.synthesize(  # type: ignore[call-arg]
            text, voice=voice, sample_rate=sample_rate, take=take,
        ))
    scored = [
        (index, measure(_samples(speech.wav), speech.sample_rate, text))
        for index, speech in enumerate(speeches)
    ]
    best = best_take(scored)
    order = sorted(range(count), key=lambda i: (-scored[i][1].score, i))
    return speeches[best], [speeches[i] for i in order if i != best]


def _samples(wav: bytes) -> Any:
    """Отсчёты WAV как float в [-1, 1]."""
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(wav)) as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0


def takes_available(entry: VoiceEntry, directory: Path) -> int:
    """Сколько дублей реплики реально лежит в библиотеке, включая основной."""
    return 1 + sum(1 for name in entry.takes if (directory / name).exists())


def verify_delivery(catalogue: Catalogue, directory: Path, gender: Gender) -> Verification:
    """Озвучены ли служебные реплики подачи для этого голоса."""
    path = directory / MANIFEST_NAME
    phrases = catalogue.phrases()
    if not path.exists():
        return Verification(
            missing=tuple(p.key for p in phrases), stale=(), orphaned=(), version_mismatch=True,
        )
    manifest = VoiceManifest.load(path)
    missing: list[str] = []
    stale: list[str] = []
    for phrase in phrases:
        entry = manifest.entries.get(phrase.key)
        if entry is None or not (directory / entry.file).exists():
            missing.append(phrase.key)
        elif entry.text_hash != text_hash(phrase.spoken(gender)):
            stale.append(phrase.key)
    return Verification(
        missing=tuple(missing), stale=tuple(stale), orphaned=(), version_mismatch=False,
    )


def _previous(
    directory: Path, voice: str, sample_rate: int, synthesis_id: str
) -> dict[str, VoiceEntry]:
    """Прошлый манифест — если он про тот же голос и ту же частоту.

    Смена голоса или частоты обесценивает всю библиотеку: смешивать в одной
    анкете два голоса нельзя (ТЗ §8.3).
    """
    path = directory / MANIFEST_NAME
    if not path.exists():
        return {}
    manifest = VoiceManifest.load(path)
    if (manifest.voice != voice or manifest.sample_rate != sample_rate
            or manifest.synthesis_id != synthesis_id):
        return {}
    return manifest.entries


def verify(
    questionnaire: Questionnaire,
    directory: Path,
) -> Verification:
    """Проверить озвучку перед запуском волны."""
    path = directory / MANIFEST_NAME
    if not path.exists():
        keys = tuple(n.audio_key or n.id for n in speakable(questionnaire))
        return Verification(missing=keys, stale=(), orphaned=(), version_mismatch=True)

    manifest = VoiceManifest.load(path)
    missing: list[str] = []
    stale: list[str] = []
    expected: set[str] = set()

    for node in speakable(questionnaire):
        key = node.audio_key or node.id
        expected.add(key)
        entry = manifest.entries.get(key)
        if entry is None or not (directory / entry.file).exists():
            missing.append(key)
        elif entry.text_hash != text_hash(node.text):
            stale.append(key)

    return Verification(
        missing=tuple(missing),
        stale=tuple(stale),
        orphaned=tuple(sorted(set(manifest.entries) - expected)),
        version_mismatch=manifest.version_id != questionnaire.version_id,
    )


def total_duration(manifest: VoiceManifest, keys: Iterable[str] | None = None) -> float:
    """Суммарная длительность перечисленных реплик."""
    selected: Sequence[str] = list(keys) if keys is not None else list(manifest.entries)
    return sum(manifest.entries[k].duration_sec for k in selected if k in manifest.entries)


def straight_path(questionnaire: Questionnaire) -> list[Node]:
    """Узлы «прямого» прохода анкеты: согласие дано, фильтры пройдены,
    все вопросы отвечены с первого раза.

    Ветвление игнорируется сознательно: это нижняя оценка длительности.
    Реальное интервью длиннее — на переспросы, паузы и уточнения.
    """
    path: list[Node] = []
    current: str | None = questionnaire.entry_id
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        node = questionnaire.node(current)
        path.append(node)
        current = node.next
    return path


@dataclass(frozen=True, slots=True)
class DurationEstimate:
    """Оценка длительности интервью (ТЗ §3.3: не более 10 минут у 90%)."""

    robot_sec: float
    questions: int
    answer_sec: float
    pause_sec: float

    @property
    def total_sec(self) -> float:
        return self.robot_sec + self.questions * (self.answer_sec + self.pause_sec)

    @property
    def within_limit(self) -> bool:
        return self.total_sec <= 600

    @property
    def note(self) -> str:
        return (
            f"речь робота {self.robot_sec / 60:.1f} мин + {self.questions} ответов "
            f"по {self.answer_sec:.0f} с с паузой {self.pause_sec:.0f} с "
            f"= {self.total_sec / 60:.1f} мин"
        )


def estimate_duration(
    questionnaire: Questionnaire,
    manifest: VoiceManifest,
    *,
    answer_sec: float = 3.0,
    pause_sec: float = 1.0,
) -> DurationEstimate:
    """Оценить длительность интервью по фактической длине реплик.

    Числа ответа и паузы — допущения до пилота, а не измерения. Их и надо
    заменить замером на первых же реальных звонках: §3.3 требует, чтобы
    в лимит укладывалось 90% завершённых интервью, и проверяется это
    фактом, а не расчётом.
    """
    path = straight_path(questionnaire)
    keys = [n.audio_key or n.id for n in path if prepare(n.text)]
    asks = sum(1 for n in path if not isinstance(n, InfoNode | EndNode))
    return DurationEstimate(
        robot_sec=total_duration(manifest, keys),
        questions=asks,
        answer_sec=answer_sec,
        pause_sec=pause_sec,
    )
