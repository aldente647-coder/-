"""Проверка записей распознаванием: модель могла повторить слово или
ошибиться в произношении, и ни один просодический балл этого не увидит.

Каждый файл библиотеки — основной и все дубли — распознаётся независимой
моделью и сверяется по словам с утверждённой формулировкой. Дубль, который
не совпал, из библиотеки убирается: подача не должна сыграть его в живом
звонке. Если не совпал сам основной файл и заменить его нечем, реплика
помечается — это уже повод пересинтезировать, а не молча выпустить.

Распознаватель — любой вызываемый объект «путь → текст»; в бою это Whisper
на MLX, в тестах подстановка. Сравнение допускает то, что распознавание
не обязано воспроизводить: регистр, «ё», пунктуацию, произношение латинских
названий.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path

from socmon.voice.render import MANIFEST_NAME, VoiceEntry, VoiceManifest
from socmon.voice.spoken import words

Transcriber = Callable[[Path], str]


@dataclass(frozen=True, slots=True)
class Check:
    key: str
    file: str
    expected: str
    heard: str
    exact: bool
    similarity: float
    """Сходство слитных строк, 0–1. Окончания слов и склейки распознаватель
    путает регулярно: «напряжения» вместо «напряжение», «недоверяете»
    вместо «не доверяете». Это не ошибка записи."""

    missing: tuple[str, ...]
    """Ожидаемые слова, которых распознавание не услышало."""

    extra: tuple[str, ...]
    """Услышанные слова, которых в формулировке нет: повтор или оговорка."""

    @property
    def passed(self) -> bool:
        """Совпало дословно или расходится не сильнее, чем путается
        распознаватель: окончание, склейка двух слов, цифра вместо слова.

        Повтор слова из формулировки — всегда провал, даже при высоком
        сходстве: «вы вы согласны» слитно почти совпадает с оригиналом,
        а в трубке это заикание."""
        if self.exact:
            return True
        # Служебные словечки («в», «и», «не») распознаватель дописывает
        # и теряет сам; повтором считается только значимое слово.
        repeated = any(len(word) >= 4 and word in words(self.expected) for word in self.extra)
        return self.similarity >= PASS_SIMILARITY and not repeated


@dataclass(frozen=True, slots=True)
class Report:
    checks: tuple[Check, ...]
    pruned: tuple[str, ...]
    """Файлы дублей, убранные из библиотеки."""

    failed_primary: tuple[str, ...]
    """Реплики, у которых не прошёл основной файл и заменить нечем."""

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def exact(self) -> int:
        return sum(1 for c in self.checks if c.exact)

    def save(self, path: Path) -> None:
        payload = {
            "checks": [asdict(c) for c in self.checks],
            "pruned": list(self.pruned),
            "failed_primary": list(self.failed_primary),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


PASS_SIMILARITY = 0.9


def compare(
    expected: str, heard: str
) -> tuple[bool, float, tuple[str, ...], tuple[str, ...]]:
    """Сверить по словам. Порядок важен: переставленные слова — не та фраза."""
    want = words(expected)
    got = words(heard)
    if want == got:
        return True, 1.0, (), ()
    similarity = SequenceMatcher(None, "".join(want), "".join(got)).ratio()
    # Мультимножество, а не множество: повторённое слово — лишнее слово.
    missing = tuple((Counter(want) - Counter(got)).elements())
    extra = tuple((Counter(got) - Counter(want)).elements())
    return False, round(similarity, 3), missing, extra


def audit(
    directory: Path,
    texts: dict[str, str],
    transcribe: Transcriber,
    *,
    prune: bool = False,
    keys: Sequence[str] | None = None,
) -> Report:
    """Проверить библиотеку и при `prune` убрать провалившиеся дубли.

    `texts` — утверждённая формулировка по ключу; реплики, которых в нём
    нет, не проверяются: сверять не с чем.
    """
    manifest = VoiceManifest.load(directory / MANIFEST_NAME)
    checks: list[Check] = []
    pruned: list[str] = []
    failed_primary: list[str] = []
    entries = dict(manifest.entries)

    for key, entry in manifest.entries.items():
        if key not in texts or (keys is not None and key not in keys):
            continue
        expected = texts[key]
        results: dict[str, Check] = {}
        for name in (entry.file, *entry.takes):
            path = directory / name
            if not path.exists():
                continue
            heard = transcribe(path)
            exact, similarity, missing, extra = compare(expected, heard)
            results[name] = Check(key, name, expected, heard, exact, similarity, missing, extra)
        checks.extend(results.values())

        good = [name for name, check in results.items() if check.passed]
        if not prune:
            continue
        if entry.file in good:
            kept = tuple(t for t in entry.takes if t in good)
            dropped = [t for t in entry.takes if t not in good]
        elif good:
            # Основной не прошёл, но есть годный дубль — он и станет основным.
            promoted = good[0]
            kept = tuple(t for t in good if t != promoted)
            dropped = [entry.file, *(t for t in entry.takes if t not in good)]
            (directory / promoted).replace(directory / f"{key}.wav")
            dropped = [d for d in dropped if d != entry.file]
            entries[key] = _with_takes(entry, kept)
            for name in dropped:
                (directory / name).unlink(missing_ok=True)
                pruned.append(name)
            continue
        else:
            failed_primary.append(key)
            continue
        entries[key] = _with_takes(entry, kept)
        for name in dropped:
            (directory / name).unlink(missing_ok=True)
            pruned.append(name)

    if prune:
        VoiceManifest(
            questionnaire=manifest.questionnaire, version_id=manifest.version_id,
            voice=manifest.voice, sample_rate=manifest.sample_rate, entries=entries,
            synthesis_id=manifest.synthesis_id,
        ).save(directory)
    return Report(tuple(checks), tuple(pruned), tuple(failed_primary))


def _with_takes(entry: VoiceEntry, takes: tuple[str, ...]) -> VoiceEntry:
    return replace(entry, takes=takes)


def whisper_transcriber(model_path: Path) -> Transcriber:
    """Whisper на MLX как распознаватель. Ленивый импорт: на сервере
    ни MLX, ни Whisper нет и не нужно."""
    import mlx_whisper

    def transcribe(path: Path) -> str:
        result = mlx_whisper.transcribe(
            str(path), path_or_hf_repo=str(model_path), language="ru",
            temperature=0.0, condition_on_previous_text=False,
        )
        return str(result["text"]).strip()

    return transcribe
