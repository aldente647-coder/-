"""Именованные варианты озвучки для сравнения и демо."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VoiceOption:
    id: str
    name: str
    gender: str


VOICE_OPTIONS = (
    VoiceOption("denis", "Денис", "Мужской голос, живая запись"),
    VoiceOption("anna", "Анна", "Женский голос"),
    VoiceOption("alex", "Алексей", "Мужской голос"),
    VoiceOption("baya", "Бая", "Женский голос"),
    VoiceOption("kseniya", "Ксения", "Женский голос"),
    VoiceOption("xenia", "Ксения X", "Женский голос"),
    VoiceOption("aidar", "Айдар", "Мужской голос"),
    VoiceOption("eugene", "Евгений", "Мужской голос"),
)
#: Голоса, которые синтезирует Qwen по референсу. «Денис» — клон живой записи
#: владельца от 09.09.2026; «Анна» и «Алексей» — синтетические референсы.
CLONED = {"denis", "anna", "alex"}
SILERO_OPTIONS = tuple(v for v in VOICE_OPTIONS if v.id not in CLONED)
NATURAL_OPTIONS = tuple(v for v in VOICE_OPTIONS if v.id in CLONED)
VOICE_BY_ID = {v.id: v for v in VOICE_OPTIONS}
SAMPLES = {
    "consent": "Короткий вопрос",
    "intro": "Знакомство и объяснение опроса",
    "q07_material": "Вопрос с вариантами",
    "end_no_consent": "Завершение разговора",
}


def voice_directory(media: Path, voice: str) -> Path:
    if voice and voice not in VOICE_BY_ID:
        raise ValueError("неизвестный голос")
    return media / "voices" / voice if voice else media


def questionnaire_directory(media: Path, version_id: str, voice: str = "") -> Path:
    """Сначала запись именно этой редакции, затем общая библиотека волн."""
    fallback = voice_directory(media, voice)
    if not re.fullmatch(r"[0-9a-f]{16}", version_id):
        raise ValueError("неверный идентификатор редакции анкеты")
    selected = media / "questionnaires" / version_id / (voice or "anna")
    if selected.is_dir() and selected.resolve().is_relative_to(media.resolve()):
        return selected
    return fallback


def safe_audio(directory: Path, filename: str) -> Path | None:
    """Не отдавать файлы, на которые указывает ссылка за пределы библиотеки."""
    path = directory / filename
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        return None
    return path
