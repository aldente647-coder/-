"""Словарь ударений: формулировка с отметками, где ударение.

Русское ударение — то, на чём синтез ошибается чаще всего, и ошибка
слышна каждому: «зво́нит» вместо «звони́т» выдаёт машину сильнее любого
тембра. Модель, читая текст без отметок, угадывает; словарь не угадывает.

Разметка — RUAccent: словарь плюс модель для омографов («за́мок» и «замо́к»
различаются по контексту). Отметка — «+» перед ударной гласной, как ждёт
русская дообучка F5-TTS. Слова при этом те же самые: проверка `same_words`
из речевой разметки продолжает работать — «+» не буква.

Омографы модель разбирает не всегда верно. На этот случай — файл исключений
рядом: слово в форме, как в анкете, и правильная отметка. Исключения
побеждают словарь всегда: их ставил человек, который слушал.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

EXCEPTIONS_PATH = Path(__file__).with_name("accent-exceptions.txt")
_MARK = "+"
_WORD = re.compile(r"[а-яёА-ЯЁ]+")


class AccentError(RuntimeError):
    pass


def load_exceptions(path: Path = EXCEPTIONS_PATH) -> dict[str, str]:
    """Исключения: строка «слово → сл+ово», пустые строки и # — комментарии."""
    table: dict[str, str] = {}
    if not path.exists():
        return table
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "→" not in line:
            raise AccentError(f"исключение без стрелки: {line!r}")
        plain, marked = (part.strip() for part in line.split("→", 1))
        if plain.replace("ё", "е").lower() != marked.replace(_MARK, "").replace("ё", "е").lower():
            raise AccentError(f"исключение меняет слово: {line!r}")
        table[plain.lower()] = marked
    return table


class Accentuator:
    """Расставляет ударения. Модель грузится лениво и один раз."""

    def __init__(self, exceptions: dict[str, str] | None = None) -> None:
        self._exceptions = exceptions if exceptions is not None else load_exceptions()
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            self._model = _load_ruaccent()
        return self._model

    def mark(self, text: str) -> str:
        """Текст с «+» перед ударными гласными."""
        marked = str(self._load().process_all(text))
        return self._apply_exceptions(marked)

    def _apply_exceptions(self, marked: str) -> str:
        if not self._exceptions:
            return marked

        def fix(match: re.Match[str]) -> str:
            word = match.group(0)
            plain = word.replace(_MARK, "").lower()
            override = self._exceptions.get(plain)
            if override is None:
                return word
            return _match_case(override, word.replace(_MARK, ""))

        return re.sub(rf"[а-яёА-ЯЁ{re.escape(_MARK)}]+", fix, marked)


def strip_marks(text: str) -> str:
    return text.replace(_MARK, "")


def marked_share(text: str) -> float:
    """Доля слов длиннее двух букв, получивших ударение. Диагностика
    словаря: на анкете ждём около девяти десятых."""
    words = [w for w in _WORD.findall(strip_marks(text)) if len(w) > 2]
    if not words:
        return 0.0
    marked = sum(1 for w in re.findall(rf"[а-яёА-ЯЁ{re.escape(_MARK)}]+", text)
                 if _MARK in w and len(strip_marks(w)) > 2)
    return marked / len(words)


def _match_case(override: str, original: str) -> str:
    if original[:1].isupper():
        return override[:1].upper() + override[1:]
    return override


def _load_ruaccent() -> Any:
    """RUAccent с обходом несовместимости версий: ONNX-модель ждёт вход
    token_type_ids, а библиотека его не подаёт."""
    try:
        import numpy as np
        import onnxruntime
        from ruaccent import RUAccent
    except ImportError as exc:  # pragma: no cover
        raise AccentError("нужен пакет ruaccent (группа accent)") from exc

    original = onnxruntime.InferenceSession.run
    if not getattr(original, "_socmon_patched", False):
        def run(self: Any, outputs: Any, feed: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            names = {i.name for i in self.get_inputs()}
            if "token_type_ids" in names and "token_type_ids" not in feed:
                feed = dict(feed, token_type_ids=np.zeros_like(feed["input_ids"]))
            return original(self, outputs, feed, *args, **kwargs)
        run._socmon_patched = True  # type: ignore[attr-defined]
        onnxruntime.InferenceSession.run = run

    model = RUAccent()
    model.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
    return model
