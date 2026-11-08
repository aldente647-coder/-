"""Словарь ударений: исключения, доля размеченных слов, сохранность слов."""

from __future__ import annotations

from pathlib import Path

import pytest

from socmon.voice.accent import AccentError, Accentuator, load_exceptions, marked_share, strip_marks
from socmon.voice.spoken import same_words


def test_exceptions_override_the_dictionary(tmp_path: Path) -> None:
    path = tmp_path / "ex.txt"
    path.write_text("замок → зам+ок  # на двери\n", encoding="utf-8")
    acc = Accentuator(load_exceptions(path))
    assert acc._apply_exceptions("Ст+арый з+амок на двер+и") == "Ст+арый зам+ок на двер+и"
    # Регистр первой буквы сохраняется.
    assert acc._apply_exceptions("З+амок висит") == "Зам+ок висит"


def test_exception_cannot_change_the_word(tmp_path: Path) -> None:
    path = tmp_path / "ex.txt"
    path.write_text("замок → з+амки\n", encoding="utf-8")
    with pytest.raises(AccentError, match="меняет слово"):
        load_exceptions(path)


def test_marks_are_not_letters() -> None:
    marked = "Скаж+ите, пож+алуйста, в+ам уж+е исп+олнилось восемн+адцать л+ет?"
    plain = "Скажите, пожалуйста, вам уже исполнилось восемнадцать лет?"
    assert same_words(strip_marks(marked), plain)
    assert marked_share(marked) == pytest.approx(1.0)
    assert marked_share("вам уже лет") < 0.5


@pytest.mark.skipif(
    pytest.importorskip("ruaccent", reason="словарь ударений не установлен") is None, reason=""
)
def test_dictionary_marks_most_questionnaire_words() -> None:
    """На анкете словарь обязан размечать почти всё: неразмеченное слово
    модель прочтёт наугад."""
    from socmon.core.questionnaire import Questionnaire
    from socmon.voice.render import speakable

    acc = Accentuator()
    q = Questionnaire.from_yaml("src/socmon/questionnaires/wave-base.yaml")
    texts = [" ".join(n.text.split()) for n in speakable(q)]
    marked = [acc.mark(t) for t in texts]
    for plain, out in zip(texts, marked, strict=True):
        assert same_words(plain, strip_marks(out)), plain
    share = sum(marked_share(m) for m in marked) / len(marked)
    assert share > 0.85, share
