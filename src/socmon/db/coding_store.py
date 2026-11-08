"""Сохранение результатов ночного разбора.

Мост между `socmon.coding`, который считает, и базой, которая хранит.
Отдельный модуль, потому что правило здесь одно и жёсткое: **разбор
никогда не трогает первичный ответ**. Он добавляет строки в отдельную
таблицу, и перекодирование новой версией классификатора даёт новую
строку, а не правит старую (ТЗ §11.3).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from socmon.coding.analysis import InterviewAnalysis
from socmon.db.models import Interview, OpenAnswerCoding


def save_analysis(
    db: DbSession,
    interview_id: uuid.UUID,
    analysis: InterviewAnalysis,
    *,
    model: str,
) -> list[OpenAnswerCoding]:
    """Сохранить разбор интервью.

    Повторный прогон той же версией классификатора ничего не добавляет:
    уникальность по «интервью + узел + версия» не даёт задвоить разметку,
    если ночной батч перезапустили.
    """
    existing = {
        row.node_id
        for row in db.scalars(
            select(OpenAnswerCoding).where(
                OpenAnswerCoding.interview_id == interview_id,
                OpenAnswerCoding.classifier_version == analysis.classifier_version,
            )
        )
    }

    saved: list[OpenAnswerCoding] = []
    for quote, coding in zip(analysis.quotes, analysis.codings, strict=False):
        if quote.node_id in existing:
            continue
        mentions = [
            {
                "surface": m.surface, "kind": m.kind.value,
                "key": m.key, "ambiguous": m.ambiguous,
                "candidates": list(m.candidates),
            }
            for m in analysis.mentions
        ]
        row = OpenAnswerCoding(
            interview_id=interview_id,
            node_id=quote.node_id,
            classifier_version=analysis.classifier_version,
            model=model,
            themes=list(coding.codes),
            entities=mentions,
            sentiment={
                "tone": analysis.tone.tone.value,
                "confidence": analysis.tone.confidence,
                "signals": [s.name for s in analysis.tone.signals],
                "disclaimer": analysis.tone.disclaimer,
            },
            ambiguous=any(m.ambiguous for m in analysis.mentions) or coding.unmatched,
        )
        db.add(row)
        saved.append(row)
    db.flush()
    return saved


def pending_review(
    db: DbSession, wave_id: uuid.UUID, *, limit: int = 100
) -> Sequence[OpenAnswerCoding]:
    """Очередь ручной проверки (ТЗ §11.3).

    Сюда попадает то, что система сама пометила сомнительным: неоднозначные
    сущности и неразмеченные ответы. Это не замена случайной выборки,
    которой требует §11.3, а дополнение к ней — проверять только
    подозрительное значит не увидеть систематической ошибки.
    """
    return db.scalars(
        select(OpenAnswerCoding)
        .join(Interview, Interview.id == OpenAnswerCoding.interview_id)
        .where(
            Interview.wave_id == wave_id,
            OpenAnswerCoding.ambiguous.is_(True),
            OpenAnswerCoding.reviewed_at.is_(None),
        )
        .order_by(OpenAnswerCoding.created_at)
        .limit(limit)
    ).all()


def mark_reviewed(
    db: DbSession, coding_id: uuid.UUID, reviewer: str
) -> OpenAnswerCoding | None:
    """Отметить разметку проверенной человеком.

    Сама разметка при этом не меняется: проверяющий подтверждает или
    отправляет на исправление, но модельный вывод остаётся модельным —
    §11.3 требует, чтобы автоматически сгенерированное было отличимо
    от проверенного, а не подменялось им.
    """
    row = db.get(OpenAnswerCoding, coding_id)
    if row is None:
        return None
    row.reviewed_by = reviewer
    row.reviewed_at = datetime.now(UTC)
    db.flush()
    return row


def theme_counts(db: DbSession, wave_id: uuid.UUID) -> dict[str, int]:
    """Сколько раз каждая тема встретилась в волне."""
    rows = db.scalars(
        select(OpenAnswerCoding.themes)
        .join(Interview, Interview.id == OpenAnswerCoding.interview_id)
        .where(Interview.wave_id == wave_id)
    ).all()
    counts: dict[str, int] = {}
    for themes in rows:
        for code in themes or []:
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def review_progress(db: DbSession, wave_id: uuid.UUID) -> tuple[int, int]:
    """Сколько сомнительных разметок проверено из скольких."""
    base = (
        select(func.count())
        .select_from(OpenAnswerCoding)
        .join(Interview, Interview.id == OpenAnswerCoding.interview_id)
        .where(Interview.wave_id == wave_id, OpenAnswerCoding.ambiguous.is_(True))
    )
    total = int(db.scalar(base) or 0)
    done = int(db.scalar(base.where(OpenAnswerCoding.reviewed_at.isnot(None))) or 0)
    return done, total
