"""Множественный выбор и отфильтрованные вопросы

Анкета Заказчика содержит вопросы «до трёх ответов» (источники новостей,
доверие к СМИ) и вопросы под фильтром — про детский сад спрашивают только
тех, у кого есть дошкольники.

Первое требует места под несколько кодов: node_id входит в первичный ключ,
и второй ответ на тот же вопрос в таблицу не помещается. Второе снимает
нижнюю границу с числа попыток: у незаданного вопроса их ноль.

Revision ID: c4a1f27de8b3
Revises: b7cf962b217b
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c4a1f27de8b3"
down_revision: str | Sequence[str] | None = "b7cf962b217b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "survey"


def upgrade() -> None:
    op.add_column(
        "answer",
        sa.Column("codes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        schema=SCHEMA,
    )
    # Существующие ответы — одиночные: набор из одного кода, тот же самый.
    # Так колонка сразу непротиворечива, и выгрузкам не нужен особый случай
    # «старые строки без codes».
    op.execute(
        f"UPDATE {SCHEMA}.answer "  # noqa: S608
        f"SET codes = jsonb_build_array(code) WHERE code IS NOT NULL"
    )
    op.drop_constraint("attempts_positive", "answer", schema=SCHEMA, type_="check")
    op.create_check_constraint(
        "attempts_positive", "answer", "attempts >= 0", schema=SCHEMA
    )


def downgrade() -> None:
    op.execute(f"DELETE FROM {SCHEMA}.answer WHERE attempts = 0")  # noqa: S608
    op.drop_constraint("attempts_positive", "answer", schema=SCHEMA, type_="check")
    op.create_check_constraint(
        "attempts_positive", "answer", "attempts >= 1", schema=SCHEMA
    )
    op.drop_column("answer", "codes", schema=SCHEMA)
