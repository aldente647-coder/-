"""Модель данных комплекса.

Опорный раздел — ТЗ §10.1: перечень того, что должно храниться по каждому
номеру и каждой попытке. Плюс §4.2 (дедупликация, история контактов, запрет
повторного включения номера в текущую волну), §6.2 (стоп-лист на все волны),
§7.3 (версионирование анкет), §11.3 (запрет скрытого изменения исторических
данных), §14.2 (разделение контуров).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from socmon.core.codes import AaporCode, CallStage, Disposition, Outcome
from socmon.db.base import (
    CONTACT_SCHEMA,
    OPS_SCHEMA,
    SURVEY_SCHEMA,
    Base,
    TimestampedMixin,
    TZDateTime,
    VersionedMixin,
)
from socmon.domain.validity import Reason, Validity
from socmon.domain.wave import WaveStatus


def _enum(python_enum: type, name: str, schema: str) -> Enum:
    """Нативный PostgreSQL-enum. Значения берутся из value, а не из имени:
    коды AAPOR выглядят как «1.1», и именем такое не запишешь."""
    return Enum(
        python_enum,
        name=name,
        schema=schema,
        values_callable=lambda e: [member.value for member in e],
    )


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


# ===========================================================================
# Контур survey: обезличенный массив. Телефонов здесь нет ни в одном столбце.
# ===========================================================================


class Wave(Base, TimestampedMixin):
    """Двухнедельная волна — единица отчётности (ТЗ §3.1)."""

    __tablename__ = "wave"
    __table_args__ = (
        CheckConstraint("ends_at > starts_at", name="period_ordered"),
        CheckConstraint("target_max >= target_min", name="target_range"),
        {"schema": SURVEY_SCHEMA},
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    starts_at: Mapped[datetime] = mapped_column(TZDateTime)
    ends_at: Mapped[datetime] = mapped_column(TZDateTime)
    target_min: Mapped[int] = mapped_column(Integer, default=1500)
    target_max: Mapped[int] = mapped_column(Integer, default=2000)

    status: Mapped[WaveStatus] = mapped_column(
        _enum(WaveStatus, "wave_status", SURVEY_SCHEMA),
        default=WaveStatus.DRAFT, index=True,
    )
    """Состояние волны. Переходы только через `domain.wave.transition`."""

    reference_distributions: Mapped[dict[str, dict[str, float]] | None] = mapped_column(
        JSONB
    )
    """Эталонные распределения населения от Заказчика (ТЗ §5.4).

    Хранятся на волне, а не в настройках: Заказчик вправе уточнить их
    между волнами, и старые расчёты должны остаться воспроизводимыми
    со своими эталонами, а не пересчитываться задним числом.
    """
    """ТЗ §5.2: минимум 1500, целевой диапазон 1500-2000 валидных интервью."""

    closed_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    """Волна закрывается и больше не принимает интервью. Добавлять номера
    внутрь идущей волны нельзя — это ломает рамку выборки этой волны."""


class QuestionnaireVersion(Base, TimestampedMixin):
    """Снимок анкеты. Неизменяем: правки создают новую строку (ТЗ §7.3, §11.3)."""

    __tablename__ = "questionnaire_version"
    __table_args__ = (
        UniqueConstraint("wave_id", "version_id", name="uq_questionnaire_version_wave_version"),
        {"schema": SURVEY_SCHEMA},
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    wave_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SURVEY_SCHEMA}.wave.id"))
    name: Mapped[str] = mapped_column(String(200))
    version_id: Mapped[str] = mapped_column(String(16), index=True)
    """Хеш содержимого анкеты. Совпадает с Questionnaire.version_id."""

    source_yaml: Mapped[str] = mapped_column(Text)
    """Исходник целиком. Данные без анкеты, по которой они собраны,
    невоспроизводимы — а §12 требует воспроизводимости."""

    core_keys: Mapped[list[str]] = mapped_column(JSONB)
    approved_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    approved_by: Mapped[str | None] = mapped_column(String(200))
    """Волна не запускается, пока анкета не утверждена Заказчиком (ТЗ §15.3)."""


class Interview(Base, TimestampedMixin, VersionedMixin):
    """Одно интервью. Первичный ключ — псевдоним респондента (ТЗ §14.2):
    сам идентификатор ничего не говорит о человеке."""

    __tablename__ = "interview"
    __table_args__ = (
        CheckConstraint("duration_sec >= 0", name="duration_non_negative"),
        CheckConstraint(
            "weight IS NULL OR weight > 0", name="weight_positive"
        ),
        Index("ix_interview_wave_validity", "wave_id", "validity"),
        {"schema": SURVEY_SCHEMA},
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    wave_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SURVEY_SCHEMA}.wave.id"), index=True)
    questionnaire_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SURVEY_SCHEMA}.questionnaire_version.id")
    )

    started_at: Mapped[datetime] = mapped_column(TZDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    duration_sec: Mapped[float] = mapped_column(Numeric(8, 2), default=0)
    timezone: Mapped[str] = mapped_column(String(64))
    """ТЗ §10.1 требует хранить часовой пояс, а не только момент времени."""

    consent: Mapped[bool] = mapped_column(Boolean, default=False)
    outcome: Mapped[Outcome] = mapped_column(_enum(Outcome, "outcome", SURVEY_SCHEMA))
    aapor_code: Mapped[AaporCode] = mapped_column(_enum(AaporCode, "aapor_code", SURVEY_SCHEMA))
    transferred_to_operator: Mapped[bool] = mapped_column(Boolean, default=False)
    unrecognized_count: Mapped[int] = mapped_column(Integer, default=0)

    recording_uri: Mapped[str | None] = mapped_column(String(500))
    """Смешанная дорожка."""

    recording_respondent_uri: Mapped[str | None] = mapped_column(String(500))
    recording_robot_uri: Mapped[str | None] = mapped_column(String(500))
    """Раздельные каналы «респондент» и «робот» (ТЗ §10.2).

    Хранятся отдельно, а не одной ссылкой: на них диаризация тривиальна,
    и транскрипция не путает говорящих на перебиваниях. Свести две дорожки
    в одну можно всегда, разделить сведённую — уже нет.
    """

    transcript_uri: Mapped[str | None] = mapped_column(String(500))
    """Только ссылки. Медиа лежит в объектном хранилище и отдаётся подписанным
    URL с коротким сроком жизни (ТЗ §14.2)."""

    is_test: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    """Контрольный звонок. В массив и статистику не попадает никогда."""

    validity: Mapped[Validity | None] = mapped_column(
        _enum(Validity, "validity", SURVEY_SCHEMA), index=True
    )
    validity_reasons: Mapped[list[str] | None] = mapped_column(JSONB)
    validity_details: Mapped[list[str] | None] = mapped_column(JSONB)
    """Основания исключения — в выгрузку по ТЗ §13.2."""

    weight: Mapped[float | None] = mapped_column(Numeric(10, 6))
    """Вес после статистической коррекции (ТЗ §5.4). Пустой до расчёта."""

    answers: Mapped[list[Answer]] = relationship(
        back_populates="interview", cascade="all, delete-orphan"
    )
    events: Mapped[list[InterviewEvent]] = relationship(
        back_populates="interview", cascade="all, delete-orphan"
    )


class Answer(Base):
    """Ячейка массива данных. Первичный ответ респондента.

    Модель кодирования его не трогает: постанализ пишет в отдельную таблицу
    (ТЗ §11.3 — автоматические результаты маркируются, история не переписывается).
    """

    __tablename__ = "answer"
    __table_args__ = (
        CheckConstraint("code IS NULL OR (code BETWEEN 0 AND 99)", name="code_range"),
        # Ноль попыток — не ошибка, а отфильтрованный вопрос: анкета его
        # не задавала, и ставить единицу значило бы утверждать, что задавала.
        CheckConstraint("attempts >= 0", name="attempts_positive"),
        CheckConstraint(
            "code IS NOT NULL OR text IS NOT NULL", name="answer_not_empty"
        ),
        Index("ix_answer_core_key", "core_key"),
        {"schema": SURVEY_SCHEMA},
    )

    interview_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SURVEY_SCHEMA}.interview.id", ondelete="CASCADE"), primary_key=True
    )
    node_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    core_key: Mapped[str | None] = mapped_column(String(64))
    """Стабильный ключ показателя ядра. Живёт между волнами и версиями анкеты —
    по нему строятся динамические ряды (ТЗ §3.1)."""

    code: Mapped[int | None] = mapped_column(Integer)
    codes: Mapped[list[int] | None] = mapped_column(JSONB)
    """Все отмеченные варианты вопроса «до трёх ответов». У обычного вопроса
    здесь один код — тот же, что в `code`.

    Отдельная колонка, а не строки-дубли: `node_id` входит в первичный ключ,
    и второй ответ на тот же вопрос в эту таблицу просто не помещается.
    Держать же множественный выбор только в `code` — значит терять
    все названные варианты, кроме первого."""

    text: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    answered_at: Mapped[datetime] = mapped_column(TZDateTime)

    interview: Mapped[Interview] = relationship(back_populates="answers")


class InterviewEvent(Base):
    """Шкала времени интервью (ТЗ §10.1). Только вставка."""

    __tablename__ = "interview_event"
    __table_args__ = (
        Index("ix_interview_event_interview_at", "interview_id", "at"),
        {"schema": SURVEY_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    interview_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SURVEY_SCHEMA}.interview.id", ondelete="CASCADE")
    )
    at: Mapped[datetime] = mapped_column(TZDateTime)
    kind: Mapped[str] = mapped_column(String(64))
    node_id: Mapped[str | None] = mapped_column(String(64))
    detail: Mapped[str | None] = mapped_column(Text)

    interview: Mapped[Interview] = relationship(back_populates="events")


class OpenAnswerCoding(Base, TimestampedMixin):
    """Результат ночного кодирования открытого ответа (ТЗ §11.1, §11.2).

    Отдельная таблица, а не колонки в ``answer``: первичный ответ и модельная
    интерпретация не должны жить в одной строке, иначе однажды кто-нибудь
    перезапишет первое вторым. Строки только добавляются — перекодирование
    новой версией классификатора даёт новую строку, а старая остаётся.
    """

    __tablename__ = "open_answer_coding"
    __table_args__ = (
        UniqueConstraint(
            "interview_id", "node_id", "classifier_version",
            name="uq_open_answer_coding_interview_node_classifier",
        ),
        CheckConstraint("is_model_generated", name="always_marked_as_model_output"),
        {"schema": SURVEY_SCHEMA},
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    interview_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SURVEY_SCHEMA}.interview.id", ondelete="CASCADE")
    )
    node_id: Mapped[str] = mapped_column(String(64))
    classifier_version: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(200))

    themes: Mapped[list[str]] = mapped_column(JSONB, default=list)
    entities: Mapped[list[dict[str, object]]] = mapped_column(JSONB, default=list)
    """Персоны, медиа, каналы, сообщества, местные лидеры (ТЗ §11.1)."""

    sentiment: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    ambiguous: Mapped[bool] = mapped_column(Boolean, default=False)
    """Неоднозначные сущности не сливаются автоматически: нужна ручная
    проверка или согласованное правило разрешения (ТЗ §7.5)."""

    reviewed_by: Mapped[str | None] = mapped_column(String(200))
    reviewed_at: Mapped[datetime | None] = mapped_column(TZDateTime)

    is_model_generated: Mapped[bool] = mapped_column(Boolean, default=True)
    """Маркировка автоматически сгенерированных результатов (ТЗ §11.3).
    Ограничением зафиксировано в True: снять маркировку нельзя."""


# ===========================================================================
# Контур contact: номера и дозвон. Ответов здесь нет.
# ===========================================================================


class ContactBatch(Base, TimestampedMixin):
    """Загрузка базы от Заказчика (ТЗ §4.2)."""

    __tablename__ = "contact_batch"
    __table_args__ = ({"schema": CONTACT_SCHEMA},)

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(200))
    rows_total: Mapped[int] = mapped_column(Integer, default=0)
    rows_loaded: Mapped[int] = mapped_column(Integer, default=0)
    rows_duplicate: Mapped[int] = mapped_column(Integer, default=0)
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0)

    seed: Mapped[str] = mapped_column(String(64), default="")
    """Зерно случайного порядка обзвона.

    ТЗ §4.2 требует случайного порядка, §5.1 — воспроизводимости отбора
    с фиксацией алгоритма и seed. Одно без другого не работает: случайность
    без зерна невозможно перепроверить, а порядок без случайности даёт
    систематическое смещение по тому признаку, по которому база
    отсортирована у Заказчика.
    """


class Contact(Base, TimestampedMixin, VersionedMixin):
    """Абонент. Номер лежит зашифрованным, ищется по слепому индексу."""

    __tablename__ = "contact"
    __table_args__ = ({"schema": CONTACT_SCHEMA},)

    id: Mapped[uuid.UUID] = _uuid_pk()
    blind_index: Mapped[bytes] = mapped_column(LargeBinary(32), unique=True)
    """HMAC нормализованного номера. Даёт дедупликацию и поиск, не раскрывая
    номер (ТЗ §4.2, §14.2)."""

    phone_ciphertext: Mapped[bytes] = mapped_column(LargeBinary)
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(f"{CONTACT_SCHEMA}.contact_batch.id")
    )

    do_not_call: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    """Стоп-лист. Действует на все последующие волны (ТЗ §6.2)."""

    do_not_call_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    invalid_number: Mapped[bool] = mapped_column(Boolean, default=False)
    never_answers: Mapped[bool] = mapped_column(Boolean, default=False)
    """Систематически не отвечающие исключаются из обзвона (ТЗ §4.2)."""

    deactivated_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    """Контакты выключают, а не стирают: история звонков — главная ценность
    базы, и терять её нельзя."""

    dial_order: Mapped[int] = mapped_column(BigInteger, default=0, index=True)
    """Ключ случайного порядка обзвона.

    Выводится детерминированно из зерна загрузки и слепого индекса, поэтому
    порядок случаен, но воспроизводим: по зерну его можно восстановить
    и показать проверяющему (ТЗ §5.1).
    """


class StopListEvent(Base):
    """Журнал стоп-листа. Только вставка: снятие — это тоже событие."""

    __tablename__ = "stop_list_event"
    __table_args__ = (
        CheckConstraint("action IN ('added', 'removed')", name="known_action"),
        Index("ix_stop_list_event_contact_at", "contact_id", "at"),
        {"schema": CONTACT_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    contact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{CONTACT_SCHEMA}.contact.id"))
    at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now())
    action: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(String(200))
    """Снятие со стоп-листа — ответственное действие, у него должен быть автор."""


class CallAttempt(Base, TimestampedMixin, VersionedMixin):
    """Одна попытка дозвона (ТЗ §10.1).

    Ссылки на интервью здесь намеренно нет. Иначе эта таблица связала бы номер
    с ответами напрямую и обошла разделение контуров: соединение возможно
    только через ``identity_link``.
    """

    __tablename__ = "call_attempt"
    __table_args__ = (
        UniqueConstraint(
            "contact_id", "wave_id", "attempt_no", name="uq_call_attempt_contact_wave_attempt"
        ),
        CheckConstraint("attempt_no >= 1", name="attempt_no_positive"),
        CheckConstraint(
            "ended_at IS NULL OR started_at IS NULL OR ended_at >= started_at",
            name="ended_after_started",
        ),
        Index("ix_call_attempt_scheduled", "scheduled_at", "stage"),
        {"schema": CONTACT_SCHEMA},
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    contact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{CONTACT_SCHEMA}.contact.id"), index=True
    )
    wave_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    """Волна как значение, без внешнего ключа: контур ответов не должен быть
    обязательным для работы дозвона."""

    attempt_no: Mapped[int] = mapped_column(Integer)
    scheduled_at: Mapped[datetime] = mapped_column(TZDateTime)
    started_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    ended_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    duration_sec: Mapped[float | None] = mapped_column(Numeric(8, 2))

    stage: Mapped[CallStage] = mapped_column(
        _enum(CallStage, "call_stage", CONTACT_SCHEMA), default=CallStage.DIALING
    )
    disposition: Mapped[Disposition | None] = mapped_column(
        _enum(Disposition, "disposition", CONTACT_SCHEMA)
    )
    """Техническая причина обрыва. Пусто, если связь отработала штатно."""

    phone_line_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(f"{OPS_SCHEMA}.phone_line.id")
    )
    provider_call_id: Mapped[str | None] = mapped_column(String(200), index=True)
    """Идентификатор звонка на стороне движка телефонии. Поле общее: движок
    за абстракцией, и формат у разных реализаций разный."""

    is_test: Mapped[bool] = mapped_column(Boolean, default=False)


class CallEventRecord(Base):
    """Событие движка телефонии по попытке дозвона.

    Две обязанности сразу. Первая — шкала времени попытки (ТЗ §10.1).
    Вторая — защита от повторной доставки: уникальный `provider_event_id`
    делает обработку идемпотентной. Доставка «хотя бы один раз» — норма для
    вебхуков, и без этого один звонок дважды займёт оператора.

    Хранится в базе, а не в кеше с коротким TTL: у этой таблицы есть
    и аудиторская ценность, а разбор инцидента через неделю по кешу
    невозможен.
    """

    __tablename__ = "call_event"
    __table_args__ = (
        UniqueConstraint("provider_event_id", name="uq_call_event_provider_event_id"),
        Index("ix_call_event_attempt_at", "attempt_id", "at"),
        {"schema": CONTACT_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    attempt_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{CONTACT_SCHEMA}.call_attempt.id", ondelete="CASCADE")
    )
    provider_event_id: Mapped[str] = mapped_column(String(200))
    at: Mapped[datetime] = mapped_column(TZDateTime)
    kind: Mapped[str] = mapped_column(String(64))
    stage_before: Mapped[str] = mapped_column(String(32))
    stage_after: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str | None] = mapped_column(Text)


class IdentityLink(Base, TimestampedMixin):
    """Единственный мост между контурами (ТЗ §14.2).

    Здесь и только здесь можно узнать, какому абоненту принадлежит интервью.
    Права на таблицу выдаются отдельной ролью; аналитик их не получает.

    Уникальность по паре (контакт, волна) исполняет ТЗ §4.2: повторно включить
    номер в текущую волну после состоявшегося интервью нельзя.
    """

    __tablename__ = "identity_link"
    __table_args__ = (
        UniqueConstraint("contact_id", "wave_id", name="uq_identity_link_contact_wave"),
        {"schema": CONTACT_SCHEMA},
    )

    interview_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    contact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{CONTACT_SCHEMA}.contact.id"))
    attempt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{CONTACT_SCHEMA}.call_attempt.id"))
    wave_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))


# ===========================================================================
# Контур ops: эксплуатация
# ===========================================================================


class PhoneLine(Base, TimestampedMixin):
    """Исходящий номер из «карусели».

    Выгоревший номер выводится из ротации, но не удаляется — иначе теряется
    история звонков, сделанных с него.
    """

    __tablename__ = "phone_line"
    __table_args__ = ({"schema": OPS_SCHEMA},)

    id: Mapped[uuid.UUID] = _uuid_pk()
    msisdn: Mapped[str] = mapped_column(String(20), unique=True)
    provider: Mapped[str] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    retired_at: Mapped[datetime | None] = mapped_column(TZDateTime)
    retired_reason: Mapped[str | None] = mapped_column(Text)


class User(Base, TimestampedMixin):
    """Учётная запись кабинета.

    Роль — не украшение интерфейса, а граница доступа к персональным данным.
    ТЗ §14.2 требует разграничения прав и раздельного хранения телефонных
    номеров и ответов; роль решает, увидит ли человек номер вообще.

    Учётные записи выключаются, а не удаляются: журнал действий ссылается
    на автора, и стирание пользователя обрывает аудиторский след.
    """

    __tablename__ = "app_user"
    __table_args__ = ({"schema": OPS_SCHEMA},)

    id: Mapped[uuid.UUID] = _uuid_pk()
    login: Mapped[str] = mapped_column(String(100), unique=True)
    full_name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(32), index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    """scrypt: соль и параметры внутри строки. Открытых паролей в базе нет."""

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    session_version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    last_login_at: Mapped[datetime | None] = mapped_column(TZDateTime)


class AuditLog(Base):
    """Журнал значимых действий (ТЗ §11.3). Только вставка."""

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_entity", "entity_type", "entity_id"),
        {"schema": OPS_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(TZDateTime, server_default=func.now(), index=True)
    actor: Mapped[str] = mapped_column(String(200))
    action: Mapped[str] = mapped_column(String(100))
    entity_type: Mapped[str] = mapped_column(String(100))
    entity_id: Mapped[str] = mapped_column(String(100))
    diff: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    """Разница до/после. Чувствительные значения маскируются на записи,
    иначе журнал сам становится местом утечки."""


VALIDITY_REASONS = tuple(r.value for r in Reason)
"""Справочник оснований исключения для кодбука (ТЗ §13.2)."""
