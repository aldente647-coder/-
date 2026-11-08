"""Кабинет: границы доступа и журнал.

Главное, что здесь проверяется, — не вёрстка, а то, что роль действительно
ограничивает доступ к персональным данным. Спрятать ссылку в меню
не защита: адрес можно набрать руками.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote_plus

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from socmon.core.engine import Utterance
from socmon.db.models import AuditLog, User, Wave
from socmon.web.app import create_app
from socmon.web.artifacts import Artifact, build_archive, existing_archive
from socmon.web.security import (
    ROLE_PERMISSIONS,
    Permission,
    Principal,
    Role,
    hash_password,
    mask_phone,
    verify_password,
)
from tests.conftest import needs_db
from tests.test_db_constraints import make_interview, make_wave

pytestmark = [needs_db, pytest.mark.usefixtures("engine")]

PASSWORD = "длинный-пароль-кабинета"


@pytest.fixture
def factory(session: Session) -> sessionmaker[Session]:
    """Сессии приложения на том же соединении, что и тестовая.

    Иначе приложение работает в отдельной транзакции и не видит данных,
    подготовленных тестом: фикстура сессии живёт во вложенной транзакции
    с откатом, и её коммиты наружу не выходят.
    """
    from sqlalchemy.orm import sessionmaker as maker

    return maker(
        bind=session.get_bind(),
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
    )


@pytest.fixture
def client(factory: sessionmaker[Session]) -> Iterator[TestClient]:
    with TestClient(
        create_app(factory, secret_key="тест" * 8), base_url="https://testserver",
    ) as test_client:
        response = test_client.get("/login")
        token = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
        assert token is not None
        test_client.headers["x-csrf-token"] = token.group(1)
        yield test_client


def make_user(session: Session, role: Role, *, active: bool = True) -> User:
    user = User(
        login=f"{role.value}-{uuid.uuid4().hex[:6]}",
        full_name=f"Пользователь {role.value}",
        role=role.value,
        password_hash=hash_password(PASSWORD),
        is_active=active,
    )
    session.add(user)
    session.commit()
    return user


def sign_in(client: TestClient, user: User) -> None:
    response = client.post(
        "/login", data={"login": user.login, "password": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 303


# --------------------------------------------------------------------- вход


def test_anonymous_visitor_gets_no_data(client: TestClient) -> None:
    assert client.get("/interviews").status_code == 401
    assert client.get("/audit").status_code == 401


def test_wrong_password_is_refused(client: TestClient, session: Session) -> None:
    user = make_user(session, Role.ADMIN)
    response = client.post("/login", data={"login": user.login, "password": "неверный"})
    assert response.status_code == 401


def test_error_message_does_not_reveal_who_exists(
    client: TestClient, session: Session
) -> None:
    """Иначе форма входа превращается в способ узнать, кто заведён."""
    user = make_user(session, Role.ADMIN)
    unknown = client.post("/login", data={"login": "нет-такого", "password": "x" * 12})
    wrong = client.post("/login", data={"login": user.login, "password": "x" * 12})
    assert unknown.text.count("Неверный логин или пароль") == 1
    assert wrong.text.count("Неверный логин или пароль") == 1


def test_disabled_account_stops_working_immediately(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    """Выключение действует сразу, а не после истечения сессии."""
    user = make_user(session, Role.ADMIN)
    sign_in(client, user)
    assert client.get("/interviews").status_code == 200

    with factory() as other:
        other.get(User, user.id).is_active = False  # type: ignore[union-attr]
        other.commit()
    # Титульная страница открыта всем, поэтому проверяем закрытую.
    assert client.get("/interviews").status_code == 401


def test_login_is_journalled(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    user = make_user(session, Role.ADMIN)
    sign_in(client, user)
    with factory() as other:
        actions = other.scalars(select(AuditLog.action)).all()
    assert "login" in actions


# ------------------------------------------------------------- границы ролей


@pytest.mark.parametrize(
    ("role", "path", "allowed"),
    [
        (Role.ANALYST, "/interviews", True),
        (Role.ANALYST, "/dialling", False),   # аналитику номера не нужны
        (Role.ANALYST, "/users", False),
        (Role.ANALYST, "/audit", False),
        (Role.OPERATOR, "/dialling", False),  # оператор не управляет обзвоном
        (Role.OPERATOR, "/interviews", False),
        (Role.CODER, "/interviews", True),
        (Role.CODER, "/users", False),
        (Role.AUDITOR, "/audit", True),
        (Role.AUDITOR, "/users", False),
        (Role.CUSTOMER, "/audit", True),
        (Role.CUSTOMER, "/users", False),
        (Role.ADMIN, "/users", True),
        (Role.ADMIN, "/dialling", True),
    ],
)
def test_route_is_guarded_by_permission_not_by_menu(
    client: TestClient, session: Session, role: Role, path: str, allowed: bool
) -> None:
    sign_in(client, make_user(session, role))
    response = client.get(path)
    assert (response.status_code == 200) is allowed


def test_analyst_never_sees_phone_numbers() -> None:
    """ТЗ §14.2: номера отделены от ответов. Для аналитика это не настройка
    интерфейса, а отсутствующее право."""
    analyst = Principal("1", "a", "Аналитик", Role.ANALYST)
    assert not analyst.sees_phone_numbers
    assert Permission.VIEW_PHONE not in analyst.permissions


def test_customer_may_listen_but_not_see_numbers() -> None:
    """Права раздельные: уровень доступа тут не помог бы — «главнее»
    не значит «нужнее»."""
    customer = Principal("1", "c", "Заказчик", Role.CUSTOMER)
    assert customer.can(Permission.LISTEN_RECORDING)
    assert not customer.can(Permission.VIEW_PHONE)


def test_only_admin_holds_every_permission() -> None:
    for role in Role:
        if role is Role.ADMIN:
            assert ROLE_PERMISSIONS[role] == frozenset(Permission)
        else:
            assert ROLE_PERMISSIONS[role] != frozenset(Permission)


# ------------------------------------------------------------------- записи


def test_listening_to_a_recording_is_journalled(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    """ТЗ §14.2 требует журнала доступа, §13.4 — просмотра записи при
    наличии уровня доступа. Значит записывается и просмотр, а не только
    изменение."""
    wave = make_wave(session, f"w-{uuid.uuid4().hex[:6]}")
    interview = make_interview(session, wave, recording_uri="s3://rec/1.wav")
    session.commit()

    sign_in(client, make_user(session, Role.ADMIN))
    assert client.get(f"/interviews/{interview.id}/recording").status_code == 200

    with factory() as other:
        entry = other.scalar(
            select(AuditLog).where(AuditLog.action == "listen_recording")
        )
    assert entry is not None
    assert entry.entity_id == str(interview.id)


def test_role_without_the_right_cannot_reach_the_recording(
    client: TestClient, session: Session
) -> None:
    wave = make_wave(session, f"w-{uuid.uuid4().hex[:6]}")
    interview = make_interview(session, wave, recording_uri="s3://rec/1.wav")
    session.commit()

    sign_in(client, make_user(session, Role.ANALYST))
    assert client.get(f"/interviews/{interview.id}/recording").status_code == 403


def test_missing_recording_is_reported_not_invented(
    client: TestClient, session: Session
) -> None:
    wave = make_wave(session, f"w-{uuid.uuid4().hex[:6]}")
    interview = make_interview(session, wave)
    session.commit()
    sign_in(client, make_user(session, Role.ADMIN))
    assert client.get(f"/interviews/{interview.id}/recording").status_code == 404


# ------------------------------------------------------------- пользователи


def test_admin_cannot_lock_himself_out(
    client: TestClient, session: Session
) -> None:
    admin = make_user(session, Role.ADMIN)
    sign_in(client, admin)
    response = client.post(f"/users/{admin.id}/toggle", follow_redirects=False)
    assert response.status_code == 400


def test_disabling_a_user_is_journalled_with_the_diff(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    sign_in(client, make_user(session, Role.ADMIN))
    victim = make_user(session, Role.ANALYST)

    client.post(f"/users/{victim.id}/toggle", follow_redirects=False)
    with factory() as other:
        entry = other.scalar(select(AuditLog).where(AuditLog.action == "toggle_user"))
        refreshed = other.get(User, victim.id)
    assert refreshed is not None and refreshed.is_active is False
    assert entry is not None and entry.diff == {
        "is_active": {"before": True, "after": False}
    }


# --------------------------------------------------------------------- пароли


def test_password_is_never_stored_in_the_clear() -> None:
    stored = hash_password(PASSWORD)
    assert PASSWORD not in stored
    assert stored.startswith("scrypt$")


def test_short_password_is_refused() -> None:
    with pytest.raises(ValueError, match="короче 12"):
        hash_password("коротко")


def test_corrupted_hash_does_not_let_anyone_in() -> None:
    assert verify_password(PASSWORD, "мусор") is False
    assert verify_password(PASSWORD, "scrypt$broken") is False


def test_masked_number_keeps_only_the_tail() -> None:
    """Хвоста хватает свериться с абонентом и не хватает унести базу."""
    masked = mask_phone("+79491234567")
    assert masked.endswith("4567")
    assert "9491" not in masked


# ================================================================ аналитика


REFERENCE_YAML = Path("src/socmon/questionnaires/core-v1.yaml").read_text(
    encoding="utf-8"
)


def prepare_wave(
    session: Session, *, reference: dict[str, dict[str, float]] | None
) -> Wave:
    """Волна с настоящей анкетой и, при необходимости, эталонами.

    Именно настоящей: с заглушкой вместо YAML аналитика падает на разборе
    анкеты и до проверяемого поведения не доходит.
    """
    from tests.test_db_constraints import make_version

    wave = make_wave(session, f"w-{uuid.uuid4().hex[:6]}")
    wave.reference_distributions = reference
    make_version(session, wave, source_yaml=REFERENCE_YAML)
    session.commit()
    return wave


def test_analytics_refuses_to_count_without_reference_distributions(
    client: TestClient, session: Session
) -> None:
    """Показать невзвешенные цифры как взвешенные — худшее, что можно
    сделать. Отказ считать здесь важнее результата (ТЗ §5.4)."""
    prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.ANALYST))

    response = client.get("/analytics")
    assert response.status_code == 200
    assert "эталонные распределения" in response.text
    assert "Взвешивание" in response.text


def test_analytics_reports_an_empty_wave_plainly(
    client: TestClient, session: Session
) -> None:
    wave = prepare_wave(session, reference={"gender": {"Мужской": 0.46, "Женский": 0.54}})
    sign_in(client, make_user(session, Role.ANALYST))

    response = client.get(f"/analytics?wave={wave.code}")
    assert response.status_code == 200
    assert "нет валидных интервью" in response.text


def test_analytics_is_closed_to_roles_without_the_right(
    client: TestClient, session: Session
) -> None:
    sign_in(client, make_user(session, Role.OPERATOR))
    assert client.get("/analytics").status_code == 403


# =================================================================== разбор


def test_coding_queue_is_closed_to_the_customer(
    client: TestClient, session: Session
) -> None:
    """Ручная проверка разметки — работа кодировщика, а не руководителя."""
    sign_in(client, make_user(session, Role.CUSTOMER))
    assert client.get("/coding").status_code == 403


def test_coder_sees_the_review_queue(client: TestClient, session: Session) -> None:
    prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.CODER))
    assert client.get("/coding").status_code == 200


def test_marking_reviewed_keeps_the_output_model_generated(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    """§11.3: автоматически сгенерированное должно оставаться отличимым
    от проверенного, а не подменяться им."""
    from socmon.db.models import OpenAnswerCoding

    wave = prepare_wave(session, reference=None)
    interview = make_interview(session, wave)
    row = OpenAnswerCoding(
        interview_id=interview.id, node_id="q_open_worry",
        classifier_version="abc123", model="rules-v1",
        themes=["prices"], entities=[], ambiguous=True,
    )
    session.add(row)
    session.commit()

    coder = make_user(session, Role.CODER)
    sign_in(client, coder)
    assert client.post(f"/coding/{row.id}/reviewed", follow_redirects=False).status_code == 303

    with factory() as other:
        refreshed = other.get(OpenAnswerCoding, row.id)
        assert refreshed is not None
        assert refreshed.reviewed_by == coder.login
        assert refreshed.reviewed_at is not None
        assert refreshed.is_model_generated is True
        assert refreshed.themes == ["prices"]

        entry = other.scalar(select(AuditLog).where(AuditLog.action == "review_coding"))
    assert entry is not None


def test_reviewing_a_missing_coding_is_a_404(
    client: TestClient, session: Session
) -> None:
    sign_in(client, make_user(session, Role.CODER))
    response = client.post(f"/coding/{uuid.uuid4()}/reviewed", follow_redirects=False)
    assert response.status_code == 404


# ================================================================= выгрузки


def test_export_without_reference_distributions_explains_itself(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    """Не пятисотка: причина понятна человеку, и он может её устранить."""
    wave = prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.CUSTOMER))

    response = client.post(f"/exports/{wave.id}", follow_redirects=False)
    assert response.status_code == 303
    assert "error=" in response.headers["location"]

    with factory() as other:
        assert other.scalar(
            select(AuditLog).where(AuditLog.action == "export_package")
        ) is None  # неудавшаяся сборка выгрузкой не считается


def test_export_is_closed_to_the_operator(
    client: TestClient, session: Session
) -> None:
    sign_in(client, make_user(session, Role.OPERATOR))
    assert client.get("/exports").status_code == 403


def test_analyst_may_export_the_anonymised_dataset(
    client: TestClient, session: Session
) -> None:
    """Массив обезличен, поэтому выгрузка аналитику доступна,
    а номера — нет."""
    prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.ANALYST))
    assert client.get("/exports").status_code == 200


def test_menu_only_shows_what_the_role_may_open(
    client: TestClient, session: Session
) -> None:
    """Меню и права должны совпадать — расхождение сбивает с толку,
    даже если маршрут защищён."""
    sign_in(client, make_user(session, Role.ANALYST))
    body = client.get("/").text
    assert 'href="/analytics"' in body
    assert 'href="/exports"' in body
    assert 'href="/coding"' not in body
    assert 'href="/users"' not in body


# =================================================================== волны


def test_wave_management_is_closed_to_the_analyst(
    client: TestClient, session: Session
) -> None:
    sign_in(client, make_user(session, Role.ANALYST))
    assert client.get("/waves").status_code == 403


def test_customer_creates_a_wave_as_a_draft(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    """Черновик, а не активная: волна запускается только после проверки
    готовности."""
    from socmon.domain.wave import WaveStatus

    sign_in(client, make_user(session, Role.CUSTOMER))
    code = f"w-{uuid.uuid4().hex[:6]}"
    response = client.post(
        "/waves", data={"code": code, "name": "Тестовая"}, follow_redirects=False
    )
    assert response.status_code == 303

    with factory() as other:
        row = other.scalar(select(Wave).where(Wave.code == code))
        assert row is not None
        assert row.status is WaveStatus.DRAFT
        assert (row.ends_at - row.starts_at).days == 14  # ТЗ §3.1
        assert other.scalar(select(AuditLog).where(AuditLog.action == "create_wave"))


def test_duplicate_wave_code_is_refused(
    client: TestClient, session: Session
) -> None:
    wave = prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.CUSTOMER))
    response = client.post(
        "/waves", data={"code": wave.code, "name": "Дубль"}, follow_redirects=False
    )
    assert response.status_code == 400


def test_unprepared_wave_cannot_be_started(
    client: TestClient, session: Session
) -> None:
    """Запустив волну с устаревшей озвучкой, мы обзвоним тысячу человек
    не тем вопросом."""
    wave = prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.CUSTOMER))

    response = client.post(
        f"/waves/{wave.id}/status", data={"target": "active"}, follow_redirects=False
    )
    assert response.status_code == 400
    assert "не готова" in response.json()["detail"]


def test_closed_wave_cannot_be_reopened(
    client: TestClient, session: Session
) -> None:
    """По закрытой волне уже сдан оперативный пакет (ТЗ §13.1)."""
    from socmon.domain.wave import WaveStatus

    wave = prepare_wave(session, reference=None)
    wave.status = WaveStatus.CLOSED
    session.commit()

    sign_in(client, make_user(session, Role.CUSTOMER))
    response = client.post(
        f"/waves/{wave.id}/status", data={"target": "active"}, follow_redirects=False
    )
    assert response.status_code == 400


def test_closing_a_wave_stamps_the_time_and_journals_it(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    from socmon.domain.wave import WaveStatus

    wave = prepare_wave(session, reference=None)
    wave.status = WaveStatus.ACTIVE
    session.commit()

    sign_in(client, make_user(session, Role.CUSTOMER))
    assert client.post(
        f"/waves/{wave.id}/status", data={"target": "closed"}, follow_redirects=False
    ).status_code == 303

    with factory() as other:
        row = other.get(Wave, wave.id)
        assert row is not None
        assert row.status is WaveStatus.CLOSED
        assert row.closed_at is not None
        entry = other.scalar(select(AuditLog).where(AuditLog.action == "wave_status"))
    assert entry is not None
    assert entry.diff == {"status": {"before": "active", "after": "closed"}}


def test_unknown_status_is_refused(client: TestClient, session: Session) -> None:
    wave = prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.CUSTOMER))
    response = client.post(
        f"/waves/{wave.id}/status", data={"target": "летит"}, follow_redirects=False
    )
    assert response.status_code == 400


def test_readiness_blockers_are_shown_on_the_page(
    client: TestClient, session: Session
) -> None:
    """Система обязана показать, чего не хватает, а не молча не поехать."""
    prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.CUSTOMER))

    body = client.get("/waves").text
    # Число с правильной формой слова: «1 препятствий» читается как недоделка.
    assert "препятстви" in body
    assert "эталонные распределения" in body or "взвешивание" in body


# ================================================================== табло


def test_board_shows_live_stages(client: TestClient, session: Session) -> None:
    wave = prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.ADMIN))

    body = client.get(f"/board?wave={wave.code}").text
    assert "Сейчас на линиях" in body
    assert "гудки" in body
    assert 'http-equiv="refresh"' in body  # обновляется само, без JavaScript


def test_board_is_open_to_the_operator(client: TestClient, session: Session) -> None:
    """Дежурному оператору табло нужно: по нему видно, ждать ли перевода."""
    prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.OPERATOR))
    assert client.get("/board").status_code == 200


# ======================================================= выгрузка файлов

MARGINS = {
    "gender": {"Мужской": 0.46, "Женский": 0.54},
    "age_group": {"18-34": 0.30, "35-54": 0.36, "55+": 0.34},
    "territory": {"Северный": 0.30, "Центральный": 0.42, "Южный": 0.28},
}

SCRIPT = {
    "consent": "да", "screen_age": "да", "screen_territory": "да",
    "q_life": "шесть", "q_family_change": "не изменилось", "q_material": "четыре",
    "q_safety": "семь", "q_anxiety": "восемь", "q_trust": "четыре",
    "q_fairness": "пять", "q_conflict": "шесть", "q_authorities": "пять",
    "q_local_problem": "цены", "q_collective": "скорее готов",
    "q_open_leader": "никого", "q_open_worry": "цены выросли",
    "d_gender": "женский", "d_age_group": "мне сорок два",
    "d_territory": "центральный", "d_settlement": "город",
    "d_education": "высшее", "d_employment": "работаю",
    "d_nationality": "русский", "d_politics": "иногда",
}


def seed_interviews(session: Session, wave: Wave, count: int = 30) -> None:
    """Наполнить волну настоящими интервью через движок анкеты."""
    import os

    from socmon.core.engine import InterviewSession
    from socmon.core.questionnaire import Questionnaire
    from socmon.db.models import CallAttempt, Contact, ContactBatch, QuestionnaireVersion
    from socmon.db.storage import CallContext, save_interview

    questionnaire = Questionnaire.from_yaml("src/socmon/questionnaires/core-v1.yaml")
    version = session.scalar(
        select(QuestionnaireVersion).where(QuestionnaireVersion.wave_id == wave.id)
    )
    assert version is not None
    batch = ContactBatch(name="тест", source="тест", rows_total=count)
    session.add(batch)
    session.flush()

    genders = ["женский", "мужской"]
    ages = ["мне двадцать", "мне сорок два", "мне шестьдесят"]
    territories = ["северный", "центральный", "южный"]

    for i in range(count):
        contact = Contact(
            batch_id=batch.id, blind_index=os.urandom(32),
            phone_ciphertext=os.urandom(40),
        )
        session.add(contact)
        session.flush()
        attempt = CallAttempt(
            contact_id=contact.id, wave_id=wave.id, attempt_no=1,
            scheduled_at=datetime.now(UTC),
        )
        session.add(attempt)
        session.flush()

        interview = InterviewSession(questionnaire)
        interview.begin()
        answers = dict(
            SCRIPT, d_gender=genders[i % 2], d_age_group=ages[i % 3],
            d_territory=territories[(i // 5) % 3],
        )
        while interview.current_node_id:
            interview.submit(Utterance.said(answers[interview.current_node_id]))
        save_interview(session, interview, CallContext(
            wave_id=wave.id, questionnaire_version_id=version.id,
            contact_id=contact.id, attempt_id=attempt.id,
        ))
    session.commit()


def test_package_is_built_and_downloaded(
    client: TestClient, session: Session, factory: sessionmaker[Session],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сквозная проверка: из базы в архив и обратно к человеку."""
    import zipfile

    from socmon.web import app as web_app

    monkeypatch.setattr(web_app, "build_archive", _archive_into(tmp_path))
    monkeypatch.setattr(web_app, "existing_archive", _archive_from(tmp_path))

    wave = prepare_wave(session, reference=MARGINS)
    seed_interviews(session, wave)
    sign_in(client, make_user(session, Role.CUSTOMER))

    built = client.post(f"/exports/{wave.id}", follow_redirects=False)
    assert built.status_code == 303
    assert "error=" not in built.headers["location"]

    downloaded = client.get(f"/exports/{wave.id}/download")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "application/zip"

    archive = tmp_path / f"wave-{wave.code}.zip"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert any("массив" in n for n in names)
    assert any("кодбук" in n for n in names)
    assert any(n.endswith(".xlsx") for n in names)
    assert any(n.endswith(".docx") for n in names)

    with factory() as other:
        actions = set(other.scalars(select(AuditLog.action)).all())
    assert {"export_package", "download_package"} <= actions


def _archive_into(root: Path) -> Callable[[Session, Wave], Artifact]:
    """Сборка в каталог теста, а не в рабочий: выгрузки волн не должны
    смешиваться между прогонами."""
    def wrapper(db: Session, wave: Wave) -> Artifact:
        return build_archive(db, wave, root=root)

    return wrapper


def _archive_from(root: Path) -> Callable[[str], Artifact | None]:
    def wrapper(code: str) -> Artifact | None:
        return existing_archive(code, root=root)

    return wrapper


def test_download_before_build_is_a_404(
    client: TestClient, session: Session, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from socmon.web import app as web_app

    monkeypatch.setattr(web_app, "existing_archive", _archive_from(tmp_path))
    wave = prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.CUSTOMER))
    assert client.get(f"/exports/{wave.id}/download").status_code == 404


def test_download_is_closed_to_roles_without_the_right(
    client: TestClient, session: Session
) -> None:
    wave = prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.OPERATOR))
    assert client.get(f"/exports/{wave.id}/download").status_code == 403


# ================================================================== анкеты


def test_questionnaire_page_is_closed_to_the_analyst(
    client: TestClient, session: Session
) -> None:
    sign_in(client, make_user(session, Role.ANALYST))
    assert client.get("/questionnaires").status_code == 403


def test_uploading_a_questionnaire_creates_a_version(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    from socmon.core.questionnaire import Questionnaire
    from socmon.db.models import QuestionnaireVersion

    wave = make_wave(session, f"w-{uuid.uuid4().hex[:6]}")
    session.commit()
    sign_in(client, make_user(session, Role.CUSTOMER))

    response = client.post(
        f"/questionnaires/{wave.id}/upload",
        files={"file": ("core.yaml", REFERENCE_YAML.encode(), "application/x-yaml")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=" not in response.headers["location"]

    expected = Questionnaire.from_yaml("src/socmon/questionnaires/core-v1.yaml")
    with factory() as other:
        row = other.scalar(
            select(QuestionnaireVersion).where(QuestionnaireVersion.wave_id == wave.id)
        )
    assert row is not None
    assert row.version_id == expected.version_id


def test_broken_questionnaire_is_not_saved(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    """Не прошедшая проверку анкета не сохраняется: иначе волну можно
    запустить с графом, из которого нет выхода."""
    from socmon.db.models import QuestionnaireVersion

    wave = make_wave(session, f"w-{uuid.uuid4().hex[:6]}")
    session.commit()
    sign_in(client, make_user(session, Role.CUSTOMER))

    response = client.post(
        f"/questionnaires/{wave.id}/upload",
        files={"file": ("bad.yaml", b"name: 'kek'\nnodes: []\n", "application/x-yaml")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=" in response.headers["location"]

    with factory() as other:
        assert other.scalar(
            select(QuestionnaireVersion).where(QuestionnaireVersion.wave_id == wave.id)
        ) is None


def test_questionnaire_cannot_be_swapped_mid_wave(
    client: TestClient, session: Session
) -> None:
    """Половина интервью по одной редакции вопросов и половина по другой —
    этого никакая версия уже не разъяснит (ТЗ §3.1, §7.3)."""
    from socmon.domain.wave import WaveStatus

    wave = prepare_wave(session, reference=None)
    wave.status = WaveStatus.ACTIVE
    session.commit()
    sign_in(client, make_user(session, Role.CUSTOMER))

    response = client.post(
        f"/questionnaires/{wave.id}/upload",
        files={"file": ("core.yaml", REFERENCE_YAML.encode(), "application/x-yaml")},
        follow_redirects=False,
    )
    # Сообщение уезжает в адрес процентным кодированием — сверяем по смыслу.
    location = unquote_plus(response.headers["location"])
    assert "error=" in response.headers["location"]
    assert "черновик" in location


def test_approving_a_version_stamps_who_and_when(
    client: TestClient, session: Session, factory: sessionmaker[Session]
) -> None:
    from socmon.db.models import QuestionnaireVersion

    wave = prepare_wave(session, reference=None)
    version = session.scalar(
        select(QuestionnaireVersion).where(QuestionnaireVersion.wave_id == wave.id)
    )
    assert version is not None
    sign_in(client, make_user(session, Role.CUSTOMER))

    assert client.post(
        f"/questionnaires/{version.id}/approve", follow_redirects=False
    ).status_code == 303

    with factory() as other:
        row = other.get(QuestionnaireVersion, version.id)
        assert row is not None
        assert row.approved_at is not None
        assert row.approved_by
        assert other.scalar(
            select(AuditLog).where(AuditLog.action == "approve_questionnaire")
        )


def test_double_approval_is_refused(
    client: TestClient, session: Session
) -> None:
    from socmon.db.models import QuestionnaireVersion

    wave = prepare_wave(session, reference=None)
    version = session.scalar(
        select(QuestionnaireVersion).where(QuestionnaireVersion.wave_id == wave.id)
    )
    assert version is not None
    sign_in(client, make_user(session, Role.CUSTOMER))

    client.post(f"/questionnaires/{version.id}/approve", follow_redirects=False)
    again = client.post(f"/questionnaires/{version.id}/approve", follow_redirects=False)
    assert again.status_code == 400


def test_stale_voice_is_shown_as_a_blocker_on_the_page(
    client: TestClient, session: Session
) -> None:
    """Самый опасный случай, и он должен быть виден крупно, а не строкой
    в таблице."""
    prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.CUSTOMER))
    body = client.get("/questionnaires").text
    assert "Озвучка" in body or "озвучк" in body


# =================================================================== база


def test_contact_base_is_closed_to_the_analyst(
    client: TestClient, session: Session
) -> None:
    sign_in(client, make_user(session, Role.ANALYST))
    assert client.get("/contacts").status_code == 403


def test_upload_refuses_to_work_without_encryption_keys(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Класть номера открытым текстом нельзя ни при каких обстоятельствах
    (ТЗ §14.2)."""
    monkeypatch.delenv("SOCMON_PHONE_INDEX_KEY", raising=False)
    monkeypatch.delenv("SOCMON_PHONE_ENC_KEY", raising=False)
    sign_in(client, make_user(session, Role.CUSTOMER))

    response = client.post(
        "/contacts/upload",
        files={"file": ("base.csv", b"+79491234567\n", "text/csv")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert "SOCMON_PHONE" in unquote_plus(response.headers["location"])


def test_uploading_a_base_loads_and_journals_it(
    client: TestClient, session: Session, factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from socmon.db.models import Contact, ContactBatch

    monkeypatch.setenv("SOCMON_PHONE_INDEX_KEY", "aa" * 32)
    monkeypatch.setenv("SOCMON_PHONE_ENC_KEY", "bb" * 32)
    sign_in(client, make_user(session, Role.CUSTOMER))

    payload = "phone\n+79491234567\n89491234568\nне номер\n+79491234567\n"
    response = client.post(
        "/contacts/upload",
        files={"file": ("base.csv", payload.encode(), "text/csv")},
        data={"name": "Поставка 1"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=" not in response.headers["location"]

    with factory() as other:
        batch = other.scalar(select(ContactBatch).where(ContactBatch.name == "Поставка 1"))
        assert batch is not None
        assert batch.rows_loaded == 2      # два разных номера
        assert batch.rows_rejected == 1    # «не номер»
        assert batch.rows_duplicate == 1   # повтор в файле
        assert batch.seed

        stored = other.scalars(select(Contact)).all()
        assert all(b"9491234567" not in c.phone_ciphertext for c in stored)

        entry = other.scalar(
            select(AuditLog).where(AuditLog.action == "import_contacts")
        )
    assert entry is not None
    assert entry.diff is not None and entry.diff["loaded"] == 2


def test_oversized_upload_is_sent_to_the_command_line(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Держать миллион номеров в памяти веб-процесса незачем."""
    monkeypatch.setenv("SOCMON_PHONE_INDEX_KEY", "aa" * 32)
    monkeypatch.setenv("SOCMON_PHONE_ENC_KEY", "bb" * 32)
    from socmon.web import app as web_app

    monkeypatch.setattr(web_app, "MAX_CONTACTS_BYTES", 64)
    sign_in(client, make_user(session, Role.CUSTOMER))

    response = client.post(
        "/contacts/upload",
        files={"file": ("big.csv", b"+79491234567\n" * 50, "text/csv")},
        follow_redirects=False,
    )
    assert "socmon contacts import" in unquote_plus(response.headers["location"])


def test_non_utf8_file_is_reported_not_mangled(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOCMON_PHONE_INDEX_KEY", "aa" * 32)
    monkeypatch.setenv("SOCMON_PHONE_ENC_KEY", "bb" * 32)
    sign_in(client, make_user(session, Role.CUSTOMER))

    response = client.post(
        "/contacts/upload",
        files={"file": ("base.csv", "телефон\n+79491234567\n".encode("cp1251"), "text/csv")},
        follow_redirects=False,
    )
    assert "UTF-8" in unquote_plus(response.headers["location"])


# ============================================== что видит человек в браузере


def test_browser_without_login_gets_the_form_not_json(client: TestClient) -> None:
    """Голый {"detail": "..."} в окне браузера формально верен, а на деле
    человек видит непонятный текст и решает, что ничего не работает.
    Поймано на живом стенде."""
    response = client.get(
        "/interviews", headers={"Accept": "text/html"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


@pytest.fixture
def demo_media(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Озвучка демо-анкеты для тестов демонстрации.

    WAV в репозитории не хранятся (`media/` в .gitignore), поэтому на чистой
    машине библиотеку собирает подставной синтезатор: тестам нужна полнота
    библиотеки, а не звучание. Так демо проверяется одинаково везде, а не
    только там, где озвучка случайно лежит в рабочем каталоге.
    """
    from socmon.core.questionnaire import Questionnaire
    from socmon.voice.render import render, render_catalogue
    from socmon.web import app as web_app
    from tests.test_voice import FakeSynthesizer

    media = tmp_path_factory.mktemp("media")
    synthesizer = FakeSynthesizer()
    voice = web_app.DEFAULT_DEMO_VOICE
    render(Questionnaire.from_yaml(web_app.DEMO_QUESTIONNAIRE), synthesizer, media, voice=voice)
    render_catalogue(
        web_app.DELIVERY_CATALOGUE, synthesizer, media,
        voice=voice, gender=web_app.VOICE_GENDER.get(voice, "f"),
    )
    monkeypatch.setattr(web_app, "MEDIA_DIR", media)
    return media


def test_root_is_one_entry_point_for_everyone(client: TestClient, demo_media: Path) -> None:
    """Одна ссылка на всё: гость видит оба пути и не должен угадывать
    адреса разделов."""
    response = client.get("/", headers={"Accept": "text/html"})
    assert response.status_code == 200
    assert 'href="/demo"' in response.text
    assert 'href="/login"' in response.text


def test_root_shows_the_dashboard_to_those_who_entered(
    client: TestClient, session: Session
) -> None:
    sign_in(client, make_user(session, Role.ADMIN))
    body = client.get("/").text
    assert "Обзор" in body
    assert 'href="/analytics"' in body


def test_role_without_dashboard_lands_on_the_demo(
    client: TestClient, session: Session
) -> None:
    """Кодировщику сводка по волне не положена — но и в тупик его
    отправлять незачем."""
    sign_in(client, make_user(session, Role.CODER))
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/demo"


def test_api_client_still_gets_json(client: TestClient) -> None:
    """Программе за данными страница не нужна."""
    response = client.get("/interviews", headers={"Accept": "application/json"})
    assert response.status_code == 401
    assert response.json()["detail"]


def test_login_returns_to_the_page_you_wanted(
    client: TestClient, session: Session
) -> None:
    response = client.get(
        "/analytics", headers={"Accept": "text/html"}, follow_redirects=False
    )
    assert "next=%2Fanalytics" in response.headers["location"]

    user = make_user(session, Role.ADMIN)
    landed = client.post(
        "/login",
        data={"login": user.login, "password": PASSWORD, "next": "/analytics"},
        follow_redirects=False,
    )
    assert landed.headers["location"] == "/analytics"


def test_login_never_redirects_off_site(
    client: TestClient, session: Session
) -> None:
    """Открытый редирект по чужому адресу — это фишинг с нашего адреса."""
    user = make_user(session, Role.ADMIN)
    for hostile in ("https://зло.example/", "//зло.example/", "http://зло.example"):
        response = client.post(
            "/login",
            data={"login": user.login, "password": PASSWORD, "next": hostile},
            follow_redirects=False,
        )
        assert response.headers["location"] == "/"


def test_forbidden_page_explains_itself(
    client: TestClient, session: Session
) -> None:
    sign_in(client, make_user(session, Role.ANALYST))
    response = client.get("/users", headers={"Accept": "text/html"})
    assert response.status_code == 403
    assert "Недостаточно прав" in response.text
    assert "Аналитик" in response.text


def test_favicon_does_not_litter_the_log(client: TestClient) -> None:
    assert client.get("/favicon.ico").status_code == 204


# ============================================================= демо-опрос


def test_demo_is_open_without_login(client: TestClient, demo_media: Path) -> None:
    """Демонстрацию должен открыть кто угодно: Заказчику надо услышать
    голос и формулировки, а не читать про них."""
    response = client.get("/demo")
    assert response.status_code == 200
    assert "Как звучит опрос" in response.text


def test_demo_offers_the_first_prompts_with_audio(client: TestClient, demo_media: Path) -> None:
    import re

    body = client.get("/demo").text
    keys = re.findall(r'data-audio="([^"]+)"', body)
    assert "intro_v1" in keys
    assert "consent_v1" in keys


def test_demo_audio_is_served(client: TestClient, demo_media: Path) -> None:
    response = client.get("/demo/audio/intro_v1")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_demo_audio_refuses_paths_outside_the_library(client: TestClient) -> None:
    """Путь, склеенный из пользовательского ввода, — это выход за пределы
    каталога, а рядом лежат другие файлы сервера."""
    for hostile in ("../../etc/passwd", "..%2f..%2fetc%2fpasswd", "a/b", "•"):
        assert client.get(f"/demo/audio/{hostile}").status_code == 404


def test_demo_walks_the_questionnaire(client: TestClient, demo_media: Path) -> None:
    import re

    body = client.get("/demo").text
    token = re.search(r'name="token" value="([^"]+)"', body).group(1)  # type: ignore[union-attr]

    after = client.post("/demo/answer", data={"token": token, "reply": "да"})
    assert after.status_code == 200
    assert "screen_age_v1" in after.text
    assert "Робот записал" in after.text


def test_demo_shows_what_the_robot_recorded(client: TestClient, demo_media: Path) -> None:
    """Показываем код, а не пересказ: это то самое, что попадёт в массив."""
    import re

    body = client.get("/demo").text
    token = re.search(r'name="token" value="([^"]+)"', body).group(1)  # type: ignore[union-attr]
    after = client.post("/demo/answer", data={"token": token, "reply": "да"})
    assert re.search(r"код \d+", after.text)


def test_demo_never_touches_the_database(
    client: TestClient, factory: sessionmaker[Session], demo_media: Path) -> None:
    """Ответы случайных посетителей не должны лежать рядом с массивом
    волны (ТЗ §5.3)."""
    import re

    from socmon.db.models import Interview

    body = client.get("/demo").text
    token = re.search(r'name="token" value="([^"]+)"', body).group(1)  # type: ignore[union-attr]
    with factory() as db:
        before = db.scalar(select(func.count()).select_from(Interview))

    for reply in ("да", "да", "да", "семь"):
        client.post("/demo/answer", data={"token": token, "reply": reply})

    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Interview)) == before


def test_unknown_demo_token_starts_over(client: TestClient) -> None:
    response = client.post(
        "/demo/answer", data={"token": "нет-такого", "reply": "да"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/demo"


def test_login_page_points_to_the_demo(client: TestClient) -> None:
    assert 'href="/demo"' in client.get("/login").text


# ============================================================== подписи


def test_status_codes_are_shown_in_russian(
    client: TestClient, session: Session
) -> None:
    """«draft» и «valid» понятны разработчику, а не методологу
    и не Заказчику."""
    prepare_wave(session, reference=None)
    sign_in(client, make_user(session, Role.CUSTOMER))

    body = client.get("/waves").text
    assert "черновик" in body
    assert ">draft<" not in body


def test_plural_forms_are_correct() -> None:
    from socmon.web.labels import plural

    cases = {1: "1 препятствие", 2: "2 препятствия", 5: "5 препятствий",
             11: "11 препятствий", 21: "21 препятствие", 22: "22 препятствия"}
    for count, expected in cases.items():
        assert plural(count, "препятствие", "препятствия", "препятствий") == expected


def test_unknown_status_falls_back_to_the_code() -> None:
    """Лучше показать код, чем пустоту: по коду понятно, что искать."""
    from socmon.web.labels import ru

    assert ru("неведомое_состояние") == "неведомое_состояние"
    assert ru(None) == "—"


def test_every_state_enum_has_a_russian_label() -> None:
    """Новое состояние без подписи вылезет на экран английским кодом."""
    from socmon.core.codes import CallStage, Disposition, Outcome
    from socmon.domain.validity import Validity
    from socmon.domain.wave import WaveStatus
    from socmon.web.labels import ru

    for enum in (CallStage, Disposition, Outcome, Validity, WaveStatus):
        for member in enum:
            assert ru(member) != member.value, f"нет подписи для {enum.__name__}.{member.value}"


def _forms(html: str) -> list[str]:
    return re.findall(r"<form\b.*?</form>", html, flags=re.S)


def test_demo_never_puts_two_reply_controls_in_one_form(
    client: TestClient, demo_media: Path,
) -> None:
    """Варианты ответа и поле свободного ввода — разные формы.

    В одной форме оба поля называются reply, и браузер отправляет их
    вместе: значение нажатой кнопки и пустую строку ввода. Пустая идёт
    последней и перебивает кнопку — робот получает молчание, переспрашивает,
    а со второго раза уходит на прощание. Нажимаешь «Согласен» и оказываешься
    на отказе, ничего не нажав неправильно.

    Проверка структурная, а не поведенческая: тест, отправляющий одно поле
    reply, эту ошибку не увидит вовсе — её делает браузер, а не сервер.
    """
    forms = _forms(client.get("/demo").text)
    assert forms, "на странице демонстрации нет ни одной формы"
    for form in forms:
        # Несколько кнопок с одним именем — норма: браузер отправляет
        # только нажатую. Смертельно другое — кнопка и поле ввода вместе:
        # тогда уходят оба значения.
        has_button = bool(re.search(r'<button[^>]*name="reply"', form))
        has_input = bool(re.search(r'<input[^>]*name="reply"', form))
        assert not (has_button and has_input), form


def test_demo_survives_the_browsers_own_payload(client: TestClient, demo_media: Path) -> None:
    """То, что реально уходит из браузера при нажатии кнопки варианта.

    Даже если разметка снова съедет, сервер обязан понять нажатие,
    а не принять пустое поле за молчание.
    """
    body = client.get("/demo").text
    token = re.search(r'name="token" value="([^"]+)"', body).group(1)  # type: ignore[union-attr]
    response = client.post(
        "/demo/answer",
        content=f"token={token}&reply=%D0%B4%D0%B0&reply=",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    assert "Извините за беспокойство" not in response.text
    assert "код 1" in response.text


def test_demo_enter_in_the_text_field_cannot_press_an_option(
    client: TestClient, demo_media: Path,
) -> None:
    """Enter в текстовом поле нажимает первую кнопку своей формы.

    Пока поле лежало рядом с вариантами, напечатать «нет» и нажать Enter
    значило ответить «да» — противоположным ответом.
    """
    forms = _forms(client.get("/demo").text)
    typed = [f for f in forms if re.search(r'<input[^>]*name="reply"', f)]
    assert typed, "поле свободного ответа исчезло со страницы"
    for form in typed:
        assert not re.search(r'<button[^>]*name="reply"', form), (
            "рядом с полем свободного ответа снова стоит кнопка варианта: "
            "Enter в поле нажмёт её"
        )


def test_demo_counter_adds_up_on_a_clean_pass(client: TestClient, demo_media: Path) -> None:
    """Пройденный без ошибок опрос должен показывать полный счёт.

    Раньше числитель считал только закодированные ответы, а знаменатель —
    все вопросы анкеты, включая семь открытых, у которых кода нет
    по устройству. Сойтись эти числа не могли никогда: безупречный проход
    выглядел недобором, и это первое, о чём спрашивал каждый, кто проходил
    демонстрацию.
    """
    from socmon.core.questionnaire import Questionnaire
    from tests.test_monitoring_questionnaire import SCRIPT

    questionnaire = Questionnaire.from_yaml(
        "src/socmon/questionnaires/wave-base.yaml"
    )
    by_audio = {(n.audio_key or n.id): n.id for n in questionnaire.nodes}
    play = re.compile(r'data-audio="([A-Za-z0-9_]+)"')

    html = client.get("/demo").text
    for _ in range(80):
        token = re.search(r'name="token" value="([^"]+)"', html)
        if not token:
            break
        keys = play.findall(html)
        node = by_audio.get(keys[-1]) if keys else None
        assert node, "не опознан текущий вопрос"
        response = client.post(
            "/demo/answer", data={"token": token.group(1), "reply": SCRIPT[node]}
        )
        assert response.status_code == 200
        html = response.text

    asked = int(re.search(r"Задано вопросов: (\d+)", html).group(1))  # type: ignore[union-attr]
    answered = int(re.search(r"получено ответов: (\d+)", html).group(1))  # type: ignore[union-attr]
    assert asked == answered, f"задано {asked}, отвечено {answered}"
    assert "не разобрано" not in html
    assert "Робот понял всё" in html


def test_password_change_revokes_session_and_is_journalled(
    client: TestClient, session: Session, factory: sessionmaker[Session],
) -> None:
    from socmon.db.users import change_password

    user = make_user(session, Role.ADMIN)
    sign_in(client, user)
    assert client.get("/interviews").status_code == 200
    with factory() as other:
        change_password(other, user.login, "новый-пароль-достаточно-длинный")
        other.commit()
    assert client.get("/interviews").status_code == 401
    with factory() as other:
        row = other.scalar(select(AuditLog).where(AuditLog.action == "change_password"))
        assert row and row.diff == {"sessions_revoked": True}


def test_reenabling_account_does_not_restore_old_session(
    client: TestClient, session: Session, factory: sessionmaker[Session],
) -> None:
    user = make_user(session, Role.CUSTOMER)
    sign_in(client, user)
    admin = make_user(session, Role.ADMIN)
    with TestClient(create_app(factory, secret_key="тест" * 8),
                    base_url="https://testserver") as control:
        form = control.get("/login")
        token = re.search(r'name="csrf_token" value="([^"]+)"', form.text)
        assert token
        control.headers["x-csrf-token"] = token[1]
        sign_in(control, admin)
        for _ in range(2):
            assert control.post(f"/users/{user.id}/toggle",
                                follow_redirects=False).status_code == 303
    assert client.get("/interviews").status_code == 401


def test_login_attempts_are_limited(client: TestClient) -> None:
    results = [client.post("/login", data={"login": "unknown", "password": "x" * 12})
               for _ in range(11)]
    assert [r.status_code for r in results[:10]] == [401] * 10
    assert results[-1].status_code == 429


def test_logout_revokes_a_copied_session(
    client: TestClient, session: Session, factory: sessionmaker[Session],
) -> None:
    user = make_user(session, Role.ADMIN)
    sign_in(client, user)
    stolen = dict(client.cookies)
    assert client.post("/logout", follow_redirects=False).status_code == 303
    with TestClient(create_app(factory, secret_key="тест" * 8),
                    base_url="https://testserver", cookies=stolen) as replay:
        assert replay.get("/interviews").status_code == 401


@pytest.fixture
def catalog_media(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from socmon.core.questionnaire import Questionnaire
    from socmon.voice.render import render
    from socmon.web import app as web_app
    from tests.test_voice import FakeSynthesizer

    monkeypatch.setattr(web_app, 'MEDIA_DIR', tmp_path)
    q = Questionnaire.from_yaml(web_app.DEMO_QUESTIONNAIRE)
    for voice in ('baya', 'eugene'):
        directory = tmp_path / 'voices' / voice
        render(q, FakeSynthesizer(), directory, voice=voice)
        (directory / 'sample_consent_8000.wav').write_bytes(
            (directory / 'consent_v1.wav').read_bytes(),
        )
    return tmp_path


def test_voice_gallery_only_offers_rendered_variants(
    client: TestClient, catalog_media: Path,
) -> None:
    response = client.get('/voices')
    assert response.status_code == 200
    assert '/demo?voice=baya' in response.text
    assert '/demo?voice=eugene' in response.text
    assert '/demo?voice=aidar' not in response.text
    assert response.text.count('<audio ') == 2


def test_new_voice_pair_and_legacy_comparison(
    client: TestClient, catalog_media: Path,
) -> None:
    from socmon.core.questionnaire import Questionnaire
    from socmon.voice.render import render
    from socmon.web.app import DEMO_QUESTIONNAIRE
    from tests.test_voice import FakeSynthesizer

    for voice in ('anna', 'alex'):
        target = catalog_media / 'voices' / voice
        render(Questionnaire.from_yaml(DEMO_QUESTIONNAIRE), FakeSynthesizer(), target, voice=voice)
        (target / 'sample_consent_8000.wav').write_bytes((target / 'consent_v1.wav').read_bytes())
    response = client.get('/voices')
    assert response.text.count('<audio ') == 2
    assert '/demo?voice=anna' in response.text
    assert '/demo?voice=alex' in response.text
    assert client.get('/voices?legacy=true').text.count('<audio ') == 4
    for voice in ('anna', 'alex'):
        start = client.get('/demo', params={'voice': voice})
        token = re.search(r'name="token" value="([^"]+)"', start.text)
        assert token is not None
        after = client.post('/demo/answer', data={
            'token': token.group(1), 'reply': 'да', 'voice': voice,
        })
        assert 'восемнадцать' in after.text
        assert f'name="voice" value="{voice}"' in after.text


def test_selected_demo_voice_survives_a_form_submission(
    client: TestClient, catalog_media: Path,
) -> None:
    response = client.get('/demo?voice=eugene')
    token = re.search(r'name="token" value="([^"]+)"', response.text)
    assert token is not None
    after = client.post('/demo/answer', data={
        'token': token.group(1), 'reply': 'да', 'voice': 'eugene',
    })
    assert after.status_code == 200
    assert 'Евгений' in after.text
    assert 'name="voice" value="eugene"' in after.text
    assert 'восемнадцать' in after.text
    assert client.get('/demo/audio/intro_v1?voice=eugene').content == (
        catalog_media / 'voices/eugene/intro_v1.wav'
    ).read_bytes()


def test_demo_tokens_cannot_be_moved_to_another_voice(
    client: TestClient, catalog_media: Path,
) -> None:
    response = client.get('/demo?voice=eugene')
    token = re.search(r'name="token" value="([^"]+)"', response.text)
    assert token is not None
    after = client.post('/demo/answer', data={
        'token': token.group(1), 'reply': 'да', 'voice': 'baya',
    }, follow_redirects=False)
    assert after.status_code == 303
    assert after.headers['location'] == '/demo?voice=baya'


@pytest.mark.parametrize('url', [
    '/demo?voice=../other', '/voices?sample=../intro', '/voices?rate=44100',
    '/demo/audio/intro_v1?voice=unknown', '/voices/unknown/sample/consent/8000',
    '/voices/baya/sample/unknown/8000',
])
def test_voice_routes_refuse_unknown_paths(
    client: TestClient, catalog_media: Path, url: str,
) -> None:
    assert client.get(url).status_code == 404


def test_voice_samples_cannot_follow_a_symlink_out_of_the_library(
    client: TestClient, catalog_media: Path, tmp_path: Path,
) -> None:
    secret = tmp_path / 'private.wav'
    secret.write_bytes(b'private')
    target = catalog_media / 'voices/baya/sample_consent_8000.wav'
    target.unlink()
    target.symlink_to(secret)
    assert client.get('/voices/baya/sample/consent/8000').status_code == 404


@pytest.mark.parametrize('sample', ['intro', 'consent', 'q07_material', 'end_no_consent'])
def test_gallery_examples_use_the_current_demo_questionnaire(
    client: TestClient, catalog_media: Path, sample: str,
) -> None:
    from html import unescape

    from socmon.core.questionnaire import Questionnaire
    from socmon.web.app import DEMO_QUESTIONNAIRE

    response = client.get('/voices', params={'sample': sample})
    assert response.status_code == 200
    expected = Questionnaire.from_yaml(DEMO_QUESTIONNAIRE).node(sample).text
    assert expected in unescape(response.text)
