"""Кабинет: маршруты и проверка прав.

Право проверяется на каждом маршруте, а не в шаблоне. Спрятать ссылку
в меню — не защита: адрес можно набрать руками, и в системе с телефонными
номерами и записями разговоров это не теоретическое рассуждение.

Второй сквозной принцип — журнал. ТЗ §14.2 требует журнала доступа
и операций, и записывается не только изменение, но и **просмотр
чувствительного**: кто и когда слушал запись разговора.
"""

# Без `from __future__ import annotations`: он превращает аннотации в строки,
# а FastAPI разрешает их в глобальном пространстве модуля. Псевдонимы
# зависимостей объявлены внутри фабрики приложения — снаружи их не видно,
# и каждый маршрут падает с «Field required: db».
import os
import re
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

import yaml
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from starlette.middleware.sessions import SessionMiddleware

from socmon.analysis.index import IndexError_
from socmon.analysis.wave import WaveSummary, load_valid_records, summarize
from socmon.analysis.weighting import Margin, WeightingError
from socmon.core.codes import CallStage, Disposition, Outcome
from socmon.core.questionnaire import Questionnaire
from socmon.db.coding_store import (
    mark_reviewed,
    pending_review,
    review_progress,
    theme_counts,
)
from socmon.db.contacts import base_summary, import_numbers, read_column
from socmon.db.models import (
    AuditLog,
    CallAttempt,
    Contact,
    ContactBatch,
    Interview,
    PhoneLine,
    QuestionnaireVersion,
    User,
    Wave,
)
from socmon.domain.identity import PhoneVault
from socmon.domain.validity import Validity
from socmon.domain.wave import (
    Readiness,
    WaveError,
    WaveSetup,
    WaveStatus,
    can_transition,
    readiness,
    transition,
)
from socmon.voice.catalog import (
    CLONED,
    SAMPLES,
    VOICE_BY_ID,
    VOICE_OPTIONS,
    safe_audio,
    voice_directory,
)
from socmon.voice.delivery import VOICE_GENDER, Catalogue
from socmon.voice.render import MANIFEST_NAME, VoiceManifest, verify
from socmon.web.artifacts import ArtifactError, build_archive, existing_archive
from socmon.web.demo import DemoStudio
from socmon.web.labels import plural, ru
from socmon.web.protection import (
    LoginLimiter,
    RequestProtection,
    csrf_token,
    local_redirect,
    verify_csrf,
)
from socmon.web.security import (
    ROLE_PERMISSIONS,
    ROLE_TITLES,
    Permission,
    Principal,
    Role,
    hash_password,
    mask_phone,
    verify_password,
)

TEMPLATES = Path(__file__).parent / "templates"
MEDIA_DIR = Path(os.environ.get("SOCMON_MEDIA_DIR", "media"))
"""Каталог озвучки.

Берётся из окружения: относительный путь зависит от того, откуда
запущена служба, и однажды не найдётся — ровно это и случилось
на стенде, где файлы лежали в одном каталоге, а манифест в другом.
"""
#: Служебные реплики подачи — общие для всех голосов; форма рода
#: подбирается по голосу.
DELIVERY_CATALOGUE = Catalogue.from_yaml()

#: Основная библиотека собрана этим голосом; от него зависит род служебных
#: реплик, когда голос не выбран явно.
DEFAULT_DEMO_VOICE = "anna"


def _library(directory: Path) -> VoiceManifest | None:
    """Манифест библиотеки, если он есть. Без него подача играет только
    вопросы анкеты — откликов, которых не записано, она не просит."""
    path = directory / MANIFEST_NAME
    try:
        return VoiceManifest.load(path) if path.exists() else None
    except (OSError, ValueError, KeyError):
        return None


DEMO_QUESTIONNAIRE = (
    Path(__file__).resolve().parent.parent / "questionnaires" / "wave-base.yaml"
)
"""Эталонная анкета для демонстрации.

Ищется от расположения пакета, а не от рабочего каталога: служба
запускается откуда угодно, и относительный путь однажды не найдётся.
"""

MAX_CONTACTS_BYTES = 8 * 1024 * 1024
"""Потолок на загрузку базы через кабинет. Миллион номеров — это около
15 МБ, и такие файлы грузятся командой, а не через форму."""

MAX_QUESTIONNAIRE_BYTES = 512 * 1024
"""Потолок на загружаемую анкету. Реальная — около 20 КБ; всё сильно
большее означает не анкету, а недоразумение."""
"""Каталог озвучки. Проверяется на полноту перед запуском волны."""


def audit(
    db: Session,
    actor: Principal,
    action: str,
    entity_type: str,
    entity_id: str,
    diff: dict[str, Any] | None = None,
) -> None:
    """Записать действие в журнал.

    Журнал только на вставку: правки и удаления через приложение нет.
    Иначе он не свидетельство, а черновик.
    """
    db.add(AuditLog(
        actor=f"{actor.login} ({actor.role.value})",
        action=action, entity_type=entity_type, entity_id=entity_id, diff=diff,
    ))
    db.flush()


def create_app(
    session_factory: sessionmaker[Session], *, secret_key: str, https_only: bool = True,
) -> FastAPI:
    if len(secret_key) < 32:
        raise ValueError("SOCMON_SECRET_KEY должен содержать не меньше 32 символов")
    app = FastAPI(
        title="Соцмониторинг — кабинет", docs_url=None, redoc_url=None,
        openapi_url=None, dependencies=[Depends(verify_csrf)],
    )
    app.add_middleware(RequestProtection)
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret_key,
        https_only=https_only,
        same_site="lax",
        max_age=8 * 3600,
    )
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.globals["csrf_token"] = csrf_token
    login_limiter = LoginLimiter()
    account_limiter = LoginLimiter(limit=30)
    dummy_password_hash = hash_password(secret_key)
    templates.env.globals["ROLE_TITLES"] = ROLE_TITLES
    templates.env.globals["Permission"] = Permission
    templates.env.globals["ROLE_PERMISSIONS"] = ROLE_PERMISSIONS
    templates.env.filters["ru"] = ru
    templates.env.globals["plural"] = plural

    def wants_html(request: Request) -> bool:
        """Пришёл человек браузером или программа за данными."""
        return "text/html" in request.headers.get("accept", "")

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, exc: HTTPException) -> Any:
        """Показать человеку страницу, а не JSON.

        Без этого заход на кабинет без входа отдаёт `{"detail": "..."}`
        прямо в окно браузера. Формально верно, практически — человек
        видит непонятный текст и решает, что ничего не работает.
        """
        if not wants_html(request):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            target = request.url.path
            suffix = (
                "?" + urlencode({"next": target})
                if target not in ("/", "/login") else ""
            )
            return RedirectResponse(
                f"/login{suffix}", status_code=status.HTTP_303_SEE_OTHER
            )

        return templates.TemplateResponse(
            request=request, name="error.html",
            context={"code": exc.status_code, "detail": exc.detail, "me": None},
            status_code=exc.status_code,
        )

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Any:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def get_db() -> Iterator[Session]:
        with session_factory() as session:
            yield session
            session.commit()

    Db = Annotated[Session, Depends(get_db)]

    def current_user(request: Request, db: Db) -> Principal:
        user_id = request.session.get("user_id")
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "не выполнен вход")
        try:
            parsed_id = uuid.UUID(str(user_id))
        except ValueError:
            request.session.clear()
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "не выполнен вход") from None
        row = db.get(User, parsed_id)
        if (row is None or not row.is_active
                or request.session.get("session_version") != row.session_version):
            # Выключенная учётная запись перестаёт действовать немедленно,
            # а не после истечения сессии.
            request.session.clear()
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "учётная запись выключена")
        return Principal(
            user_id=str(row.id), login=row.login,
            full_name=row.full_name, role=Role(row.role),
        )

    Me = Annotated[Principal, Depends(current_user)]

    def maybe_user(request: Request, db: Db) -> Principal | None:
        """Кто зашёл, если вообще заходил.

        Отдельно от `current_user`: на титульной странице отсутствие входа
        не ошибка, а обычное состояние посетителя.
        """
        try:
            return current_user(request, db)
        except HTTPException:
            return None

    Maybe = Annotated[Principal | None, Depends(maybe_user)]

    def require(permission: Permission) -> Callable[[Principal], Principal]:
        def guard(me: Me) -> Principal:
            if not me.can(permission):
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN,
                    f"роли «{ROLE_TITLES[me.role]}» это недоступно",
                )
            return me
        return guard

    def page(request: Request, name: str, me: Principal, **context: Any) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request, name=name, context={"me": me, **context}
        )

    # ------------------------------------------------------- демонстрация

    studio: dict[str, DemoStudio] = {}

    def selected_directory(voice: str) -> Path:
        try:
            return voice_directory(MEDIA_DIR, voice)
        except ValueError:
            raise HTTPException(404, "нет такого голоса") from None

    @app.get("/voices", response_class=HTMLResponse)
    def voice_gallery(
        request: Request, me: Maybe, sample: str = "consent", rate: int = 8000,
        legacy: bool = False,
    ) -> HTMLResponse:
        if sample not in SAMPLES or rate not in (8000, 24000):
            raise HTTPException(404, "нет такого примера")
        questionnaire = Questionnaire.from_yaml(DEMO_QUESTIONNAIRE)
        available = [v for v in VOICE_OPTIONS if safe_audio(
            selected_directory(v.id), f"sample_{sample}_{rate}.wav",
        )]
        ready = {v.id for v in available if verify(
            questionnaire, selected_directory(v.id),
        ).ready}
        has_natural = bool(CLONED & ready)
        if has_natural and not legacy:
            available = [v for v in available if v.id in CLONED]
        return templates.TemplateResponse(
            request=request, name="voices.html",
            context={"me": me, "voices": available, "ready": ready, "samples": SAMPLES,
                     "has_natural": has_natural, "legacy": legacy,
                     "sample": sample, "rate": rate, "text": questionnaire.node(sample).text},
        )

    @app.get("/voices/{voice}/sample/{sample}/{rate}")
    def voice_sample(voice: str, sample: str, rate: int) -> Any:
        if voice not in VOICE_BY_ID or sample not in SAMPLES or rate not in (8000, 24000):
            raise HTTPException(404, "нет такого примера")
        path = safe_audio(selected_directory(voice), f"sample_{sample}_{rate}.wav")
        if path is None:
            raise HTTPException(404, "пример ещё не готов")
        return FileResponse(path, media_type="audio/wav")

    def demo_studio(voice: str = "") -> DemoStudio | None:
        """Демонстрация поднимается на эталонной анкете, если она озвучена."""
        directory = selected_directory(voice)
        if voice in studio:
            return studio[voice]
        try:
            questionnaire = Questionnaire.from_yaml(DEMO_QUESTIONNAIRE)
        except (OSError, ValueError):
            return None
        if not verify(questionnaire, directory).ready:
            return None
        studio[voice] = DemoStudio(
            questionnaire,
            catalogue=DELIVERY_CATALOGUE,
            gender=VOICE_GENDER.get(voice or DEFAULT_DEMO_VOICE, "f"),
            library=_library(directory),
            directory=directory,
        )
        return studio[voice]

    @app.get("/demo", response_class=HTMLResponse)
    def demo_start(request: Request, me: Maybe, voice: str = "") -> HTMLResponse:
        """Публичная демонстрация опроса. Вход не нужен."""
        shop = demo_studio(voice)
        if shop is None:
            return templates.TemplateResponse(
                request=request, name="demo.html",
                context={"me": me, "unavailable": True},
            )
        token, step = shop.start()
        return templates.TemplateResponse(
            request=request, name="demo.html",
            context={"me": me, "token": token, "step": step,
                     "questionnaire": shop.questionnaire, "voice": voice,
                     "voice_name": VOICE_BY_ID[voice].name if voice else "Текущий голос"},
        )

    @app.post("/demo/answer", response_class=HTMLResponse)
    async def demo_answer(request: Request, me: Maybe) -> HTMLResponse:
        # Форму разбираем сами, а не параметрами маршрута. Браузер способен
        # прислать несколько полей reply разом — значение нажатой кнопки
        # и пустую строку ввода рядом с ней. Разбор по имени берёт последнее,
        # то есть пустое, и нажатие кнопки превращается в молчание.
        # Разметка сейчас такого не допускает, но зависеть от разметки
        # в этом месте не стоит: цена ошибки — ответ, противоположный
        # тому, что человек выбрал.
        form = await request.form()
        token = str(form.get("token") or "")
        reply = next(
            (str(value) for value in form.getlist("reply") if str(value).strip()),
            "",
        )
        voice = str(form.get("voice") or "")
        shop = demo_studio(voice)
        step = shop.answer(token, reply) if shop else None
        if shop is None or step is None:
            return RedirectResponse(  # type: ignore[return-value]
                "/demo" + ("?" + urlencode({"voice": voice}) if voice else ""),
                status_code=status.HTTP_303_SEE_OTHER
            )
        return templates.TemplateResponse(
            request=request, name="demo.html",
            context={"me": me, "token": token, "step": step,
                     "questionnaire": shop.questionnaire, "voice": voice,
                     "voice_name": VOICE_BY_ID[voice].name if voice else "Текущий голос"},
        )

    @app.get("/demo/audio/{key}")
    def demo_audio(key: str, voice: str = "") -> Any:
        """Отдать записанную реплику.

        Имя файла собирается из белого списка символов: путь, склеенный
        из пользовательского ввода, — это выход за пределы каталога,
        а рядом лежат другие файлы сервера.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "нет такой реплики")
        path = safe_audio(selected_directory(voice), f"{key}.wav")
        if path is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "нет такой реплики")
        return FileResponse(path, media_type="audio/wav")

    # ------------------------------------------------------------------ вход

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, next: str = "/") -> HTMLResponse:
        return templates.TemplateResponse(
            request=request, name="login.html", context={"next": next}
        )

    @app.post("/login")
    def login(
        request: Request, db: Db,
        login: Annotated[str, Form(max_length=100)],
        password: Annotated[str, Form(max_length=1024)],
        next: Annotated[str, Form()] = "/",
    ) -> Any:
        address = request.client.host if request.client else "unknown"
        if not login_limiter.allow(address) or not account_limiter.allow(login):
            raise HTTPException(429, "Слишком много попыток входа. Повторите через 5 минут.")
        row = db.scalar(select(User).where(User.login == login, User.is_active.is_(True)))
        verified = verify_password(password, row.password_hash if row else dummy_password_hash)
        if row is None or not verified:
            # Одна и та же формулировка на неверный логин и неверный пароль:
            # иначе форма превращается в способ узнать, кто заведён в системе.
            return templates.TemplateResponse(
                request=request, name="login.html",
                context={"error": "Неверный логин или пароль"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        row.last_login_at = datetime.now(UTC)
        form_token = csrf_token(request)
        request.session.clear()
        request.session["csrf_token"] = form_token
        request.session["user_id"] = str(row.id)
        request.session["session_version"] = row.session_version
        db.add(AuditLog(
            actor=f"{row.login} ({row.role})", action="login",
            entity_type="user", entity_id=str(row.id), diff=None,
        ))
        # Возвращаем туда, куда человек шёл, но только внутрь кабинета:
        # открытый редирект по чужому адресу — это фишинг с нашего адреса.
        target = local_redirect(next)
        return RedirectResponse(target, status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/logout")
    def logout(request: Request, db: Db) -> Any:
        user_id = request.session.get("user_id")
        if user_id:
            try:
                parsed_id = uuid.UUID(str(user_id))
            except ValueError:
                parsed_id = None
            if parsed_id is not None:
                row = db.get(User, parsed_id, with_for_update=True)
                if (row is not None
                        and request.session.get("session_version") == row.session_version):
                    row.session_version += 1
        request.session.clear()
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    # ------------------------------------------------------------------ обзор

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, db: Db, me: Maybe) -> HTMLResponse:
        """Единая точка входа.

        Гостю — что это такое и два пути: послушать опрос или войти.
        Вошедшему — сводка по волне. Одна ссылка на всё, чтобы её можно
        было просто дать человеку, не объясняя, куда идти дальше.
        """
        if me is None:
            shop_ready = demo_studio() is not None
            return templates.TemplateResponse(
                request=request, name="welcome.html",
                context={"me": None, "demo_ready": shop_ready},
            )
        if not me.can(Permission.VIEW_DASHBOARD):
            return RedirectResponse(  # type: ignore[return-value]
                "/demo", status_code=status.HTTP_303_SEE_OTHER
            )
        waves = db.scalars(select(Wave).order_by(Wave.starts_at.desc())).all()
        current = waves[0] if waves else None
        stats = _wave_stats(db, current.id) if current else {}
        return page(request, "dashboard.html", me, waves=waves, wave=current, stats=stats)

    # -------------------------------------------------------------- интервью

    @app.get("/interviews", response_class=HTMLResponse)
    def interviews(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.VIEW_INTERVIEWS))],
        validity: str = "", outcome: str = "", limit: int = 100,
    ) -> HTMLResponse:
        query = select(Interview).order_by(Interview.started_at.desc())
        if validity:
            query = query.where(Interview.validity == Validity(validity))
        if outcome:
            query = query.where(Interview.outcome == Outcome(outcome))
        rows = db.scalars(query.limit(max(1, min(limit, 500)))).all()
        return page(
            request, "interviews.html", me, interviews=rows,
            validity=validity, outcome=outcome,
            validities=list(Validity), outcomes=list(Outcome),
        )

    @app.get("/interviews/{interview_id}", response_class=HTMLResponse)
    def interview_card(
        request: Request, db: Db, interview_id: uuid.UUID,
        me: Annotated[Principal, Depends(require(Permission.VIEW_INTERVIEWS))],
    ) -> HTMLResponse:
        row = db.get(Interview, interview_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "интервью не найдено")
        return page(request, "interview.html", me, interview=row)

    @app.get("/interviews/{interview_id}/recording")
    def recording(
        request: Request, db: Db, interview_id: uuid.UUID,
        me: Annotated[Principal, Depends(require(Permission.LISTEN_RECORDING))],
    ) -> Any:
        """Выдать ссылку на запись.

        ТЗ §13.4 разрешает просмотр записи «при наличии соответствующего
        уровня доступа», §14.2 требует журнала доступа. Поэтому каждое
        обращение — строка в журнале, и обезличенный ID в ней есть,
        а номера нет.
        """
        row = db.get(Interview, interview_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "интервью не найдено")
        audit(db, me, "listen_recording", "interview", str(interview_id))
        respondent_uri = row.recording_respondent_uri or row.recording_uri
        if not respondent_uri:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "запись отсутствует")
        return {"respondent": respondent_uri, "robot": row.recording_robot_uri}

    # ---------------------------------------------------------------- обзвон

    @app.get("/dialling", response_class=HTMLResponse)
    def dialling(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_DIALLING))],
        limit: int = 100,
    ) -> HTMLResponse:
        rows = db.scalars(
            select(CallAttempt).order_by(CallAttempt.scheduled_at.desc())
            .limit(max(1, min(limit, 500)))
        ).all()
        return page(request, "dialling.html", me, attempts=rows, mask=mask_phone)

    # --------------------------------------------------------------- волны

    @app.get("/waves", response_class=HTMLResponse)
    def waves_page(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_WAVE))],
    ) -> HTMLResponse:
        rows = db.scalars(select(Wave).order_by(Wave.starts_at.desc())).all()
        checks = {w.id: _readiness_of(db, w) for w in rows}
        return page(
            request, "waves.html", me, waves=rows, checks=checks,
            can_transition=can_transition, WaveStatus=WaveStatus,
        )

    @app.post("/waves")
    def create_wave(
        db: Db,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_WAVE))],
        code: Annotated[str, Form(pattern=r"^[A-Za-z0-9_-]{1,32}$")],
        name: Annotated[str, Form(min_length=1, max_length=200)],
        target_min: Annotated[int, Form()] = 1500,
        target_max: Annotated[int, Form()] = 2000,
    ) -> Any:
        """Завести волну.

        Двухнедельная по умолчанию: §3.1 называет волну единицей отчётности
        и задаёт её длину, а не оставляет на усмотрение.
        """
        if db.scalar(select(Wave).where(Wave.code == code)):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "волна с таким кодом есть")
        starts = datetime.now(UTC)
        row = Wave(
            code=code, name=name, starts_at=starts,
            ends_at=starts + timedelta(days=14),
            target_min=target_min, target_max=target_max,
            status=WaveStatus.DRAFT,
        )
        db.add(row)
        db.flush()
        audit(db, me, "create_wave", "wave", str(row.id), {"code": code})
        return RedirectResponse("/waves", status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/waves/{wave_id}/status")
    def change_wave_status(
        db: Db, wave_id: uuid.UUID,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_WAVE))],
        target: Annotated[str, Form()],
    ) -> Any:
        """Сменить состояние волны.

        Запуск возможен только при полной готовности: запустив волну
        с устаревшей озвучкой, мы обзвоним тысячу человек не тем вопросом.
        """
        row = db.get(Wave, wave_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "волна не найдена")
        try:
            wanted = WaveStatus(target)
        except ValueError:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"неизвестное состояние «{target}»"
            ) from None

        if wanted is WaveStatus.ACTIVE and row.status is WaveStatus.DRAFT:
            check = _readiness_of(db, row)
            if not check.ready:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "волна не готова: " + "; ".join(c.name for c in check.blockers),
                )
        try:
            was, row.status = row.status, transition(row.status, wanted)
        except WaveError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None

        if row.status is WaveStatus.CLOSED:
            row.closed_at = datetime.now(UTC)
        audit(db, me, "wave_status", "wave", str(wave_id),
              {"status": {"before": was.value, "after": row.status.value}})
        return RedirectResponse("/waves", status_code=status.HTTP_303_SEE_OTHER)

    # ---------------------------------------------------------- живое табло

    @app.get("/board", response_class=HTMLResponse)
    def board(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.VIEW_DASHBOARD))],
        wave: str = "",
    ) -> HTMLResponse:
        """Что происходит на линиях прямо сейчас (ТЗ §13.4).

        Страница обновляется сама метатегом, без единой строки JavaScript:
        на защищённой машине рядом с записями разговоров это одной
        зависимостью меньше.
        """
        waves = db.scalars(select(Wave).order_by(Wave.starts_at.desc())).all()
        chosen = _pick_wave(waves, wave)
        if chosen is None:
            return page(request, "board.html", me, waves=waves, wave=None)

        stages: dict[CallStage, int] = {
            stage: count
            for stage, count in db.execute(
                select(CallAttempt.stage, func.count())
                .where(CallAttempt.wave_id == chosen.id)
                .group_by(CallAttempt.stage)
            ).all()
        }
        dispositions: dict[Disposition, int] = {
            disposition: count
            for disposition, count in db.execute(
                select(CallAttempt.disposition, func.count())
                .where(
                    CallAttempt.wave_id == chosen.id,
                    CallAttempt.disposition.isnot(None),
                )
                .group_by(CallAttempt.disposition)
            ).all()
            if disposition is not None
        }
        return page(
            request, "board.html", me, waves=waves, wave=chosen,
            stages=stages, dispositions=dispositions,
            live=[s for s in CallStage if s is not CallStage.FINISHED],
            stats=_wave_stats(db, chosen.id),
        )

    # ---------------------------------------------------------------- база

    @app.get("/contacts", response_class=HTMLResponse)
    def contacts(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_CONTACTS))],
        error: str = "", loaded: str = "",
    ) -> HTMLResponse:
        batches = db.scalars(
            select(ContactBatch).order_by(ContactBatch.created_at.desc()).limit(50)
        ).all()
        return page(
            request, "contacts.html", me,
            summary=base_summary(db), batches=batches,
            vault_ready=_vault_ready(), error=error, loaded=loaded,
            max_upload_mb=MAX_CONTACTS_BYTES // (1024 * 1024),
        )

    @app.post("/contacts/upload")
    async def upload_contacts(
        db: Db,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_CONTACTS))],
        file: Annotated[UploadFile, File()],
        name: Annotated[str, Form()] = "",
    ) -> Any:
        """Загрузить базу из файла (ТЗ §4.2).

        Через кабинет — файлы разумного размера. Миллион номеров грузится
        командой `socmon contacts import`: держать такой файл в памяти
        веб-процесса незачем, а загрузка занимает минуты, и обрывать её
        тайм-аутом запроса было бы обидно.
        """
        def back(**params: str) -> Any:
            return RedirectResponse(
                "/contacts?" + urlencode(params),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        try:
            vault = PhoneVault.from_env()
        except RuntimeError as exc:
            return back(error=str(exc))

        raw = await file.read(MAX_CONTACTS_BYTES + 1)
        if len(raw) > MAX_CONTACTS_BYTES:
            return back(error=(
                f"файл больше {MAX_CONTACTS_BYTES // (1024 * 1024)} МБ — "
                f"загрузите командой socmon contacts import"
            ))
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return back(error="файл не в кодировке UTF-8")

        report = import_numbers(
            db, vault, read_column(text),
            name=name or (file.filename or "загрузка"),
            source=f"кабинет, {me.login}",
        )
        audit(db, me, "import_contacts", "contact_batch", str(report.batch_id), {
            "total": report.total, "loaded": report.loaded,
            "rejected": report.rejected, "seed": report.seed,
        })
        return back(loaded=str(report.batch_id))

    # -------------------------------------------------------------- анкеты

    @app.get("/questionnaires", response_class=HTMLResponse)
    def questionnaires(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_QUESTIONNAIRE))],
        wave: str = "", error: str = "",
    ) -> HTMLResponse:
        waves = db.scalars(select(Wave).order_by(Wave.starts_at.desc())).all()
        chosen = _pick_wave(waves, wave)
        versions = []
        if chosen is not None:
            versions = list(db.scalars(
                select(QuestionnaireVersion)
                .where(QuestionnaireVersion.wave_id == chosen.id)
                .order_by(QuestionnaireVersion.created_at.desc())
            ).all())
        return page(
            request, "questionnaires.html", me, waves=waves, wave=chosen,
            versions=versions,
            details={v.id: _questionnaire_report(v) for v in versions},
            error=error,
        )

    @app.post("/questionnaires/{wave_id}/upload")
    async def upload_questionnaire(
        db: Db, wave_id: uuid.UUID,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_QUESTIONNAIRE))],
        file: Annotated[UploadFile, File()],
    ) -> Any:
        """Привязать анкету к волне.

        Только к черновику: подменить анкету посреди волны значит получить
        половину интервью по одной редакции вопросов и половину по другой,
        и никакая версия этого уже не разъяснит (ТЗ §3.1, §7.3).
        """
        row = db.get(Wave, wave_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "волна не найдена")

        def back(message: str) -> Any:
            return RedirectResponse(
                "/questionnaires?" + urlencode({"wave": row.code, "error": message}),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        if row.status is not WaveStatus.DRAFT:
            return back("анкету можно привязать только к черновику волны")

        raw = await file.read(MAX_QUESTIONNAIRE_BYTES + 1)
        if len(raw) > MAX_QUESTIONNAIRE_BYTES:
            return back("файл больше допустимого размера")
        try:
            parsed = Questionnaire.model_validate(yaml.safe_load(raw.decode("utf-8")))
        except Exception as exc:
            return back(f"анкета не проходит проверку: {str(exc)[:200]}")

        if db.scalar(
            select(QuestionnaireVersion).where(
                QuestionnaireVersion.wave_id == wave_id,
                QuestionnaireVersion.version_id == parsed.version_id,
            )
        ):
            return back("такая версия анкеты уже привязана к волне")

        version = QuestionnaireVersion(
            wave_id=wave_id, name=parsed.name, version_id=parsed.version_id,
            source_yaml=raw.decode("utf-8"), core_keys=list(parsed.core_keys),
        )
        db.add(version)
        db.flush()
        audit(db, me, "attach_questionnaire", "questionnaire_version",
              str(version.id), {"version_id": parsed.version_id, "wave": row.code})
        return RedirectResponse(
            "/questionnaires?" + urlencode({"wave": row.code}),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/questionnaires/{version_id}/approve")
    def approve_questionnaire(
        db: Db, version_id: uuid.UUID,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_QUESTIONNAIRE))],
    ) -> Any:
        """Утвердить версию анкеты (ТЗ §7.2, §7.3).

        Утверждение привязано к идентификатору версии: правка любой
        формулировки даёт новый идентификатор, и утверждение вместе с ним
        обнуляется. Иначе «утверждённая анкета» означала бы что угодно.
        """
        row = db.get(QuestionnaireVersion, version_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "версия не найдена")
        if row.approved_at is not None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "версия уже утверждена")
        row.approved_at = datetime.now(UTC)
        row.approved_by = me.full_name
        audit(db, me, "approve_questionnaire", "questionnaire_version",
              str(version_id), {"version_id": row.version_id})
        return RedirectResponse(
            "/questionnaires", status_code=status.HTTP_303_SEE_OTHER
        )

    # ------------------------------------------------------------ аналитика

    @app.get("/analytics", response_class=HTMLResponse)
    def analytics(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.VIEW_ANALYTICS))],
        wave: str = "",
    ) -> HTMLResponse:
        """Взвешивание и индекс по волне.

        Считается на лету, а не берётся из кеша: §12 требует
        воспроизводимости расчёта, и цифра на экране должна получаться
        из тех же данных и той же методики, что и цифра в выгрузке.
        """
        waves = db.scalars(select(Wave).order_by(Wave.starts_at.desc())).all()
        chosen = _pick_wave(waves, wave)
        if chosen is None:
            return page(request, "analytics.html", me, waves=waves, wave=None)

        summary, problem = _summarize_wave(db, chosen)
        return page(
            request, "analytics.html", me,
            waves=waves, wave=chosen, summary=summary, problem=problem,
        )

    # -------------------------------------------------------------- разбор

    @app.get("/coding", response_class=HTMLResponse)
    def coding(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.REVIEW_CODING))],
        wave: str = "",
    ) -> HTMLResponse:
        waves = db.scalars(select(Wave).order_by(Wave.starts_at.desc())).all()
        chosen = _pick_wave(waves, wave)
        if chosen is None:
            return page(request, "coding.html", me, waves=waves, wave=None)

        done, total = review_progress(db, chosen.id)
        return page(
            request, "coding.html", me, waves=waves, wave=chosen,
            queue=pending_review(db, chosen.id, limit=50),
            themes=theme_counts(db, chosen.id),
            reviewed=done, ambiguous_total=total,
        )

    @app.post("/coding/{coding_id}/reviewed")
    def confirm_coding(
        db: Db, coding_id: uuid.UUID,
        me: Annotated[Principal, Depends(require(Permission.REVIEW_CODING))],
    ) -> Any:
        """Отметить разметку проверенной.

        Сам вывод остаётся модельным: §11.3 требует, чтобы автоматически
        сгенерированное было отличимо от проверенного, а не подменялось им.
        """
        row = mark_reviewed(db, coding_id, me.login)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "разметка не найдена")
        audit(db, me, "review_coding", "open_answer_coding", str(coding_id))
        return RedirectResponse("/coding", status_code=status.HTTP_303_SEE_OTHER)

    # ------------------------------------------------------------- выгрузки

    @app.get("/exports", response_class=HTMLResponse)
    def exports(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.EXPORT_DATA))],
        wave: str = "", error: str = "",
    ) -> HTMLResponse:
        waves = db.scalars(select(Wave).order_by(Wave.starts_at.desc())).all()
        chosen = _pick_wave(waves, wave)
        return page(
            request, "exports.html", me, waves=waves, wave=chosen,
            stats=_wave_stats(db, chosen.id) if chosen else {},
            artifact=existing_archive(chosen.code) if chosen else None,
            error=error,
        )

    @app.post("/exports/{wave_id}")
    def build_export(
        db: Db, wave_id: uuid.UUID,
        me: Annotated[Principal, Depends(require(Permission.EXPORT_DATA))],
    ) -> Any:
        """Собрать оперативный пакет (ТЗ §13.2).

        Сборка синхронная: на волне в две тысячи интервью она занимает
        секунды. Когда объёмы вырастут, её место — в очереди задач,
        а не в запросе.
        """
        row = db.get(Wave, wave_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "волна не найдена")
        try:
            artifact = build_archive(db, row)
        except ArtifactError as exc:
            # Не пятисотка: причина понятна человеку и он может её устранить.
            return RedirectResponse(
                "/exports?" + urlencode({"wave": row.code, "error": str(exc)}),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        audit(db, me, "export_package", "wave", str(wave_id),
              {"code": row.code, "files": len(artifact.files)})
        return RedirectResponse(
            "/exports?" + urlencode({"wave": row.code}),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/exports/{wave_id}/download")
    def download_export(
        db: Db, wave_id: uuid.UUID,
        me: Annotated[Principal, Depends(require(Permission.EXPORT_DATA))],
    ) -> Any:
        """Отдать собранный архив.

        Имя файла выводится из кода волны, а не берётся из запроса: путь,
        собранный из пользовательского ввода, — это обход каталогов,
        а рядом лежат выгрузки других волн.
        """
        row = db.get(Wave, wave_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "волна не найдена")
        artifact = existing_archive(row.code)
        if artifact is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "пакет ещё не собран")

        audit(db, me, "download_package", "wave", str(wave_id), {"code": row.code})
        return FileResponse(
            artifact.path,
            media_type="application/zip",
            filename=f"wave-{row.code}.zip",
        )

    # -------------------------------------------------------------- журнал

    @app.get("/audit", response_class=HTMLResponse)
    def audit_log(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.VIEW_AUDIT))],
        limit: int = 200,
    ) -> HTMLResponse:
        rows = db.scalars(
            select(AuditLog).order_by(AuditLog.at.desc()).limit(max(1, min(limit, 500)))
        ).all()
        return page(request, "audit.html", me, entries=rows)

    # ----------------------------------------------------------- пользователи

    @app.get("/users", response_class=HTMLResponse)
    def users(
        request: Request, db: Db,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_USERS))],
    ) -> HTMLResponse:
        rows = db.scalars(select(User).order_by(User.login)).all()
        return page(request, "users.html", me, users=rows, roles=list(Role))

    @app.post("/users/{user_id}/toggle")
    def toggle_user(
        db: Db, user_id: uuid.UUID,
        me: Annotated[Principal, Depends(require(Permission.MANAGE_USERS))],
    ) -> Any:
        """Выключить или включить учётную запись.

        Именно выключить: журнал ссылается на автора, и удаление
        пользователя обрывает аудиторский след.
        """
        row = db.get(User, user_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "пользователь не найден")
        if str(row.id) == me.user_id:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "нельзя выключить самого себя"
            )
        was = row.is_active
        row.is_active = not was
        row.session_version += 1
        audit(db, me, "toggle_user", "user", str(user_id),
              {"is_active": {"before": was, "after": row.is_active}})
        return RedirectResponse("/users", status_code=status.HTTP_303_SEE_OTHER)

    return app


@dataclass(frozen=True, slots=True)
class QuestionnaireReport:
    """Что известно о версии анкеты для страницы."""

    valid: bool
    problem: str
    nodes: int
    questions: int
    core_keys: int
    voice_missing: int
    voice_stale: int
    voice_orphaned: int

    @property
    def voice_ready(self) -> bool:
        return self.voice_missing == 0 and self.voice_stale == 0


def _vault_ready() -> bool:
    """Заданы ли ключи шифрования номеров.

    Без них загрузка невозможна: класть номера открытым текстом нельзя
    ни при каких обстоятельствах (ТЗ §14.2).
    """
    try:
        PhoneVault.from_env()
    except RuntimeError:
        return False
    return True


def _questionnaire_report(version: QuestionnaireVersion) -> QuestionnaireReport:
    """Проверить версию: граф анкеты и полнота озвучки."""
    questionnaire = _questionnaire_of(version)
    if questionnaire is None:
        return QuestionnaireReport(
            valid=False, problem="сохранённый текст не разбирается как анкета",
            nodes=0, questions=0, core_keys=0,
            voice_missing=0, voice_stale=0, voice_orphaned=0,
        )
    from socmon.voice.catalog import questionnaire_directory

    check = verify(questionnaire, questionnaire_directory(MEDIA_DIR, questionnaire.version_id))
    asks = sum(1 for n in questionnaire.nodes if n.kind not in ("info", "end"))
    return QuestionnaireReport(
        valid=True, problem="",
        nodes=len(questionnaire.nodes), questions=asks,
        core_keys=len(questionnaire.core_keys),
        voice_missing=len(check.missing),
        voice_stale=len(check.stale),
        voice_orphaned=len(check.orphaned),
    )


def _readiness_of(db: Session, wave: Wave) -> Readiness:
    """Собрать факты о волне и прогнать проверку готовности."""
    version = db.scalar(
        select(QuestionnaireVersion).where(QuestionnaireVersion.wave_id == wave.id)
    )
    questionnaire = _questionnaire_of(version) if version else None

    missing = stale = 0
    if questionnaire is not None:
        from socmon.voice.catalog import questionnaire_directory

        check = verify(questionnaire, questionnaire_directory(MEDIA_DIR, questionnaire.version_id))
        missing, stale = len(check.missing), len(check.stale)

    lines = int(db.scalar(
        select(func.count()).select_from(PhoneLine).where(PhoneLine.is_active.is_(True))
    ) or 0)
    contacts = int(db.scalar(
        select(func.count()).select_from(Contact).where(
            Contact.do_not_call.is_(False), Contact.invalid_number.is_(False)
        )
    ) or 0)

    return readiness(WaveSetup(
        has_questionnaire=version is not None,
        questionnaire_valid=questionnaire is not None,
        voice_missing=missing,
        voice_stale=stale,
        has_reference_distributions=bool(wave.reference_distributions),
        caller_ids_active=lines,
        contacts_available=contacts,
        target_min=wave.target_min,
        calling_window_configured=True,
    ))


def _questionnaire_of(version: QuestionnaireVersion) -> Questionnaire | None:
    try:
        return Questionnaire.model_validate(yaml.safe_load(version.source_yaml))
    except Exception:
        return None


def _pick_wave(waves: Sequence[Wave], code: str) -> Wave | None:
    """Выбрать волну по коду, иначе последнюю."""
    if code:
        for wave in waves:
            if wave.code == code:
                return wave
    return waves[0] if waves else None


def _summarize_wave(db: Session, wave: Wave) -> tuple[WaveSummary | None, str | None]:
    """Посчитать сводку по волне или объяснить, почему не вышло.

    Отказ считать здесь важнее результата: без эталонных распределений
    Заказчика (§5.4) взвешивания не бывает, и показать невзвешенные цифры
    как взвешенные — худшее, что можно сделать.
    """
    if not wave.reference_distributions:
        return None, (
            "Не заданы эталонные распределения населения. Взвешивание "
            "по ТЗ §5.4 невозможно, а показывать невзвешенные цифры "
            "как взвешенные нельзя."
        )
    version = db.scalar(
        select(QuestionnaireVersion).where(QuestionnaireVersion.wave_id == wave.id)
    )
    if version is None:
        return None, "К волне не привязана версия анкеты."

    try:
        questionnaire = Questionnaire.model_validate_json(version.source_yaml) \
            if version.source_yaml.lstrip().startswith("{") \
            else Questionnaire.model_validate(yaml.safe_load(version.source_yaml))
    except Exception:
        return None, "Не удалось разобрать сохранённую версию анкеты."

    margins = [
        Margin(variable, dict(targets))
        for variable, targets in wave.reference_distributions.items()
    ]
    records = load_valid_records(
        db, wave.id, questionnaire, [m.variable for m in margins]
    )
    if not records:
        return None, "В волне нет валидных интервью."

    try:
        return summarize(
            records, margins, wave_id=wave.id, target_min=wave.target_min
        ), None
    except (WeightingError, IndexError_) as exc:
        return None, str(exc)


def _wave_stats(db: Session, wave_id: uuid.UUID) -> dict[str, int]:
    """Счётчики волны для обзора."""
    def count(*conditions: Any) -> int:
        query = select(func.count()).select_from(Interview).where(
            Interview.wave_id == wave_id, Interview.is_test.is_(False)
        )
        for condition in conditions:
            query = query.where(condition)
        return int(db.scalar(query) or 0)

    attempts = int(db.scalar(
        select(func.count()).select_from(CallAttempt).where(CallAttempt.wave_id == wave_id)
    ) or 0)
    finished = int(db.scalar(
        select(func.count()).select_from(CallAttempt).where(
            CallAttempt.wave_id == wave_id, CallAttempt.stage == CallStage.FINISHED
        )
    ) or 0)
    return {
        "attempts": attempts,
        "finished": finished,
        "interviews": count(),
        "valid": count(Interview.validity == Validity.VALID),
        "invalid": count(Interview.validity == Validity.INVALID),
        "completed": count(Interview.outcome == Outcome.COMPLETED),
    }
