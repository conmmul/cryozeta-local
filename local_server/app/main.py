"""FastAPI application for the CryoZeta local server.

Local-only by design: binds loopback, has no accounts, and never contacts an
external inference service. Refuses to start on a non-loopback interface unless
the operator explicitly opts in.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .auth import (
    SESSION_COOKIE,
    RateLimiter,
    is_public_path,
    issue_session,
    load_or_create_secret,
    read_session,
    verify_passphrase,
)
from .config import Settings, get_settings
from .db import JobStore
from .discovery import find_cryozeta_repo, query_nvidia_cached, select_pixi_env
from .identity import describe_source, resolve_submitter
from .mapfile import MapError
from .msa import PROTEIN_NON_PAIRING, PROTEIN_PAIRING, RNA_NON_PAIRING
from .paths import job_paths
from .preflight import run_preflight
from .results import build_job_zip, collect_results, read_log_tail
from .security import SecurityError, resolve_within
from .sequences import SeqType
from .states import InferenceMode, RunMode, stages_for
from .submission import (
    SequenceInput,
    SubmissionRequest,
    ValidationError,
    prepare_job,
)
from .worker import JobManager

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


class LanExposureError(RuntimeError):
    pass


def _safe_next(target: str) -> str:
    """Only allow same-site relative redirects after login.

    Prevents /login?next=https://evil.example from turning the login form into
    an open redirect.
    """
    if not target or not target.startswith("/") or target.startswith("//"):
        return "/"
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc:
        return "/"
    return target


def _resolve_gpus(settings: Settings) -> list[int]:
    """Decide which GPU indices this server schedules onto."""
    if settings.gpus:
        return [int(p.strip()) for p in settings.gpus.split(",") if p.strip()]
    info = query_nvidia_cached()
    if info.gpus:
        return [g.index for g in info.gpus]
    # No GPU visible (e.g. running the fake runner in tests): still provide a
    # single worker slot so the job pipeline is exercisable.
    return [0]


def create_app(
    settings: Settings | None = None,
    *,
    start_workers: bool = True,
    command_builder: Any = None,
) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()

    # A non-loopback bind is only safe with a passphrase. ALLOW_LAN remains as
    # a deliberate, documented escape hatch for genuinely trusted networks.
    if not settings.is_loopback() and not settings.auth_required():
        if not settings.allow_lan:
            raise LanExposureError(
                f"Refusing to bind {settings.host} with no authentication.\n\n"
                "  Anyone who can reach this address could run code on your GPUs "
                "and read every result.\n\n"
                "  Set a passphrase (recommended):\n"
                "      python -m app.cli set-password\n\n"
                "  Or, on a genuinely trusted network, override deliberately:\n"
                "      export CRYOZETA_WEB_ALLOW_LAN=1"
            )

    store = JobStore(settings.db_path)
    session_secret = load_or_create_secret(settings.secret_file)
    rate_limiter = RateLimiter()
    repo = find_cryozeta_repo(settings.cryozeta_repo)
    # The scheduler's GPU list is fixed at startup because one worker thread is
    # created per GPU. Everything the UI *displays*, though, is re-probed per
    # request so a driver that appears late is picked up without a restart.
    gpus = _resolve_gpus(settings)
    pixi_env = settings.pixi_env or (
        select_pixi_env(query_nvidia_cached()) if query_nvidia_cached().available else None
    )

    manager = JobManager(
        settings,
        store,
        repo,
        gpus,
        pixi_env=pixi_env,
        command_builder=command_builder,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_workers:
            manager.start()
        else:
            # Still perform crash recovery so the UI is honest about state.
            store.mark_orphans_interrupted()
        yield
        manager.shutdown()
        store.close()

    app = FastAPI(title="CryoZeta local server", lifespan=lifespan)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        """Gate every page behind a session when a passphrase is configured.

        A Tailscale-identified user is already authenticated by Tailscale, so
        they are let through without a second login.
        """
        if not settings.auth_required() or is_public_path(request.url.path):
            return await call_next(request)

        if settings.trust_tailscale_headers and resolve_submitter(
            client_host=request.client.host if request.client else None,
            headers=request.headers,
            trust_proxy_headers=True,
        ):
            return await call_next(request)

        subject = read_session(
            request.cookies.get(SESSION_COOKIE),
            session_secret,
            max_age=settings.session_max_age,
        )
        if subject is None:
            target = request.url.path
            if request.url.query:
                target += f"?{request.url.query}"
            return RedirectResponse(
                url=f"/login?next={quote(target, safe='')}", status_code=303
            )
        return await call_next(request)

    app.state.settings = settings
    app.state.store = store
    app.state.manager = manager
    app.state.repo = repo
    app.state.gpus = gpus
    app.state.pixi_env = pixi_env

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals.update(
        {
            "MSA_PROTEIN_NON_PAIRING": PROTEIN_NON_PAIRING,
            "MSA_PROTEIN_PAIRING": PROTEIN_PAIRING,
            "MSA_RNA": RNA_NON_PAIRING,
        }
    )

    def fmt_time(value: float | None) -> str:
        if not value:
            return "--"
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))

    def fmt_duration(seconds: float | None) -> str:
        if seconds is None:
            return "--"
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")

    templates.env.filters["fmt_time"] = fmt_time
    templates.env.filters["fmt_duration"] = fmt_duration

    def ctx(request: Request, **extra: Any) -> dict[str, Any]:
        base = {
            "request": request,
            "settings": settings,
            "gpus": gpus,
            "repo": repo,
            "pixi_env": pixi_env,
            "nvidia": query_nvidia_cached(),
            "viewer": resolve_submitter(
                client_host=request.client.host if request.client else None,
                headers=request.headers,
                trust_proxy_headers=settings.trust_tailscale_headers,
            ),
            "auth_required": settings.auth_required(),
            "identity_source": describe_source(
                request.client.host if request.client else None,
                settings.trust_tailscale_headers,
            ),
            "counts": store.counts_by_status(),
        }
        base.update(extra)
        return base

    def render(
        request: Request, name: str, status_code: int = 200, **extra: Any
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name=name,
            context=ctx(request, **extra),
            status_code=status_code,
        )

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return RedirectResponse(url="/new", status_code=303)

    @app.get("/new", response_class=HTMLResponse)
    def new_job_page(request: Request):
        return render(
            request,
            "new_job.html",
            seq_types=list(SeqType),
            inference_modes=list(InferenceMode),
            errors=[],
            threshold=settings.large_complex_threshold,
        )

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request):
        return render(request, "jobs.html", jobs=store.list(limit=300))

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        paths = job_paths(settings.jobs_dir, job.id)
        return render(
            request,
            "job_detail.html",
            job=job,
            stages=stages_for(job.run_mode, job.inference_mode),
            log=read_log_tail(paths.log_file),
        )

    @app.get("/jobs/{job_id}/results", response_class=HTMLResponse)
    def results_page(request: Request, job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        paths = job_paths(settings.jobs_dir, job.id)
        return render(
            request,
            "results.html",
            job=job,
            results=collect_results(paths.output_dir),
        )

    @app.get("/preflight", response_class=HTMLResponse)
    def preflight_page(request: Request):
        # This page diagnoses broken machines, so it must degrade to a
        # readable error rather than a bare 500 that tells the user nothing.
        try:
            report = run_preflight(settings)
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            import traceback

            return render(
                request,
                "preflight_error.html",
                status_code=500,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        return render(request, "preflight.html", report=report)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    @app.post("/jobs")
    async def submit_job(
        request: Request,
        map_file: UploadFile,
        resolution: str = Form(...),
        contour_level: str = Form(...),
        title: str = Form(""),
        note: str = Form(""),
        gpu_index: str = Form(""),
        run_mode: str = Form(RunMode.STANDARD.value),
        inference_mode: str = Form(InferenceMode.COMBINED.value),
        overwrite: str = Form(""),
    ):
        form = await request.form()
        staging = Path(tempfile.mkdtemp(prefix="cryozeta-upload-", dir=settings.run_dir))

        try:
            if not map_file or not map_file.filename:
                raise ValidationError(["A cryo-EM map file is required."])

            staged_map = staging / Path(map_file.filename).name
            written = 0
            with open(staged_map, "wb") as dst:
                while chunk := await map_file.read(1 << 20):
                    written += len(chunk)
                    if written > settings.max_upload_bytes:
                        raise ValidationError(
                            [
                                f"Map upload exceeds the {settings.max_upload_bytes} "
                                "byte limit."
                            ]
                        )
                    dst.write(chunk)

            sequences = await _parse_sequence_rows(form, staging, settings)

            req = SubmissionRequest(
                map_source=staged_map,
                map_filename=Path(map_file.filename).name,
                resolution=resolution,
                contour_level=contour_level,
                sequences=sequences,
                title=title,
                note=note,
                gpu_index=int(gpu_index) if gpu_index.strip() else None,
                run_mode=RunMode(run_mode),
                inference_mode=InferenceMode(inference_mode),
                overwrite=bool(overwrite),
                submitted_by=resolve_submitter(
                    client_host=request.client.host if request.client else None,
                    headers=request.headers,
                    trust_proxy_headers=settings.trust_tailscale_headers,
                ),
            )
            prepared = prepare_job(req, settings, store)
        except (ValidationError, MapError, ValueError) as exc:
            errors = exc.errors if isinstance(exc, ValidationError) else [str(exc)]
            return render(
                request,
                "new_job.html",
                status_code=400,
                seq_types=list(SeqType),
                inference_modes=list(InferenceMode),
                errors=errors,
                threshold=settings.large_complex_threshold,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return RedirectResponse(url=f"/jobs/{prepared.job.id}", status_code=303)

    # ------------------------------------------------------------------
    # Job actions
    # ------------------------------------------------------------------
    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        if not manager.cancel(job_id):
            raise HTTPException(status_code=409, detail="Job is not cancellable")
        return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)

    @app.post("/jobs/{job_id}/rerun")
    def rerun_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status.is_active:
            raise HTTPException(status_code=409, detail="Job is still active")

        source = job_paths(settings.jobs_dir, job.id)

        # Stage the new job directory completely BEFORE inserting its database
        # row. A queued row is immediately claimable by a worker, so creating
        # it first would race the copy and could launch CryoZeta against a
        # half-copied tree or an input.json still pointing at the old job.
        new_id = str(uuid.uuid4())
        target = job_paths(settings.jobs_dir, new_id)
        target.ensure()
        try:
            for sub in ("input", "msa", "spec"):
                src = source.root / sub
                if src.is_dir():
                    shutil.copytree(
                        src, target.root / sub, dirs_exist_ok=True, symlinks=False
                    )
            if source.meta_file.is_file():
                shutil.copy2(source.meta_file, target.meta_file)

            # The copied JSON embeds absolute paths into the *old* job
            # directory; rewrite them so the rerun is self-contained.
            _retarget_input_json(source, target)
        except Exception:
            shutil.rmtree(target.root, ignore_errors=True)
            raise

        store.create(
            entry_name=job.entry_name,
            title=job.title,
            note=job.note,
            run_mode=job.run_mode,
            inference_mode=job.inference_mode,
            gpu_index=job.gpu_index,
            resolution=job.resolution,
            contour_level=job.contour_level,
            total_seq_len=job.total_seq_len,
            map_filename=job.map_filename,
            overwrite=True,
            sequences=job.sequences,
            submitted_by=job.submitted_by,
            job_id=new_id,
        )
        return RedirectResponse(url=f"/jobs/{new_id}", status_code=303)

    # ------------------------------------------------------------------
    # Polling / logs / downloads
    # ------------------------------------------------------------------
    @app.get("/jobs/{job_id}/status")
    def job_status(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(
            {
                "id": job.id,
                "status": job.status.value,
                "stage": job.stage,
                "gpu": job.gpu_index,
                "exit_code": job.exit_code,
                "error": job.error_summary,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "duration": job.duration_seconds,
                "terminal": job.status.is_terminal,
            }
        )

    @app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        paths = job_paths(settings.jobs_dir, job.id)
        return PlainTextResponse(read_log_tail(paths.log_file) or "(no output yet)")

    @app.get("/jobs/{job_id}/files/{relative_path:path}")
    def job_file(job_id: str, relative_path: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        paths = job_paths(settings.jobs_dir, job.id)
        try:
            target = resolve_within(paths.root, relative_path)
        except SecurityError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not target.is_file() or target.is_symlink():
            raise HTTPException(status_code=404, detail="File not found")

        inline = target.suffix.lower() in {".cif", ".json", ".csv", ".txt", ".log", ".pdb"}
        return Response(
            content=target.read_bytes(),
            media_type="text/plain" if inline else "application/octet-stream",
            headers={
                "Content-Disposition": (
                    f'{"inline" if inline else "attachment"}; filename="{target.name}"'
                )
            },
        )

    @app.get("/jobs/{job_id}/download")
    def job_zip(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        paths = job_paths(settings.jobs_dir, job.id)
        buffer = build_job_zip(paths.root, f"cryozeta-{job.entry_name}-{job.id[:8]}")
        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="cryozeta-{job.entry_name}-{job.id[:8]}.zip"'
                )
            },
        )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, next: str = "/"):
        if not settings.auth_required():
            return RedirectResponse(url="/", status_code=303)
        return render(request, "login.html", error="", next_url=_safe_next(next))

    @app.post("/login")
    def login_submit(
        request: Request, passphrase: str = Form(...), next: str = Form("/")
    ):
        if not settings.auth_required():
            return RedirectResponse(url="/", status_code=303)

        client = request.client.host if request.client else "unknown"
        if wait := rate_limiter.retry_after(client):
            return render(
                request,
                "login.html",
                status_code=429,
                error=f"Too many failed attempts. Try again in {wait} seconds.",
                next_url=_safe_next(next),
            )

        stored = settings.passphrase_hash() or ""
        if not verify_passphrase(passphrase, stored):
            rate_limiter.record_failure(client)
            return render(
                request,
                "login.html",
                status_code=401,
                error="Incorrect passphrase.",
                next_url=_safe_next(next),
            )

        rate_limiter.record_success(client)
        response = RedirectResponse(url=_safe_next(next), status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            issue_session(session_secret),
            max_age=settings.session_max_age,
            httponly=True,
            samesite="lax",
            # Only mark Secure when the request actually arrived over HTTPS,
            # otherwise the cookie would be dropped on a plain-HTTP tunnel.
            secure=request.url.scheme == "https",
            path="/",
        )
        return response

    @app.post("/logout")
    def logout():
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "gpus": gpus, "repo": str(repo) if repo else None}

    return app


async def _parse_sequence_rows(
    form: Any, staging: Path, settings: Settings
) -> list[SequenceInput]:
    """Read the repeatable sequence rows out of the multipart form."""
    types = form.getlist("seq_type")
    values = form.getlist("seq_value")
    counts = form.getlist("seq_count")
    directories = form.getlist("msa_directory")
    archives = form.getlist("msa_archive")

    rows: list[SequenceInput] = []
    # strict=False on purpose: a malformed or hand-crafted form may send
    # mismatched row counts, and processing only the complete pairs is the
    # safe behaviour. Every row is validated individually afterwards.
    for index, (seq_type, value) in enumerate(zip(types, values, strict=False)):
        if not str(value).strip():
            continue

        archive_path: Path | None = None
        if index < len(archives):
            upload = archives[index]
            # Duck-typed on purpose: raw form data yields Starlette's
            # UploadFile, which is *not* an instance of FastAPI's subclass.
            if getattr(upload, "filename", None):
                archive_path = staging / f"msa_{index}_{Path(upload.filename).name}"
                written = 0
                with open(archive_path, "wb") as dst:
                    while chunk := await upload.read(1 << 20):
                        written += len(chunk)
                        if written > settings.max_msa_archive_bytes:
                            raise ValidationError(
                                [
                                    f"Sequence {index + 1}: MSA archive exceeds the "
                                    f"{settings.max_msa_archive_bytes} byte limit."
                                ]
                            )
                        dst.write(chunk)

        rows.append(
            SequenceInput(
                seq_type=SeqType(seq_type),
                sequence=str(value),
                count=counts[index] if index < len(counts) else 1,
                msa_archive=archive_path,
                msa_directory=(
                    str(directories[index]) if index < len(directories) else None
                ),
            )
        )
    return rows


def _retarget_input_json(source: Any, target: Any) -> None:
    """Rewrite absolute paths in a copied input.json to the new job directory."""
    import json

    if not target.input_json.is_file():
        return
    text = target.input_json.read_text(encoding="utf-8")
    text = text.replace(str(source.root), str(target.root))
    target.input_json.write_text(text, encoding="utf-8")
    json.loads(text)  # fail loudly if the rewrite produced invalid JSON
