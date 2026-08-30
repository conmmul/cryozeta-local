"""Command-line entry points.

    python -m app.cli preflight          # read-only environment report
    python -m app.cli serve              # run the web server
    python -m app.cli submit ...         # queue a job without a browser
    python -m app.cli jobs               # list jobs
    python -m app.cli cancel <job-id>

Uses argparse only, so the CLI works without the web dependencies installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .auth import AuthError, hash_passphrase
from .config import get_settings
from .db import JobStore
from .preflight import format_report, run_preflight
from .sequences import SeqType
from .states import InferenceMode, JobStatus, RunMode
from .submission import SequenceInput, SubmissionRequest, ValidationError, prepare_job


def cmd_preflight(args: argparse.Namespace) -> int:
    report = run_preflight(get_settings())
    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "repo": str(report.repo) if report.repo else None,
                    "pixi": str(report.pixi_path) if report.pixi_path else None,
                    "recommended_env": report.recommended_env,
                    "gpus": [
                        {
                            "index": g.index,
                            "name": g.name,
                            "vram_gib": round(g.memory_total_gib, 1),
                            "compute_cap": g.compute_cap,
                        }
                        for g in (report.nvidia.gpus if report.nvidia else [])
                    ],
                    "checks": [
                        {
                            "name": c.name,
                            "status": c.status,
                            "detail": c.detail,
                            "remedy": c.remedy,
                        }
                        for c in report.checks
                    ],
                },
                indent=2,
            )
        )
    else:
        print(format_report(report))
    return 0 if report.ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .main import LanExposureError, create_app

    settings = get_settings()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port

    try:
        app = create_app(settings)
    except LanExposureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\n  CryoZeta local server -> http://{settings.host}:{settings.port}\n")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    """Queue a job from the command line.

    ``--sequence`` is repeatable and takes ``TYPE:SEQUENCE[:COUNT[:MSA_DIR]]``,
    e.g. ``protein:MKTAYIA...:2:/data/msa/chainA``.
    """
    settings = get_settings()
    settings.ensure_dirs()
    store = JobStore(settings.db_path)

    type_aliases = {
        "protein": SeqType.PROTEIN,
        "dna": SeqType.DNA,
        "rna": SeqType.RNA,
    }

    sequences: list[SequenceInput] = []
    for raw in args.sequence:
        parts = raw.split(":")
        if len(parts) < 2:
            print(f"error: malformed --sequence {raw!r}", file=sys.stderr)
            return 2
        kind = type_aliases.get(parts[0].strip().lower())
        if kind is None:
            print(f"error: unknown sequence type {parts[0]!r}", file=sys.stderr)
            return 2
        sequences.append(
            SequenceInput(
                seq_type=kind,
                sequence=parts[1],
                count=int(parts[2]) if len(parts) > 2 and parts[2] else 1,
                msa_directory=parts[3] if len(parts) > 3 else None,
            )
        )

    map_path = Path(args.map).expanduser()
    if not map_path.is_file():
        print(f"error: map not found: {map_path}", file=sys.stderr)
        return 2

    request = SubmissionRequest(
        map_source=map_path,
        map_filename=map_path.name,
        resolution=args.resolution,
        contour_level=args.contour_level,
        sequences=sequences,
        title=args.title or "",
        note=args.note or "",
        gpu_index=args.gpu,
        run_mode=RunMode(args.run_mode),
        inference_mode=InferenceMode(args.inference_mode),
        overwrite=args.overwrite,
    )

    try:
        prepared = prepare_job(request, settings, store)
    except ValidationError as exc:
        print("error: submission rejected:", file=sys.stderr)
        for message in exc.errors:
            print(f"  - {message}", file=sys.stderr)
        return 1

    print(f"queued job {prepared.job.id}")
    print(f"  input json : {prepared.paths.input_json}")
    print(f"  output dir : {prepared.paths.output_dir}")
    for warning in prepared.warnings:
        print(f"  warning    : {warning}")
    print(f"\nOpen http://{settings.host}:{settings.port}/jobs/{prepared.job.id}")

    if args.wait:
        return _wait_for(store, prepared.job.id)
    return 0


def _wait_for(store: JobStore, job_id: str, poll: float = 3.0) -> int:
    last_status = None
    while True:
        job = store.get(job_id)
        if job is None:
            return 1
        if job.status is not last_status:
            print(f"[{time.strftime('%H:%M:%S')}] {job.status.value} {job.stage}")
            last_status = job.status
        if job.status.is_terminal:
            if job.error_summary:
                print(job.error_summary)
            return 0 if job.status is JobStatus.COMPLETED else 1
        time.sleep(poll)


def cmd_jobs(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    store = JobStore(settings.db_path)
    jobs = store.list(limit=args.limit)
    if not jobs:
        print("no jobs")
        return 0
    print(f"{'JOB ID':38} {'STATUS':12} {'GPU':4} {'STAGE':22} TITLE")
    for job in jobs:
        gpu = "-" if job.gpu_index is None else str(job.gpu_index)
        print(
            f"{job.id:38} {job.status.value:12} {gpu:4} "
            f"{(job.stage or '-'):22} {job.display_title}"
        )
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = JobStore(settings.db_path)
    job = store.get(args.job_id)
    if job is None:
        print(f"error: no such job: {args.job_id}", file=sys.stderr)
        return 1
    if job.status is JobStatus.QUEUED:
        store.set_status(
            job.id, JobStatus.CANCELLED, error_summary="Cancelled from the CLI.",
            enforce=False,
        )
        print("cancelled")
        return 0
    print(
        "error: only queued jobs can be cancelled from the CLI; use the web UI "
        "to cancel a running job (the server owns its process group).",
        file=sys.stderr,
    )
    return 1



def cmd_set_password(args: argparse.Namespace) -> int:
    """Set the passphrase required when the server is network-reachable."""
    import getpass
    import os

    settings = get_settings()
    settings.ensure_dirs()

    if args.passphrase:
        # Accepting it as a flag is convenient for automation but leaks the
        # value into shell history and the process list, so say so.
        print(
            "warning: passing the passphrase as an argument exposes it to your "
            "shell history and to other users via the process list.",
            file=sys.stderr,
        )
        passphrase = args.passphrase
    else:
        passphrase = getpass.getpass("New lab passphrase: ")
        if passphrase != getpass.getpass("Repeat: "):
            print("error: passphrases did not match", file=sys.stderr)
            return 1

    try:
        digest = hash_passphrase(passphrase)
    except AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    target = settings.passphrase_file
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(digest + "\n")

    print(f"passphrase set: {target}")
    print("Authentication is now required for every page.")
    print("Existing sessions stay valid; use 'clear-sessions' to revoke them.")
    return 0


def cmd_clear_sessions(args: argparse.Namespace) -> int:
    """Invalidate every signed session cookie by rotating the secret."""
    settings = get_settings()
    settings.secret_file.unlink(missing_ok=True)
    print("session key rotated; everyone must sign in again")
    return 0


def cmd_disable_password(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.passphrase_file.is_file():
        print("no passphrase is set")
        return 0
    settings.passphrase_file.unlink()
    print("passphrase removed; the server will refuse a non-loopback bind again")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryozeta-web", description="CryoZeta local server"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="check this machine for CryoZeta readiness")
    p.add_argument("--json", action="store_true", help="emit machine-readable output")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("serve", help="run the web server")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("submit", help="queue a job without the browser")
    p.add_argument("--map", required=True, help="path to a .mrc/.map/.gz density map")
    p.add_argument("--resolution", required=True)
    p.add_argument("--contour-level", required=True, dest="contour_level")
    p.add_argument(
        "--sequence",
        action="append",
        required=True,
        metavar="TYPE:SEQ[:COUNT[:MSA_DIR]]",
        help="repeatable; TYPE is protein, dna or rna",
    )
    p.add_argument("--title")
    p.add_argument("--note")
    p.add_argument("--gpu", type=int)
    p.add_argument("--run-mode", default="standard", choices=[m.value for m in RunMode])
    p.add_argument(
        "--inference-mode",
        default="combined",
        choices=[m.value for m in InferenceMode],
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--wait", action="store_true", help="block until the job finishes")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser(
        "set-password", help="set the passphrase for network-reachable access"
    )
    p.add_argument(
        "--passphrase",
        help="supply non-interactively (exposed in shell history; prefer the prompt)",
    )
    p.set_defaults(func=cmd_set_password)

    p = sub.add_parser("clear-sessions", help="sign everyone out")
    p.set_defaults(func=cmd_clear_sessions)

    p = sub.add_parser("disable-password", help="remove the passphrase")
    p.set_defaults(func=cmd_disable_password)

    p = sub.add_parser("jobs", help="list jobs")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("cancel", help="cancel a queued job")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_cancel)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
