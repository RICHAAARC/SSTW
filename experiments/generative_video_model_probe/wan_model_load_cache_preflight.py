"""Isolated Wan model-load/cache preflight with bounded worker supervision.

The worker warms the local Hugging Face cache and proves that the immutable Wan
revision can be imported, loaded, scheduler-normalized, CPU-offloaded, and VAE
tiled.  It never calls the transformer, scheduler step, decoder, or exporter.
All outputs are non-formal infrastructure diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
from hashlib import sha256
from importlib.metadata import version as distribution_version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from evaluation.protocol.record_writer import write_json
from evaluation.protocol.wan_model_load_cache_preflight_contract import (
    DEFAULT_CONFIG_PATH,
    DIFFUSERS_VERSION,
    HF_HUB_DOWNLOAD_TIMEOUT_SECONDS,
    HF_HUB_ETAG_TIMEOUT_SECONDS,
    MODEL_ID,
    MODEL_REVISION,
    EXPECTED_LOADER_PHASES,
    EXPECTED_LOADER_PHASE_LEDGER,
    WanModelLoadCachePreflightConfig,
    load_wan_model_load_cache_preflight_config,
)
from runtime.core.progress import emit_progress_event


STATE_FILENAME = "wan_model_load_cache_preflight_state.json"
DECISION_FILENAME = "wan_model_load_cache_preflight_decision.json"
MANIFEST_FILENAME = "wan_model_load_cache_preflight_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


@dataclass(frozen=True)
class CacheSnapshot:
    regular_file_count: int
    regular_file_bytes: int
    incomplete_file_count: int
    lock_file_count: int
    cache_scan_warning_count: int
    free_gib: float

    def progress_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.regular_file_count,
            self.regular_file_bytes,
            self.incomplete_file_count,
            self.lock_file_count,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "model_load_cache_preflight_cache_regular_file_count": (
                self.regular_file_count
            ),
            "model_load_cache_preflight_cache_regular_file_bytes": (
                self.regular_file_bytes
            ),
            "model_load_cache_preflight_cache_incomplete_file_count": (
                self.incomplete_file_count
            ),
            "model_load_cache_preflight_cache_lock_file_count": (
                self.lock_file_count
            ),
            "model_load_cache_preflight_cache_scan_warning_count": (
                self.cache_scan_warning_count
            ),
            "model_load_cache_preflight_cache_free_gib": self.free_gib,
        }


def validate_local_huggingface_cache(
    *,
    hf_home: str | Path,
    hf_hub_cache: str | Path,
    expected_hf_home: str | Path,
    expected_hf_hub_cache: str | Path,
    drive_mount_root: str | Path | None = None,
) -> tuple[Path, Path]:
    home_input = Path(hf_home).expanduser()
    hub_input = Path(hf_hub_cache).expanduser()
    home = home_input
    hub = hub_input
    if not home.is_absolute() or not hub.is_absolute():
        raise ValueError("HF_HOME/HF_HUB_CACHE 必须是绝对本地路径")
    if home_input.is_symlink() or hub_input.is_symlink():
        raise ValueError("HF_HOME/HF_HUB_CACHE 不得是symlink")
    home = home.resolve()
    hub = hub.resolve()
    if home != Path(expected_hf_home).resolve():
        raise ValueError(f"HF_HOME 与冻结本地路径不一致: {home}")
    if hub != Path(expected_hf_hub_cache).resolve():
        raise ValueError(f"HF_HUB_CACHE 与冻结本地路径不一致: {hub}")
    if not hub.is_relative_to(home):
        raise ValueError("HF_HUB_CACHE 必须位于冻结 HF_HOME 内")
    if drive_mount_root is not None:
        drive_root = Path(drive_mount_root).expanduser().resolve()
        if home == drive_root or home.is_relative_to(drive_root):
            raise ValueError("Hugging Face cache 不得位于Drive挂载目录")
    for path in (home, hub):
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Hugging Face cache 必须是非symlink目录: {path}")
        probe = path / f".sstw_write_probe_{os.getpid()}"
        try:
            with probe.open("x", encoding="ascii") as handle:
                handle.write("sstw")
            probe.unlink()
        except OSError as exc:
            raise RuntimeError(f"Hugging Face cache 不可写: {path}") from exc
        os.statvfs(path)
    return home, hub


def snapshot_huggingface_cache(cache_root: str | Path) -> CacheSnapshot:
    unresolved_root = Path(cache_root)
    if unresolved_root.is_symlink():
        raise ValueError(f"HF cache snapshot root 无效: {unresolved_root}")
    root = unresolved_root.resolve()
    if not root.is_dir():
        raise ValueError(f"HF cache snapshot root 无效: {root}")
    file_count = 0
    file_bytes = 0
    incomplete_count = 0
    lock_count = 0
    for current_root, directory_names, file_names in os.walk(
        root, followlinks=False
    ):
        directory_names[:] = [
            name
            for name in directory_names
            if not (Path(current_root) / name).is_symlink()
        ]
        for name in file_names:
            path = Path(current_root) / name
            if path.is_symlink():
                continue
            try:
                stat_result = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            file_count += 1
            file_bytes += int(stat_result.st_size)
            if name.endswith(".incomplete"):
                incomplete_count += 1
            if name.endswith(".lock"):
                lock_count += 1
    statvfs = os.statvfs(root)
    free_bytes = int(statvfs.f_bavail) * int(statvfs.f_frsize)
    warning_count = 0
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir(root)
        warnings = getattr(cache_info, "warnings", ())
        warning_count = len(tuple(warnings))
    except Exception:
        warning_count = 1
    return CacheSnapshot(
        regular_file_count=file_count,
        regular_file_bytes=file_bytes,
        incomplete_file_count=incomplete_count,
        lock_file_count=lock_count,
        cache_scan_warning_count=warning_count,
        free_gib=free_bytes / (1024.0**3),
    )


def _read_proc_status_bytes(pid: int, key: str) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith(f"{key}:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _read_process_cpu_seconds(pid: int) -> float | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
        ticks = int(fields[13]) + int(fields[14])
        return ticks / float(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return None


def _read_host_available_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / (1024.0**3)
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_scalar_snapshot(pid: int) -> dict[str, float | int | None]:
    return {
        "worker_rss_bytes": _read_proc_status_bytes(pid, "VmRSS"),
        "worker_cpu_seconds": _read_process_cpu_seconds(pid),
        "parent_rss_bytes": _read_proc_status_bytes(os.getpid(), "VmRSS"),
        "host_available_gib": _read_host_available_gib(),
    }


def _read_worker_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _progress_signature(
    worker_state: Mapping[str, Any],
    cache: CacheSnapshot,
) -> tuple[Any, ...]:
    """Return only auditable progress, excluding RSS/CPU diagnostic jitter."""

    del worker_state
    return (cache.regular_file_count, cache.regular_file_bytes)


def _safe_scalar_text(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else "nonfinite"
    return str(value)


def _emit_supervisor_progress(
    *,
    status: str,
    elapsed_seconds: float,
    no_progress_seconds: float,
    worker_state: Mapping[str, Any],
    cache: CacheSnapshot,
    process: Mapping[str, Any],
) -> None:
    emit_progress_event(
        "wan_model_load_cache_preflight",
        (
            f"status={status} elapsed_seconds={elapsed_seconds:.1f} "
            f"no_progress_seconds={no_progress_seconds:.1f} "
            "loader_phase="
            f"{worker_state.get('model_load_cache_preflight_loader_phase', 'not_started')} "
            f"cache_file_count={cache.regular_file_count} "
            f"cache_bytes={cache.regular_file_bytes} "
            f"cache_incomplete={cache.incomplete_file_count} "
            f"cache_locks={cache.lock_file_count} "
            f"cache_scan_warnings={cache.cache_scan_warning_count} "
            f"cache_free_gib={cache.free_gib:.3f} "
            f"worker_rss_bytes={_safe_scalar_text(process.get('worker_rss_bytes'))} "
            f"worker_cpu_seconds={_safe_scalar_text(process.get('worker_cpu_seconds'))} "
            f"parent_rss_bytes={_safe_scalar_text(process.get('parent_rss_bytes'))} "
            f"host_available_gib={_safe_scalar_text(process.get('host_available_gib'))}"
        ),
    )


def _terminate_and_reap_worker(
    process: Any,
    *,
    grace_seconds: float,
) -> dict[str, bool]:
    term_sent = False
    kill_sent = False
    reaped = False
    if process.poll() is None:
        try:
            process.terminate()
            term_sent = True
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                kill_sent = True
            except ProcessLookupError:
                pass
    process.wait()
    reaped = True
    return {
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "reaped": reaped,
    }


def supervise_model_load_worker(
    *,
    process: Any,
    worker_state_path: Path,
    cache_root: Path,
    config: WanModelLoadCachePreflightConfig,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    cache_snapshot: Callable[[Path], CacheSnapshot] = snapshot_huggingface_cache,
    process_snapshot: Callable[[int], Mapping[str, Any]] = process_scalar_snapshot,
) -> dict[str, Any]:
    worker_contract = config.worker_contract
    overall = float(worker_contract["overall_timeout_seconds"])
    no_progress = float(worker_contract["no_progress_timeout_seconds"])
    poll = float(worker_contract["poll_interval_seconds"])
    emit_interval = float(worker_contract["progress_emit_interval_seconds"])
    grace = float(worker_contract["termination_grace_seconds"])
    started = monotonic()
    last_progress = started
    last_emit = started - emit_interval
    last_signature: tuple[Any, ...] | None = None
    termination = {"term_sent": False, "kill_sent": False, "reaped": False}
    outcome = "worker_error"
    try:
        while True:
            now = monotonic()
            state = _read_worker_state(worker_state_path)
            cache = cache_snapshot(cache_root)
            metrics = dict(process_snapshot(int(process.pid)))
            signature = _progress_signature(state, cache)
            if last_signature is None or signature != last_signature:
                last_progress = now
                last_signature = signature
            elapsed = now - started
            stalled = now - last_progress
            if now - last_emit >= emit_interval:
                _emit_supervisor_progress(
                    status="running",
                    elapsed_seconds=elapsed,
                    no_progress_seconds=stalled,
                    worker_state=state,
                    cache=cache,
                    process=metrics,
                )
                last_emit = now
            return_code = process.poll()
            if return_code is not None:
                process.wait()
                termination["reaped"] = True
                outcome = "success" if return_code == 0 else "worker_error"
                break
            if elapsed >= overall:
                outcome = "overall_timeout"
                termination = _terminate_and_reap_worker(
                    process, grace_seconds=grace
                )
                break
            if stalled >= no_progress:
                outcome = "no_progress_timeout"
                termination = _terminate_and_reap_worker(
                    process, grace_seconds=grace
                )
                break
            sleep(poll)
    except BaseException as primary_error:
        try:
            termination = _terminate_and_reap_worker(
                process, grace_seconds=grace
            )
        except BaseException as cleanup_error:
            try:
                primary_error.add_note(
                    "worker cleanup error type="
                    f"{type(cleanup_error).__name__}"
                )
            except Exception:
                pass
        raise
    final_state = _read_worker_state(worker_state_path)
    final_cache = cache_snapshot(cache_root)
    final_metrics = dict(process_snapshot(int(process.pid)))
    return {
        "worker_outcome": outcome,
        "worker_return_code": process.returncode,
        "termination": termination,
        "worker_state": final_state,
        "cache_snapshot": final_cache.as_dict(),
        "process_snapshot": final_metrics,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": (
            "model_load_cache_preflight_only_not_method_evidence"
        ),
    }


def _worker_entry(
    *,
    config_path: Path,
    worker_state_path: Path,
) -> int:
    config = load_wan_model_load_cache_preflight_config(config_path)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(HF_HUB_ETAG_TIMEOUT_SECONDS)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(
        HF_HUB_DOWNLOAD_TIMEOUT_SECONDS
    )
    sequence = 0
    loader_phase_ledger: list[str] = []
    snapshot_facts: dict[str, Any] = {}
    last_loader_phase = "worker_start"

    def phase_callback(phase: str, status: str) -> None:
        nonlocal last_loader_phase, sequence
        sequence += 1
        last_loader_phase = f"{phase}:{status}"
        if phase in EXPECTED_LOADER_PHASES:
            ledger_index = len(loader_phase_ledger)
            if (
                ledger_index >= len(EXPECTED_LOADER_PHASE_LEDGER)
                or last_loader_phase
                != EXPECTED_LOADER_PHASE_LEDGER[ledger_index]
            ):
                raise RuntimeError(
                    "Wan loader phase ledger 缺失、重复或乱序: "
                    f"observed={last_loader_phase}, "
                    "expected="
                    f"{EXPECTED_LOADER_PHASE_LEDGER[ledger_index] if ledger_index < len(EXPECTED_LOADER_PHASE_LEDGER) else 'complete'}"
                )
            loader_phase_ledger.append(last_loader_phase)
        _atomic_write_json(
            worker_state_path,
            {
                "record_kind": "wan_model_load_cache_worker_state",
                "model_load_cache_preflight_loader_phase": last_loader_phase,
                "model_load_cache_preflight_loader_phase_sequence": sequence,
                "model_load_cache_preflight_loader_phase_ledger": list(
                    loader_phase_ledger
                ),
                "model_load_cache_preflight_worker_pid": os.getpid(),
                "protocol_digest": config.protocol_digest,
                "generation_model_id": MODEL_ID,
                "generation_model_commit_or_hash": MODEL_REVISION,
                **snapshot_facts,
                "updated_at_utc": _utc_now(),
                "formal_result": False,
                "stage_progression_allowed": False,
                "claim_support_status": (
                    "model_load_cache_preflight_only_not_method_evidence"
                ),
            },
        )

    pipe = None
    try:
        phase_callback("worker_dependency_check", "start")
        import torch
        from huggingface_hub import snapshot_download
        from experiments.generative_video_model_probe.colab_runtime import (
            _load_video_generation_pipeline,
        )

        if distribution_version("diffusers") != DIFFUSERS_VERSION:
            raise RuntimeError(
                "diffusers version 与Wan model-load preflight冻结值不一致"
            )
        if str(getattr(torch, "__version__", "")).strip() == "":
            raise RuntimeError("torch version 不可识别")
        phase_callback("worker_dependency_check", "finish")
        phase_callback("snapshot_download", "start")
        snapshot_path = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(
                    Path(
                        os.environ[
                            str(
                                config.cache_contract[
                                    "hf_hub_cache_environment_variable"
                                ]
                            )
                        ]
                    ).resolve()
                ),
                token=os.environ.get("HF_TOKEN") or None,
            )
        ).resolve()
        frozen_hub = Path(
            os.environ[
                str(
                    config.cache_contract[
                        "hf_hub_cache_environment_variable"
                    ]
                )
            ]
        ).resolve()
        if (
            not snapshot_path.is_dir()
            or snapshot_path == frozen_hub
            or not snapshot_path.is_relative_to(frozen_hub)
        ):
            raise RuntimeError(
                "snapshot_download 返回路径未绑定冻结HF_HUB_CACHE"
            )
        if snapshot_path.name.lower() != MODEL_REVISION:
            raise RuntimeError(
                "snapshot_download 返回目录未绑定冻结model revision"
            )
        snapshot_relative = snapshot_path.relative_to(frozen_hub).as_posix()
        snapshot_facts.update(
            {
                "model_load_cache_preflight_snapshot_path_status": (
                    "local_hub_snapshot_bound"
                ),
                "model_load_cache_preflight_snapshot_path_digest": sha256(
                    snapshot_relative.encode("utf-8")
                ).hexdigest(),
                "model_load_cache_preflight_snapshot_commit": MODEL_REVISION,
                "model_load_cache_preflight_local_files_only": True,
            }
        )
        phase_callback("snapshot_download", "finish")
        pipe = _load_video_generation_pipeline(
            MODEL_ID,
            torch.bfloat16,
            revision=MODEL_REVISION,
            cache_dir=frozen_hub,
            local_files_only=True,
            phase_callback=phase_callback,
        )
        phase_callback("worker_cleanup", "start")
        maybe_free = getattr(pipe, "maybe_free_model_hooks", None)
        if callable(maybe_free):
            maybe_free()
        pipe = None
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        phase_callback("worker_cleanup", "finish")
        if tuple(loader_phase_ledger) != EXPECTED_LOADER_PHASE_LEDGER:
            raise RuntimeError(
                "Wan loader phase ledger 未完成全部冻结阶段"
            )
        _atomic_write_json(
            worker_state_path,
            {
                "record_kind": "wan_model_load_cache_worker_state",
                "model_load_cache_preflight_loader_phase": "completed",
                "model_load_cache_preflight_loader_phase_sequence": sequence + 1,
                "model_load_cache_preflight_loader_phase_ledger": list(
                    loader_phase_ledger
                ),
                "model_load_cache_preflight_worker_pid": os.getpid(),
                "protocol_digest": config.protocol_digest,
                "generation_model_id": MODEL_ID,
                "generation_model_commit_or_hash": MODEL_REVISION,
                **snapshot_facts,
                "updated_at_utc": _utc_now(),
                "model_load_cache_preflight_worker_status": "success",
                "formal_result": False,
                "stage_progression_allowed": False,
                "claim_support_status": (
                    "model_load_cache_preflight_only_not_method_evidence"
                ),
            },
        )
        return 0
    except BaseException as exc:
        _atomic_write_json(
            worker_state_path,
            {
                "record_kind": "wan_model_load_cache_worker_state",
                "model_load_cache_preflight_loader_phase": last_loader_phase,
                "model_load_cache_preflight_failure_phase": last_loader_phase,
                "model_load_cache_preflight_loader_phase_ledger": list(
                    loader_phase_ledger
                ),
                "model_load_cache_preflight_worker_pid": os.getpid(),
                "protocol_digest": config.protocol_digest,
                "generation_model_id": MODEL_ID,
                "generation_model_commit_or_hash": MODEL_REVISION,
                **snapshot_facts,
                "updated_at_utc": _utc_now(),
                "model_load_cache_preflight_worker_status": "failure",
                "model_load_cache_preflight_failure_type": type(exc).__name__,
                "formal_result": False,
                "stage_progression_allowed": False,
                "claim_support_status": (
                    "model_load_cache_preflight_only_not_method_evidence"
                ),
            },
        )
        return 1
    finally:
        if pipe is not None:
            try:
                maybe_free = getattr(pipe, "maybe_free_model_hooks", None)
                if callable(maybe_free):
                    maybe_free()
            except Exception:
                pass


def validate_model_load_worker_success_state(
    state: Mapping[str, Any],
    *,
    config: WanModelLoadCachePreflightConfig,
    expected_worker_pid: int,
) -> None:
    """Bind PASS to the exact ordered loader ledger and worker identity."""

    if state.get("model_load_cache_preflight_worker_status") != "success":
        raise ValueError("Wan model-load worker status 不是success")
    if state.get("model_load_cache_preflight_loader_phase") != "completed":
        raise ValueError("Wan model-load worker final phase 不是completed")
    if tuple(
        state.get("model_load_cache_preflight_loader_phase_ledger", ())
    ) != EXPECTED_LOADER_PHASE_LEDGER:
        raise ValueError("Wan model-load worker phase ledger 不完整或乱序")
    if state.get("protocol_digest") != config.protocol_digest:
        raise ValueError("Wan model-load worker protocol digest 不匹配")
    if state.get("generation_model_id") != MODEL_ID:
        raise ValueError("Wan model-load worker model ID 不匹配")
    if state.get("generation_model_commit_or_hash") != MODEL_REVISION:
        raise ValueError("Wan model-load worker model revision 不匹配")
    if (
        state.get("model_load_cache_preflight_snapshot_path_status")
        != "local_hub_snapshot_bound"
    ):
        raise ValueError("Wan model-load worker snapshot path status 不匹配")
    snapshot_digest = str(
        state.get("model_load_cache_preflight_snapshot_path_digest") or ""
    )
    if (
        len(snapshot_digest) != 64
        or any(character not in "0123456789abcdef" for character in snapshot_digest)
    ):
        raise ValueError("Wan model-load worker snapshot path digest 非法")
    if (
        state.get("model_load_cache_preflight_snapshot_commit")
        != MODEL_REVISION
    ):
        raise ValueError("Wan model-load worker snapshot commit 不匹配")
    if (
        state.get("model_load_cache_preflight_local_files_only")
        is not True
    ):
        raise ValueError("Wan model-load worker 未证明local-files-only")
    if state.get("model_load_cache_preflight_worker_pid") != int(
        expected_worker_pid
    ):
        raise ValueError("Wan model-load worker PID 不匹配")


def run_wan_model_load_cache_preflight(
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Wan model-load/cache preflight output_root 必须为空")
    config = load_wan_model_load_cache_preflight_config(config_path)
    cache_contract = config.cache_contract
    hf_home_value = os.environ.get(
        str(cache_contract["hf_home_environment_variable"]), ""
    )
    hf_hub_value = os.environ.get(
        str(cache_contract["hf_hub_cache_environment_variable"]), ""
    )
    _home, hub = validate_local_huggingface_cache(
        hf_home=hf_home_value,
        hf_hub_cache=hf_hub_value,
        expected_hf_home=cache_contract["expected_hf_home"],
        expected_hf_hub_cache=cache_contract["expected_hf_hub_cache"],
        drive_mount_root=cache_contract["drive_mount_root"],
    )
    records = output / "records"
    records.mkdir(parents=True, exist_ok=False)
    state_path = records / STATE_FILENAME
    _atomic_write_json(
        state_path,
        {
            "record_kind": "wan_model_load_cache_preflight_state",
            "model_load_cache_preflight_status": "worker_starting",
            "created_at_utc": _utc_now(),
            "protocol_digest": config.protocol_digest,
            "formal_result": False,
            "stage_progression_allowed": False,
            "claim_support_status": (
                "model_load_cache_preflight_only_not_method_evidence"
            ),
        },
    )
    command = [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.wan_model_load_cache_preflight",
        "--worker",
        "--config",
        str(Path(config_path).resolve()),
        "--worker-state",
        str(state_path),
    ]
    process = popen_factory(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=os.environ.copy(),
        start_new_session=False,
    )
    result = supervise_model_load_worker(
        process=process,
        worker_state_path=state_path,
        cache_root=hub,
        config=config,
    )
    success = False
    if result["worker_outcome"] == "success":
        try:
            validate_model_load_worker_success_state(
                result["worker_state"],
                config=config,
                expected_worker_pid=int(process.pid),
            )
        except (TypeError, ValueError):
            success = False
        else:
            success = True
    decision = {
        "manifest_kind": "sstw_wan_model_load_cache_preflight_decision",
        "profile_id": config.profile_id,
        "protocol_digest": config.protocol_digest,
        "model_load_cache_preflight_decision": "PASS" if success else "FAIL",
        "model_load_cache_preflight_worker_outcome": result["worker_outcome"],
        "model_load_cache_preflight_worker_return_code": (
            result["worker_return_code"]
        ),
        "model_load_cache_preflight_cache_snapshot": result["cache_snapshot"],
        "formal_result": False,
        "stage_progression_allowed": False,
        "gate0_execution_allowed": False,
        "automatic_followup_execution_allowed": False,
        "claim_support_status": (
            "model_load_cache_preflight_only_not_method_evidence"
        ),
    }
    write_json(records / DECISION_FILENAME, decision)
    write_json(
        output / MANIFEST_FILENAME,
        {
            "manifest_kind": "sstw_wan_model_load_cache_preflight_manifest",
            "profile_id": config.profile_id,
            "protocol_digest": config.protocol_digest,
            "record_paths": [
                f"records/{STATE_FILENAME}",
                f"records/{DECISION_FILENAME}",
            ],
            "model_load_cache_preflight_decision": decision[
                "model_load_cache_preflight_decision"
            ],
            "formal_result": False,
            "stage_progression_allowed": False,
            "claim_support_status": (
                "model_load_cache_preflight_only_not_method_evidence"
            ),
        },
    )
    return decision


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--worker-state")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.worker or not args.worker_state:
        raise SystemExit("该模块CLI仅供受监管model-load worker使用")
    return _worker_entry(
        config_path=Path(args.config).resolve(),
        worker_state_path=Path(args.worker_state).resolve(),
    )


if __name__ == "__main__":  # pragma: no cover - real Colab worker
    raise SystemExit(main())
