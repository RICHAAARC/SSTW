"""Lightweight tests for the bounded Wan model-load/cache preflight."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation.protocol.wan_model_load_cache_preflight_contract import (
    DEFAULT_CONFIG_PATH,
    EXPECTED_LOADER_PHASES,
    EXPECTED_LOADER_PHASE_LEDGER,
    FROZEN_PROTOCOL_DIGEST,
    MODEL_ID,
    MODEL_REVISION,
    load_wan_model_load_cache_preflight_config,
    protocol_digest,
)
from experiments.generative_video_model_probe.wan_model_load_cache_preflight import (
    CacheSnapshot,
    _worker_entry,
    _progress_signature,
    snapshot_huggingface_cache,
    supervise_model_load_worker,
    validate_model_load_worker_success_state,
    validate_local_huggingface_cache,
    run_wan_model_load_cache_preflight,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += float(seconds)


class _FakeProcess:
    def __init__(
        self,
        *,
        poll_results: list[int | None] | None = None,
        terminate_exits: bool = True,
    ) -> None:
        self.pid = 4242
        self._poll_results = list(poll_results or [None])
        self._last_poll: int | None = None
        self.terminate_exits = terminate_exits
        self.terminated = 0
        self.killed = 0
        self.waited = 0
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self._poll_results:
            self._last_poll = self._poll_results.pop(0)
        if self._last_poll is not None:
            self.returncode = self._last_poll
        return self._last_poll

    def terminate(self) -> None:
        self.terminated += 1
        if self.terminate_exits:
            self.returncode = -15
            self._last_poll = -15

    def kill(self) -> None:
        self.killed += 1
        self.returncode = -9
        self._last_poll = -9

    def wait(self, timeout: float | None = None) -> int:
        self.waited += 1
        if self.returncode is None:
            if timeout is not None:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.returncode = -9
        return self.returncode


def _snapshot(
    files: int = 0,
    *,
    size: int = 0,
    incomplete: int = 0,
    locks: int = 0,
) -> CacheSnapshot:
    return CacheSnapshot(files, size, incomplete, locks, 0, 100.0)


@pytest.mark.quick
def test_preflight_contract_digest_and_authorization_are_frozen() -> None:
    config = load_wan_model_load_cache_preflight_config()
    assert config.protocol_digest == FROZEN_PROTOCOL_DIGEST
    assert config.authorization_boundary["formal_result"] is False
    assert config.authorization_boundary["stage_progression_allowed"] is False
    assert config.authorization_boundary["transformer_forward_allowed"] is False
    assert config.authorization_boundary["scheduler_step_allowed"] is False
    assert config.authorization_boundary["decode_allowed"] is False
    assert config.authorization_boundary["video_export_allowed"] is False
    assert config.cache_contract[
        "hf_snapshot_download_controlled_download_and_cache_fill_allowed"
    ] is True
    assert config.cache_contract["manual_cache_file_deletion_allowed"] is False
    assert config.cache_contract[
        "manual_lock_or_incomplete_deletion_allowed"
    ] is False
    assert config.cache_contract[
        "manual_existing_cache_rewrite_allowed"
    ] is False


@pytest.mark.quick
@pytest.mark.parametrize(
    ("path", "mutator"),
    [
        (
            ("authorization_boundary", "gate0_execution_allowed"),
            lambda payload: True,
        ),
        (
            ("protocol_contract", "worker_contract", "overall_timeout_seconds"),
            lambda payload: 2701,
        ),
        (
            (
                "protocol_contract",
                "cache_contract",
                "hub_environment",
                "HF_HUB_DOWNLOAD_TIMEOUT",
            ),
            lambda payload: "601",
        ),
        (
            (
                "protocol_contract",
                "loader_phase_contract",
                "ordered_phases",
            ),
            lambda payload: list(reversed(payload)),
        ),
    ],
)
def test_preflight_contract_mutations_fail_closed(
    tmp_path: Path,
    path: tuple[str, ...],
    mutator,
) -> None:
    payload = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = mutator(target[path[-1]])
    config_path = tmp_path / "mutated.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_wan_model_load_cache_preflight_config(config_path)


@pytest.mark.quick
def test_self_rehashed_preflight_mutation_still_fails_frozen_digest(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    payload["protocol_contract"]["worker_contract"][
        "no_progress_timeout_seconds"
    ] = 601
    payload["protocol_digest"] = protocol_digest(payload["protocol_contract"])
    config_path = tmp_path / "self_rehashed.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen digest"):
        load_wan_model_load_cache_preflight_config(config_path)


@pytest.mark.quick
def test_cache_snapshot_counts_growth_partial_locks_and_free_space(
    tmp_path: Path,
) -> None:
    (tmp_path / "weights.bin").write_bytes(b"abc")
    (tmp_path / "download.incomplete").write_bytes(b"12345")
    (tmp_path / "repo.lock").write_bytes(b"")
    snapshot = snapshot_huggingface_cache(tmp_path)
    assert snapshot.regular_file_count == 3
    assert snapshot.regular_file_bytes == 8
    assert snapshot.incomplete_file_count == 1
    assert snapshot.lock_file_count == 1
    assert snapshot.cache_scan_warning_count >= 0
    assert snapshot.free_gib > 0


@pytest.mark.quick
def test_cache_paths_require_exact_local_writable_directories(
    tmp_path: Path,
) -> None:
    home = tmp_path / "cache"
    hub = home / "hub"
    assert validate_local_huggingface_cache(
        hf_home=home,
        hf_hub_cache=hub,
        expected_hf_home=home,
        expected_hf_hub_cache=hub,
    ) == (home.resolve(), hub.resolve())
    with pytest.raises(ValueError, match="冻结本地路径"):
        validate_local_huggingface_cache(
            hf_home=tmp_path / "wrong",
            hf_hub_cache=hub,
            expected_hf_home=home,
            expected_hf_hub_cache=hub,
        )
    drive_root = tmp_path / "drive"
    drive_home = drive_root / "cache"
    drive_hub = drive_home / "hub"
    with pytest.raises(ValueError, match="Drive"):
        validate_local_huggingface_cache(
            hf_home=drive_home,
            hf_hub_cache=drive_hub,
            expected_hf_home=drive_home,
            expected_hf_hub_cache=drive_hub,
            drive_mount_root=drive_root,
        )
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "symlink_cache"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_local_huggingface_cache(
            hf_home=symlink,
            hf_hub_cache=symlink / "hub",
            expected_hf_home=symlink,
            expected_hf_hub_cache=symlink / "hub",
        )


@pytest.mark.quick
def test_cache_growth_counts_as_progress_until_worker_success(
    tmp_path: Path,
) -> None:
    config = load_wan_model_load_cache_preflight_config()
    clock = _Clock()
    process = _FakeProcess(poll_results=[None, None, None, 0])
    snapshots = iter(
        [_snapshot(0), _snapshot(1), _snapshot(2), _snapshot(3), _snapshot(3)]
    )
    result = supervise_model_load_worker(
        process=process,
        worker_state_path=tmp_path / "missing.json",
        cache_root=tmp_path,
        config=config,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        cache_snapshot=lambda _root: next(snapshots),
        process_snapshot=lambda _pid: {
            "worker_rss_bytes": 1,
            "worker_cpu_seconds": 1.0,
        },
    )
    assert result["worker_outcome"] == "success"
    assert result["termination"]["reaped"] is True
    assert process.terminated == 0


@pytest.mark.quick
def test_stable_cache_for_600_seconds_triggers_no_progress_term_kill_reap(
    tmp_path: Path,
) -> None:
    config = load_wan_model_load_cache_preflight_config()
    clock = _Clock()
    process = _FakeProcess(terminate_exits=False)
    result = supervise_model_load_worker(
        process=process,
        worker_state_path=tmp_path / "missing.json",
        cache_root=tmp_path,
        config=config,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        cache_snapshot=lambda _root: _snapshot(1, size=100),
        process_snapshot=lambda _pid: {
            "worker_rss_bytes": 10,
            "worker_cpu_seconds": 1.0,
        },
    )
    assert result["worker_outcome"] == "no_progress_timeout"
    assert process.terminated == 1
    assert process.killed == 1
    assert process.waited >= 2


@pytest.mark.quick
def test_continuing_cache_growth_still_hits_2700_second_overall_timeout(
    tmp_path: Path,
) -> None:
    config = load_wan_model_load_cache_preflight_config()
    clock = _Clock()
    process = _FakeProcess()
    call_count = 0

    def process_snapshot(_pid: int) -> dict[str, float | int]:
        nonlocal call_count
        call_count += 1
        return {"worker_rss_bytes": 10, "worker_cpu_seconds": call_count}

    cache_call_count = 0

    def growing_cache(_root: Path) -> CacheSnapshot:
        nonlocal cache_call_count
        cache_call_count += 1
        return _snapshot(cache_call_count)

    result = supervise_model_load_worker(
        process=process,
        worker_state_path=tmp_path / "missing.json",
        cache_root=tmp_path,
        config=config,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        cache_snapshot=growing_cache,
        process_snapshot=process_snapshot,
    )
    assert result["worker_outcome"] == "overall_timeout"
    assert process.terminated == 1
    assert process.waited >= 1


@pytest.mark.quick
def test_supervisor_base_exception_terminates_and_reaps_worker(
    tmp_path: Path,
) -> None:
    config = load_wan_model_load_cache_preflight_config()
    process = _FakeProcess()
    with pytest.raises(KeyboardInterrupt):
        supervise_model_load_worker(
            process=process,
            worker_state_path=tmp_path / "missing.json",
            cache_root=tmp_path,
            config=config,
            cache_snapshot=lambda _root: (_ for _ in ()).throw(
                KeyboardInterrupt()
            ),
        )
    assert process.terminated == 1
    assert process.waited >= 1


@pytest.mark.quick
def test_progress_signature_uses_only_cache_file_count_and_bytes() -> None:
    base = _progress_signature(
        {
            "model_load_cache_preflight_loader_phase_ledger": [
                "immutable_revision_resolve:start"
            ]
        },
        _snapshot(1, size=2),
    )
    variants = [
        _progress_signature(
            {
                "model_load_cache_preflight_loader_phase_ledger": [
                    "immutable_revision_resolve:start",
                    "immutable_revision_resolve:finish",
                ]
            },
            _snapshot(1, size=2),
        ),
        _progress_signature(
            {
                "model_load_cache_preflight_loader_phase_ledger": [
                    "immutable_revision_resolve:start"
                ]
            },
            _snapshot(2, size=2),
        ),
    ]
    assert variants[0] == base
    assert variants[1] != base
    assert base == _progress_signature(
        {
            "model_load_cache_preflight_loader_phase_ledger": [
                "immutable_revision_resolve:start"
            ],
            "model_load_cache_preflight_loader_phase": "diagnostic_only",
        },
        _snapshot(1, size=2, incomplete=99, locks=88),
    )


@pytest.mark.quick
def test_cpu_and_rss_jitter_do_not_prevent_no_progress_timeout(
    tmp_path: Path,
) -> None:
    config = load_wan_model_load_cache_preflight_config()
    clock = _Clock()
    process = _FakeProcess(terminate_exits=False)
    call_count = 0

    def jitter(_pid: int) -> dict[str, float | int]:
        nonlocal call_count
        call_count += 1
        return {
            "worker_rss_bytes": call_count,
            "worker_cpu_seconds": float(call_count),
        }

    result = supervise_model_load_worker(
        process=process,
        worker_state_path=tmp_path / "missing.json",
        cache_root=tmp_path,
        config=config,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        cache_snapshot=lambda _root: _snapshot(1, size=100),
        process_snapshot=jitter,
    )
    assert result["worker_outcome"] == "no_progress_timeout"


@pytest.mark.quick
def test_worker_loads_pipeline_without_forward_scheduler_decode_or_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"forward": 0, "step": 0, "decode": 0, "export": 0}

    class Forbidden:
        def __call__(self, *_args, **_kwargs):
            calls["forward"] += 1
            raise AssertionError("pipeline forward must not run")

    class FakePipe(Forbidden):
        def __init__(self) -> None:
            self.scheduler = SimpleNamespace(
                step=lambda *_a, **_k: calls.__setitem__(
                    "step", calls["step"] + 1
                )
            )
            self.vae = SimpleNamespace(
                decode=lambda *_a, **_k: calls.__setitem__(
                    "decode", calls["decode"] + 1
                )
            )

        def maybe_free_model_hooks(self) -> None:
            return None

    def fake_loader(
        _model_id,
        _dtype,
        *,
        revision,
        cache_dir,
        local_files_only,
        phase_callback,
    ):
        assert revision == MODEL_REVISION
        assert cache_dir == hub.resolve()
        assert local_files_only is True
        for phase in EXPECTED_LOADER_PHASES[1:]:
            phase_callback(phase, "start")
            phase_callback(phase, "finish")
        return FakePipe()

    fake_torch = SimpleNamespace(
        __version__="2.6.0",
        bfloat16=object(),
        cuda=SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    hub = tmp_path / "hub"
    snapshot = (
        hub
        / "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
        / "snapshots"
        / MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **_kwargs: str(snapshot),
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe.colab_runtime."
        "_load_video_generation_pipeline",
        fake_loader,
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe."
        "wan_model_load_cache_preflight.distribution_version",
        lambda _name: "0.35.2",
    )
    state = tmp_path / "state.json"
    assert _worker_entry(
        config_path=Path(DEFAULT_CONFIG_PATH).resolve(),
        worker_state_path=state,
    ) == 0
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert tuple(
        payload["model_load_cache_preflight_loader_phase_ledger"]
    ) == EXPECTED_LOADER_PHASE_LEDGER
    assert payload["model_load_cache_preflight_snapshot_commit"] == (
        MODEL_REVISION
    )
    assert payload["model_load_cache_preflight_local_files_only"] is True
    assert calls == {"forward": 0, "step": 0, "decode": 0, "export": 0}


@pytest.mark.quick
def test_snapshot_download_failure_never_calls_pipeline_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(__version__="2.6.0", bfloat16=object())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **_kwargs: (_ for _ in ()).throw(
            OSError("planned snapshot failure")
        ),
    )
    loader_called = False

    def forbidden_loader(*_args, **_kwargs):
        nonlocal loader_called
        loader_called = True
        raise AssertionError

    monkeypatch.setattr(
        "experiments.generative_video_model_probe.colab_runtime."
        "_load_video_generation_pipeline",
        forbidden_loader,
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe."
        "wan_model_load_cache_preflight.distribution_version",
        lambda _name: "0.35.2",
    )
    state = tmp_path / "state.json"
    assert _worker_entry(
        config_path=Path(DEFAULT_CONFIG_PATH).resolve(),
        worker_state_path=state,
    ) == 1
    assert loader_called is False
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["model_load_cache_preflight_worker_status"] == "failure"
    assert payload["model_load_cache_preflight_loader_phase"] == (
        "snapshot_download:start"
    )


@pytest.mark.quick
@pytest.mark.parametrize("snapshot_kind", ["outside_cache", "wrong_commit"])
def test_snapshot_path_binding_failure_never_calls_pipeline_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_kind: str,
) -> None:
    fake_torch = SimpleNamespace(__version__="2.6.0", bfloat16=object())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    hub = tmp_path / "hub"
    hub.mkdir()
    snapshot = (
        tmp_path / "outside" / MODEL_REVISION
        if snapshot_kind == "outside_cache"
        else hub / "models--Wan" / "snapshots" / ("0" * 40)
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **_kwargs: str(snapshot),
    )
    loader_called = False

    def forbidden_loader(*_args, **_kwargs):
        nonlocal loader_called
        loader_called = True
        raise AssertionError

    monkeypatch.setattr(
        "experiments.generative_video_model_probe.colab_runtime."
        "_load_video_generation_pipeline",
        forbidden_loader,
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe."
        "wan_model_load_cache_preflight.distribution_version",
        lambda _name: "0.35.2",
    )
    state = tmp_path / "state.json"
    assert _worker_entry(
        config_path=Path(DEFAULT_CONFIG_PATH).resolve(),
        worker_state_path=state,
    ) == 1
    assert loader_called is False
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["model_load_cache_preflight_worker_status"] == "failure"
    assert payload["model_load_cache_preflight_loader_phase"] == (
        "snapshot_download:start"
    )


@pytest.mark.quick
def test_preflight_source_contains_no_method_execution_calls() -> None:
    source = Path(
        "experiments/generative_video_model_probe/"
        "wan_model_load_cache_preflight.py"
    ).read_text(encoding="utf-8")
    assert "scheduler.step(" not in source
    assert "postprocess_video(" not in source
    assert "export_to_video(" not in source
    assert "pipe(" not in source


@pytest.mark.quick
@pytest.mark.parametrize(
    ("worker_outcome", "worker_status", "expected_decision"),
    [
        ("success", "success", "PASS"),
        ("worker_error", "failure", "FAIL"),
    ],
)
def test_preflight_runner_writes_only_nonformal_infrastructure_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_outcome: str,
    worker_status: str,
    expected_decision: str,
) -> None:
    output = tmp_path / "output"
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HOME", "/content/SSTW_model_cache")
    monkeypatch.setenv("HF_HUB_CACHE", "/content/SSTW_model_cache/hub")
    monkeypatch.setattr(
        "experiments.generative_video_model_probe."
        "wan_model_load_cache_preflight.validate_local_huggingface_cache",
        lambda **_kwargs: (tmp_path, hub),
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe."
        "wan_model_load_cache_preflight.supervise_model_load_worker",
        lambda **_kwargs: {
            "worker_outcome": worker_outcome,
            "worker_return_code": 0 if worker_outcome == "success" else 1,
            "termination": {
                "term_sent": False,
                "kill_sent": False,
                "reaped": True,
            },
            "worker_state": {
                "model_load_cache_preflight_worker_status": worker_status,
                "model_load_cache_preflight_loader_phase": (
                    "completed" if worker_status == "success" else "failed"
                ),
                "model_load_cache_preflight_loader_phase_ledger": (
                    list(EXPECTED_LOADER_PHASE_LEDGER)
                    if worker_status == "success"
                    else []
                ),
                "model_load_cache_preflight_worker_pid": 4242,
                "protocol_digest": FROZEN_PROTOCOL_DIGEST,
                "generation_model_id": MODEL_ID,
                "generation_model_commit_or_hash": MODEL_REVISION,
                "model_load_cache_preflight_snapshot_path_status": (
                    "local_hub_snapshot_bound"
                ),
                "model_load_cache_preflight_snapshot_path_digest": "a" * 64,
                "model_load_cache_preflight_snapshot_commit": MODEL_REVISION,
                "model_load_cache_preflight_local_files_only": True,
            },
            "cache_snapshot": _snapshot(2).as_dict(),
            "process_snapshot": {},
            "formal_result": False,
            "stage_progression_allowed": False,
            "claim_support_status": (
                "model_load_cache_preflight_only_not_method_evidence"
            ),
        },
    )
    popen_kwargs: dict[str, object] = {}

    def fake_popen(_command, **kwargs):
        popen_kwargs.update(kwargs)
        return _FakeProcess(poll_results=[0])

    decision = run_wan_model_load_cache_preflight(
        output,
        popen_factory=fake_popen,
    )
    assert (
        decision["model_load_cache_preflight_decision"] == expected_decision
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert decision["gate0_execution_allowed"] is False
    assert popen_kwargs["start_new_session"] is False
    manifest = json.loads(
        (output / "wan_model_load_cache_preflight_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        manifest["model_load_cache_preflight_decision"] == expected_decision
    )
    assert manifest["formal_result"] is False


@pytest.mark.quick
def test_supervisor_success_without_complete_loader_ledger_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HOME", "/content/SSTW_model_cache")
    monkeypatch.setenv("HF_HUB_CACHE", "/content/SSTW_model_cache/hub")
    monkeypatch.setattr(
        "experiments.generative_video_model_probe."
        "wan_model_load_cache_preflight.validate_local_huggingface_cache",
        lambda **_kwargs: (tmp_path, hub),
    )
    monkeypatch.setattr(
        "experiments.generative_video_model_probe."
        "wan_model_load_cache_preflight.supervise_model_load_worker",
        lambda **_kwargs: {
            "worker_outcome": "success",
            "worker_return_code": 0,
            "worker_state": {
                "model_load_cache_preflight_worker_status": "success",
                "model_load_cache_preflight_loader_phase": "completed",
                "model_load_cache_preflight_worker_pid": 4242,
            },
            "cache_snapshot": _snapshot().as_dict(),
            "process_snapshot": {},
        },
    )
    decision = run_wan_model_load_cache_preflight(
        output,
        popen_factory=lambda *_a, **_k: _FakeProcess(poll_results=[0]),
    )
    assert decision["model_load_cache_preflight_decision"] == "FAIL"


@pytest.mark.quick
@pytest.mark.parametrize(
    "ledger",
    [
        list(EXPECTED_LOADER_PHASE_LEDGER[:-1]),
        list(EXPECTED_LOADER_PHASE_LEDGER)
        + [EXPECTED_LOADER_PHASE_LEDGER[-1]],
        list(reversed(EXPECTED_LOADER_PHASE_LEDGER)),
    ],
)
def test_worker_success_rejects_missing_duplicate_or_reordered_phase_ledger(
    ledger: list[str],
) -> None:
    config = load_wan_model_load_cache_preflight_config()
    with pytest.raises(ValueError, match="ledger"):
        validate_model_load_worker_success_state(
            {
                "model_load_cache_preflight_worker_status": "success",
                "model_load_cache_preflight_loader_phase": "completed",
                "model_load_cache_preflight_loader_phase_ledger": ledger,
                "model_load_cache_preflight_worker_pid": 7,
                "protocol_digest": config.protocol_digest,
                "generation_model_id": MODEL_ID,
                "generation_model_commit_or_hash": MODEL_REVISION,
            },
            config=config,
            expected_worker_pid=7,
        )
