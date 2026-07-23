"""验证 colab_test 复用原生核心环境并只补齐兼容依赖。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.bootstrap_colab_test_runtime import bootstrap_colab_test_runtime


def _versions(*, torch: str = "2.7.1", numpy: str = "2.0.2") -> dict[str, str]:
    return {
        "torch": torch,
        "numpy": numpy,
        "diffusers": "0.36.0",
        "transformers": "4.53.0",
    }


@pytest.mark.quick
def test_colab_test_runtime_uses_ready_native_environment_without_install(
    tmp_path: Path,
) -> None:
    requirements_path = tmp_path / "compatibility.txt"
    requirements_path.write_text("diffusers>=0.35.2\n", encoding="utf-8")

    def unexpected_install(
        path: Path,
        protected: dict[str, str],
    ) -> dict[str, object]:
        raise AssertionError("ready native Colab environment must not run pip")

    decision = bootstrap_colab_test_runtime(
        requirements_path=requirements_path,
        version_reader=_versions,
        probe_runner=lambda: {"ready": True, "return_code": 0, "stderr_tail": ""},
        install_runner=unexpected_install,
    )

    assert decision["runtime_compatibility_decision"] == "PASS"
    assert decision["runtime_source"] == "native_colab_environment"
    assert decision["compatibility_install_executed"] is False
    assert decision["formal_runtime_lock_checked"] is False
    assert decision["protected_core_versions"] == {
        "torch": "2.7.1",
        "numpy": "2.0.2",
    }


@pytest.mark.quick
def test_colab_test_runtime_installs_only_after_failed_capability_probe(
    tmp_path: Path,
) -> None:
    requirements_path = tmp_path / "compatibility.txt"
    requirements_path.write_text("diffusers>=0.35.2\n", encoding="utf-8")
    probe_rows = iter(
        (
            {"ready": False, "return_code": 1, "stderr_tail": "missing diffusers"},
            {"ready": True, "return_code": 0, "stderr_tail": ""},
        )
    )
    observed: dict[str, object] = {}

    def install(
        path: Path,
        protected: dict[str, str],
    ) -> dict[str, object]:
        observed["path"] = path
        observed["protected"] = protected
        return {"return_code": 0}

    decision = bootstrap_colab_test_runtime(
        requirements_path=requirements_path,
        version_reader=_versions,
        probe_runner=lambda: next(probe_rows),
        install_runner=install,
    )

    assert decision["compatibility_install_executed"] is True
    assert observed == {
        "path": requirements_path.resolve(),
        "protected": {"torch": "2.7.1", "numpy": "2.0.2"},
    }


@pytest.mark.quick
def test_colab_test_runtime_fails_if_bootstrap_changes_native_core(
    tmp_path: Path,
) -> None:
    requirements_path = tmp_path / "compatibility.txt"
    requirements_path.write_text("diffusers>=0.35.2\n", encoding="utf-8")
    version_rows = iter((_versions(), _versions(torch="2.8.0")))
    probe_rows = iter(
        (
            {"ready": False, "return_code": 1, "stderr_tail": "missing diffusers"},
            {"ready": True, "return_code": 0, "stderr_tail": ""},
        )
    )

    with pytest.raises(RuntimeError, match="意外改变 Colab 核心计算栈"):
        bootstrap_colab_test_runtime(
            requirements_path=requirements_path,
            version_reader=lambda: next(version_rows),
            probe_runner=lambda: next(probe_rows),
            install_runner=lambda path, protected: {"return_code": 0},
        )


@pytest.mark.quick
def test_colab_test_compatibility_requirements_do_not_pin_core_stack() -> None:
    source = Path(
        "requirements/colab_test_runtime_compatibility.txt"
    ).read_text(encoding="utf-8")
    requirement_names = {
        line.split("=", 1)[0].split("<", 1)[0].strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "torch" not in requirement_names
    assert "torchvision" not in requirement_names
    assert "numpy" not in requirement_names
