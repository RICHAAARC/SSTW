"""CPU-only exact-shape artifact binding for the five-output diagnostic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.protocol.frozen_feedback_signed_response_contract import (
    build_frozen_feedback_plan,
    load_frozen_feedback_signed_response_config,
)
from experiments.generative_video_model_probe.frozen_feedback_signed_response_diagnostic import (
    _full_array_checkpoint_records,
    _stable_digest,
    _validate_generation_artifact_checkpoints,
)


pytestmark = pytest.mark.integration


def test_exact_five_by_three_artifact_paths_and_readback(
    tmp_path: Path,
) -> None:
    """Read all 15 exact-shape artifacts and reject the old MP4 path."""

    plan = build_frozen_feedback_plan(
        load_frozen_feedback_signed_response_config()
    )
    latent_paths: dict[str, Path] = {}
    decoded_paths: dict[str, Path] = {}
    saved_paths: dict[str, Path] = {}
    video_paths: dict[str, Path] = {}
    records: list[dict] = []
    for index, probe in enumerate(plan):
        latent = np.zeros((1, 16, 9, 40, 64), dtype=np.float32)
        latent_path = tmp_path / f"{index:02d}_{probe.probe_id}.npz"
        np.savez_compressed(latent_path, final_latent=latent)

        decoded_path = tmp_path / f"{index:02d}_{probe.probe_id}_decoded.npy"
        decoded = np.lib.format.open_memmap(
            decoded_path,
            mode="w+",
            dtype=np.float32,
            shape=(33, 320, 512, 3),
        )
        decoded.flush()

        saved_path = tmp_path / f"{index:02d}_{probe.probe_id}_rgb24.npy"
        saved = np.lib.format.open_memmap(
            saved_path,
            mode="w+",
            dtype=np.uint8,
            shape=(33, 320, 512, 3),
        )
        saved.flush()

        video_path = tmp_path / f"{index:02d}_{probe.probe_id}.mp4"
        video_path.write_bytes(b"diagnostic-video-placeholder")
        records.extend(
            _full_array_checkpoint_records(
                probe=probe,
                plan_index=index,
                latent_path=latent_path,
                latent=latent,
                decoded_path=decoded_path,
                decoded=decoded,
                saved_rgb24_path=saved_path,
                saved_rgb24=saved,
            )
        )
        latent_paths[probe.probe_id] = latent_path
        decoded_paths[probe.probe_id] = decoded_path
        saved_paths[probe.probe_id] = saved_path
        video_paths[probe.probe_id] = video_path
        del decoded
        del saved

    batch = SimpleNamespace(
        checkpoint_records=tuple(records),
        latent_paths=latent_paths,
        decoded_paths=decoded_paths,
        saved_rgb_paths=saved_paths,
    )
    _validate_generation_artifact_checkpoints(plan, batch)

    mutated = [dict(record) for record in records]
    saved_index = next(
        index
        for index, record in enumerate(mutated)
        if record["impulse_transfer_checkpoint_id"]
        == "T_saved_video_full_rgb24"
    )
    probe_id = str(mutated[saved_index]["impulse_probe_id"])
    mutated[saved_index]["impulse_transfer_checkpoint_source_path"] = str(
        video_paths[probe_id]
    )
    mutated[saved_index]["impulse_transfer_checkpoint_record_id"] = (
        _stable_digest(
            {
                key: value
                for key, value in mutated[saved_index].items()
                if key != "impulse_transfer_checkpoint_record_id"
            }
        )
    )
    with pytest.raises(
        RuntimeError,
        match="generation checkpoint binding",
    ):
        _validate_generation_artifact_checkpoints(
            plan,
            SimpleNamespace(
                checkpoint_records=tuple(mutated),
                latent_paths=latent_paths,
                decoded_paths=decoded_paths,
                saved_rgb_paths=saved_paths,
            ),
        )
