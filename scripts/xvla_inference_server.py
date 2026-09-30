"""Serve the chess XVLA checkpoint to the Isaac runner over localhost HTTP.

Run with the LeRobot virtualenv. Requests and responses use NumPy's non-pickle
NPZ/NPY formats so this works across the LeRobot and Isaac Python versions.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy


DEFAULT_CHECKPOINT = Path(
    "/home/roboticslab/finetuned-models/xvla-chess-lowpoly-240k/checkpoints/240000/pretrained_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0, help="Seed XVLA's stochastic action-chunk sampler.")
    return parser.parse_args()


class Inference:
    def __init__(self, checkpoint: Path, device: str) -> None:
        checkpoint = checkpoint.expanduser().resolve(strict=True)
        for filename in (
            "model.safetensors",
            "policy_preprocessor_step_7_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ):
            file = checkpoint / filename
            if not file.is_file():
                raise FileNotFoundError(f"Checkpoint file is missing: {file}")
            if not os.access(file, os.R_OK):
                raise PermissionError(f"Checkpoint file is not readable by this user: {file}")
        self.policy = XVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def predict(self, payload: bytes) -> np.ndarray:
        with np.load(io.BytesIO(payload), allow_pickle=False) as request:
            state = np.asarray(request["state"], dtype=np.float32)
            task = str(request["task"].item())
            if state.shape != (6,) or not np.isfinite(state).all():
                raise ValueError("state must be six finite joint positions in radians")
            if not task:
                raise ValueError("task must be a nonempty instruction")
            observation = {"observation.state": torch.from_numpy(state), "task": task}
            for camera_name in ("top_camera", "right_wrist_camera"):
                image = request[camera_name]
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"{camera_name} must be an HWC uint8 RGB image")
                tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
                observation[f"observation.images.{camera_name}"] = tensor

        with torch.inference_mode():
            batch = self.preprocess(observation)
            chunk = self.policy.predict_action_chunk(batch)
            if chunk.ndim != 3 or chunk.shape[0] != 1:
                raise ValueError(f"Unexpected policy action shape: {tuple(chunk.shape)}")
            actions = torch.stack(
                [self.postprocess(chunk[:, index, :]).squeeze(0) for index in range(chunk.shape[1])]
            )
            result = actions.detach().cpu().float().numpy()
        if result.ndim != 2 or result.shape[1] != 6 or not np.isfinite(result).all():
            raise ValueError(f"Invalid action chunk: shape={result.shape}")
        return result


def main() -> None:
    args = parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("This bridge is local only; bind to 127.0.0.1")
    inference = Inference(args.checkpoint, args.device)
    # Seed after model construction so every checkpoint sweep starts inference
    # from the same sampling state regardless of weight-loading internals.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/infer":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("request size must be between 1 byte and 8 MiB")
                actions = inference.predict(self.rfile.read(length))
                output = io.BytesIO()
                np.save(output, actions, allow_pickle=False)
                body = output.getvalue()
            except Exception as exc:
                self.send_error(400, explain=str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    address = (args.host, args.port)
    print(
        json.dumps(
            {
                "listening": f"http://{args.host}:{args.port}",
                "checkpoint": str(args.checkpoint),
                "seed": args.seed,
            }
        ),
        flush=True,
    )
    try:
        HTTPServer(address, Handler).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
