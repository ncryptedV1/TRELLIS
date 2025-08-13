#!/usr/bin/env python3
"""
Trellis 3D Model Inference CLI

Usage:
  python scripts/trellis-inference.py --image /path/to/input.jpg
  python scripts/trellis-inference.py --image /path/to/view1.png /path/to/view2.png [/path/to/view3.png ...]

This script:
  - Loads credentials and endpoints from a .env file in the current working directory (via python-dotenv)
  - Retrieves an OAuth access token via client credentials
  - Submits one or more images to the Trellis inference endpoint
  - Streams the generated 3D asset to ./outputs/<input_basename>.glb

Required .env variables:
  AUTH_URL=<auth base url, e.g. https://auth.example.com>
  UAA_CLIENT_ID=<client id>
  UAA_SECRET=<client secret>
  BASE_URL=<api base url, e.g. https://api.example.com>
  DEPLOYMENT_ID=<deployment id>

Endpoint contract (as provided):
  - Token:  GET {AUTH_URL}/oauth/token?grant_type=client_credentials using Basic Auth (client_id, client_secret)
  - Single image inference: POST {BASE_URL}/v2/inference/deployments/{DEPLOYMENT_ID}/v1/asset-from-image/
      * multipart form with file under key "image_file"
  - Multi image inference: POST {BASE_URL}/v2/inference/deployments/{DEPLOYMENT_ID}/v1/asset-from-images/
      * multipart form with files under repeated key "image_files"
  - Common headers: AI-Resource-Group: default, Authorization: Bearer <token>
  - Response: binary GLTF/GLB content (can take ~1 minute)
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import mimetypes
import os
import sys
import uuid
from pathlib import Path
import threading
import time
from typing import IO, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


def load_env_using_dotenv(env_path: Path) -> None:
    """Load environment variables using python-dotenv.

    Requires the `python-dotenv` package. If unavailable, prompts the user to install it.
    """
    try:
        dotenv_module = importlib.import_module("dotenv")
        load_dotenv = getattr(dotenv_module, "load_dotenv")
    except ModuleNotFoundError:
        print(
            "python-dotenv is required. Install with: pip install python-dotenv",
            file=sys.stderr,
        )
        sys.exit(2)
    # Do not override already-set env vars
    load_dotenv(dotenv_path=str(env_path), override=False)


def require_env_var(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        print(f"Missing required environment variable: {key}", file=sys.stderr)
        sys.exit(2)
    return value


def build_basic_auth_header(client_id: str, client_secret: str) -> str:
    token = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def fetch_access_token(auth_url: str, client_id: str, client_secret: str, timeout_seconds: int = 30) -> str:
    token_endpoint = urljoin(auth_url.rstrip("/") + "/", "oauth/token")
    query = urlencode({"grant_type": "client_credentials"})
    token_url = f"{token_endpoint}?{query}"
    headers = {"Authorization": build_basic_auth_header(client_id, client_secret)}
    request = Request(token_url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as resp:
            body = resp.read()
    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = "<no body>"
        print(f"Failed to fetch access token: HTTP {exc.code}: {error_body}", file=sys.stderr)
        sys.exit(1)
    except URLError as exc:
        print(f"Failed to fetch access token: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        print("Token endpoint did not return JSON.", file=sys.stderr)
        sys.exit(1)

    access_token = payload.get("access_token")
    if not access_token:
        print("Token response missing 'access_token'.", file=sys.stderr)
        sys.exit(1)
    return access_token


def build_inference_url(base_url: str, endpoint_suffix: str, query_params: Optional[dict] = None) -> str:
    base = urljoin(base_url.rstrip("/") + "/", endpoint_suffix)
    if query_params:
        return f"{base}?{urlencode(query_params)}"
    return base


def open_inference_stream(
    base_url: str,
    image_path: Path,
    access_token: str,
    timeout_seconds: int = 600,
    query_params: Optional[dict] = None,
    deployment_id: str = "",
) -> IO[bytes]:
    """Submit image for inference and return a file-like HTTP response for streaming."""
    endpoint_path = f"v2/inference/deployments/{deployment_id}/v1/asset-from-image/"
    inference_url = build_inference_url(base_url, endpoint_path, query_params)

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "application/octet-stream"

    boundary = f"----TrellisBoundary{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")
    crlf = b"\r\n"

    file_bytes = image_path.read_bytes()
    parts = []
    parts.append(b"--" + boundary_bytes)
    parts.append(
        (
            f"Content-Disposition: form-data; name=\"image_file\"; filename=\"{image_path.name}\"".encode(
                "utf-8"
            )
        )
    )
    parts.append(f"Content-Type: {mime_type}".encode("utf-8"))
    parts.append(b"")

    body = crlf.join(parts) + crlf + file_bytes + crlf + b"--" + boundary_bytes + b"--" + crlf

    headers = {
        "Authorization": f"Bearer {access_token}",
        "AI-Resource-Group": "default",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }

    request = Request(inference_url, data=body, headers=headers, method="POST")
    try:
        response = urlopen(request, timeout=timeout_seconds)
        return response  # caller should close
    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = "<no body>"
        print(
            f"Inference request failed with HTTP {exc.code}: {error_body}",
            file=sys.stderr,
        )
        sys.exit(1)
    except URLError as exc:
        print(f"Inference request error: {exc}", file=sys.stderr)
        sys.exit(1)


def open_inference_stream_multi(
    base_url: str,
    image_paths: Sequence[Path],
    access_token: str,
    timeout_seconds: int = 600,
    query_params: Optional[dict] = None,
    deployment_id: str = "",
) -> IO[bytes]:
    """Submit multiple images for inference and return a file-like HTTP response."""
    endpoint_path = f"v2/inference/deployments/{deployment_id}/v1/asset-from-images/"
    inference_url = build_inference_url(base_url, endpoint_path, query_params)

    boundary = f"----TrellisBoundary{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")
    crlf = b"\r\n"

    parts: list[bytes] = []
    for image_path in image_paths:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "application/octet-stream"
        file_bytes = image_path.read_bytes()
        parts.append(b"--" + boundary_bytes)
        parts.append(
            (
                f"Content-Disposition: form-data; name=\"image_files\"; filename=\"{image_path.name}\"".encode(
                    "utf-8"
                )
            )
        )
        parts.append(f"Content-Type: {mime_type}".encode("utf-8"))
        parts.append(b"")
        parts.append(file_bytes)

    parts.append(b"--" + boundary_bytes + b"--")
    parts.append(b"")
    body = crlf.join(parts)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "AI-Resource-Group": "default",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }

    request = Request(inference_url, data=body, headers=headers, method="POST")
    try:
        response = urlopen(request, timeout=timeout_seconds)
        return response
    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = "<no body>"
        print(
            f"Inference request failed with HTTP {exc.code}: {error_body}",
            file=sys.stderr,
        )
        sys.exit(1)
    except URLError as exc:
        print(f"Inference request error: {exc}", file=sys.stderr)
        sys.exit(1)


def write_stream_to_file(response: IO[bytes], destination_path: Path, chunk_size: int = 8192) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("wb") as out_file:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)


def compute_output_path(image_path: Path, outputs_dir: Path) -> Path:
    base_name = image_path.stem  # without extension
    return outputs_dir / f"{base_name}.glb"


def compute_output_path_multi(image_paths: Sequence[Path], outputs_dir: Path) -> Path:
    if not image_paths:
        return outputs_dir / "output.glb"
    first = image_paths[0].stem
    return outputs_dir / f"{first}-multi.glb"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Trellis 3D model generation from one or more input images.",
    )
    parser.add_argument(
        "-i",
        "--image",
        nargs="+",
        required=True,
        help="One or more input image paths. If multiple, multi-image mode is used.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory for outputs (default: ./outputs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed (0 means random).",
    )
    parser.add_argument(
        "--ss-steps",
        type=int,
        default=12,
        help="Sparse structure sampler steps.",
    )
    parser.add_argument(
        "--ss-cfg",
        type=float,
        default=7.5,
        help="Sparse structure guidance strength.",
    )
    parser.add_argument(
        "--slat-steps",
        type=int,
        default=12,
        help="SLAT sampler steps.",
    )
    parser.add_argument(
        "--slat-cfg",
        type=float,
        default=3.0,
        help="SLAT guidance strength.",
    )
    parser.add_argument(
        "--multiimage-mode",
        choices=["stochastic", "multidiffusion"],
        default="stochastic",
        help="Multi-image conditioning mode.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=15,
        help="Connection timeout in seconds (default: 15).",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=600,
        help="Read timeout in seconds (default: 600).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    # Load .env into environment via python-dotenv
    load_env_using_dotenv(Path.cwd() / ".env")

    args = parse_args(argv)
    outputs_dir = Path.cwd() / args.outputs_dir
    image_inputs = [Path(p).expanduser().resolve() for p in args.image]
    for p in image_inputs:
        if not p.exists() or not p.is_file():
            print(f"Input image not found: {p}", file=sys.stderr)
            sys.exit(2)
    if len(image_inputs) == 1:
        single_image: Optional[Path] = image_inputs[0]
        multi_images: Optional[list[Path]] = None
        output_path = compute_output_path(single_image, outputs_dir)
    else:
        single_image = None
        multi_images = image_inputs
        output_path = compute_output_path_multi(multi_images, outputs_dir)

    auth_url = require_env_var("AUTH_URL")
    base_url = require_env_var("BASE_URL")
    client_id = require_env_var("UAA_CLIENT_ID")
    client_secret = require_env_var("UAA_SECRET")
    deployment_id = require_env_var("DEPLOYMENT_ID")

    print("Fetching access token...")
    access_token = fetch_access_token(auth_url, client_id, client_secret)

    print("Submitting for inference... (this can take a while)")

    # Live stopwatch for the whole submission process (request + streaming)
    stop_event = threading.Event()

    def _ticker(prefix: str, start_time: float, stop_flag: threading.Event) -> None:
        while not stop_flag.is_set():
            elapsed = int(time.perf_counter() - start_time)
            mins, secs = divmod(elapsed, 60)
            hrs, mins = divmod(mins, 60)
            time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
            print(f"\r{prefix}{time_str}", end="", flush=True)
            stop_flag.wait(1)
        print("", flush=True)

    submission_start = time.perf_counter()
    ticker_thread = threading.Thread(
        target=_ticker,
        args=("Elapsed (submission): ", submission_start, stop_event),
        daemon=True,
    )
    ticker_thread.start()

    # Build query params matching GenerationSettings
    query = {
        "seed": str(args.seed),
        "sparse_structure_sampler_steps": str(args.ss_steps),
        "sparse_structure_sampler_cfg_strength": str(args.ss_cfg),
        "slat_sampler_steps": str(args.slat_steps),
        "slat_sampler_cfg_strength": str(args.slat_cfg),
    }
    if multi_images is not None:
        query["multiimage_mode"] = args.multiimage_mode

    if single_image is not None:
        response = open_inference_stream(
            base_url=base_url,
            image_path=single_image,
            access_token=access_token,
            timeout_seconds=args.read_timeout,
            query_params=query,
            deployment_id=deployment_id,
        )
    else:
        response = open_inference_stream_multi(
            base_url=base_url,
            image_paths=multi_images,
            access_token=access_token,
            timeout_seconds=args.read_timeout,
            query_params=query,
            deployment_id=deployment_id,
        )

    print(f"Streaming output to: {output_path}")
    write_stream_to_file(response, output_path)
    submission_ms = int((time.perf_counter() - submission_start) * 1000)
    stop_event.set()
    ticker_thread.join(timeout=2)
    try:
        response.close()
    except Exception:
        pass
    print(f"Done. Saved: {output_path}")
    print(f"Submission time: {submission_ms} ms")


if __name__ == "__main__":
    main()


