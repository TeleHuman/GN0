"""Generic WebSocket client for remote GN0 navigation policies.

The client intentionally depends only on msgpack, numpy, and websockets so a
policy can run outside the GN0 simulator environment. Remote servers may either
use the recommended generic response schema:

    {"action_type": "discrete", "actions": [1, 1, 2]}
    {"action_type": "nav_delta", "actions": [[dx, dy, dyaw], ...]}

or a legacy nested response with ``action.nav_delta``.
"""

from __future__ import annotations

import functools
import time
from typing import Any

import msgpack
import numpy as np

try:
    import websockets.sync.client as ws_client
except Exception as exc:  # pragma: no cover - depends on local environment
    ws_client = None
    _WEBSOCKETS_IMPORT_ERROR = exc
else:
    _WEBSOCKETS_IMPORT_ERROR = None


def _pack_array(obj: Any) -> Any:
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _decode_keys(value: Any) -> Any:
    """Convert msgpack byte keys to str while preserving ndarray payloads."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            out[key] = _decode_keys(item)
        return out
    if isinstance(value, list):
        return [_decode_keys(item) for item in value]
    return value


class RemoteNavClient:
    """Synchronous msgpack-numpy WebSocket client for remote policies."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        request_timeout: float = 120.0,
        connect_timeout: float = 600.0,
        retry_interval: float = 2.0,
        metadata_timeout: float = 2.0,
    ) -> None:
        if ws_client is None:
            raise RuntimeError(
                "The GN0 environment is missing the 'websockets' package. "
                "Install it with: pip install websockets msgpack"
            ) from _WEBSOCKETS_IMPORT_ERROR

        self.host = host
        self.port = int(port)
        self.request_timeout = float(request_timeout)
        self.connect_timeout = float(connect_timeout)
        self.retry_interval = float(retry_interval)
        self.metadata_timeout = max(0.0, float(metadata_timeout))
        self.uri = f"ws://{self.host}:{self.port}"
        self._packer = Packer()
        self._conn = None
        self.metadata: dict[str, Any] = {}
        self.connect()

    def connect(self) -> None:
        deadline = time.time() + self.connect_timeout
        last_error: Exception | None = None

        while time.time() < deadline:
            try:
                self._conn = ws_client.connect(
                    self.uri,
                    max_size=None,
                    open_timeout=min(10.0, self.request_timeout),
                    close_timeout=5.0,
                    ping_interval=None,
                )
                if self.metadata_timeout > 0:
                    try:
                        metadata = self._conn.recv(timeout=self.metadata_timeout)
                    except TimeoutError:
                        metadata = None
                    if metadata is not None:
                        self.metadata = _decode_keys(unpackb(metadata))
                return
            except Exception as exc:  # pragma: no cover - live server timing
                last_error = exc
                self.close()
                time.sleep(self.retry_interval)

        raise TimeoutError(
            f"Timed out connecting to remote navigation server at {self.uri} "
            f"after {self.connect_timeout:.1f}s. Last error: {last_error}"
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def reset(self) -> dict[str, Any]:
        return self.request({"endpoint": "reset"})

    def infer(
        self,
        rgb: np.ndarray,
        instruction: str,
        session_id: str,
        images: np.ndarray | None = None,
        state: np.ndarray | None = None,
    ) -> dict[str, Any]:
        rgb = self._ensure_rgb_array(rgb, name="rgb")

        video = rgb
        if images is not None:
            images = self._ensure_rgb_array(images, name="images")
            if images.ndim == 3:
                images = images[None]
            video = images

        obs: dict[str, Any] = {
            "endpoint": "infer",
            "images": np.ascontiguousarray(video),
            "instruction": str(instruction or ""),
            "session_id": str(session_id),
        }

        if state is not None:
            obs["state.nav_pose"] = np.asarray(state, dtype=np.float32).reshape(3)

        return self.request(obs)

    @staticmethod
    def _ensure_rgb_array(value: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim not in (3, 4) or arr.shape[-1] != 3:
            raise ValueError(
                f"Expected {name} as HWC or THWC with 3 channels, got {arr.shape}"
            )
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._conn is None:
            self.connect()

        try:
            self._conn.send(self._packer.pack(payload))
            response = self._conn.recv(timeout=self.request_timeout)
        except Exception:
            self.close()
            raise

        if isinstance(response, str):
            raise RuntimeError(response)

        decoded = _decode_keys(unpackb(response))
        if not isinstance(decoded, dict):
            raise RuntimeError(f"Unexpected remote response type: {type(decoded)}")
        return decoded
