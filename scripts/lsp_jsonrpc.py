#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote


class JsonRpcError(RuntimeError):
    pass


class JsonRpcClient:
    def __init__(self, server: str, cwd: str | None = None):
        self.proc = subprocess.Popen(
            [server],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._next_id = 1
        self._messages: "queue.Queue[dict | None]" = queue.Queue()
        self._notifications: list[dict] = []
        self._reader = threading.Thread(target=self._reader_main, daemon=True)
        self._reader.start()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.request("shutdown", None, timeout=2.0)
                self.notify("exit", None)
            except Exception:
                pass
        try:
            self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2.0)

    def stderr_text(self) -> str:
        if self.proc.stderr is None:
            return ""
        try:
            return self.proc.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def request(self, method: str, params, timeout: float = 10.0):
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {method} response")
            message = self._next_message(remaining)
            if message is None:
                raise JsonRpcError(self._server_exit_message(f"server exited while waiting for {method}"))
            if "method" in message and "id" not in message:
                self._notifications.append(message)
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise JsonRpcError(f"{method} failed: {message['error']}")
            return message.get("result")

    def notify(self, method: str, params) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def wait_for_notification(self, method: str, predicate=None, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while True:
            for index, message in enumerate(self._notifications):
                if message.get("method") != method:
                    continue
                if predicate is not None and not predicate(message.get("params")):
                    continue
                return self._notifications.pop(index)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {method} notification")
            message = self._next_message(remaining)
            if message is None:
                raise JsonRpcError(self._server_exit_message(f"server exited while waiting for {method}"))
            if "method" in message and "id" not in message:
                if message.get("method") == method and (predicate is None or predicate(message.get("params"))):
                    return message
                self._notifications.append(message)

    def _send(self, message: dict) -> None:
        if self.proc.stdin is None:
            raise JsonRpcError("server stdin is closed")
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header + body)
        self.proc.stdin.flush()

    def _server_exit_message(self, message: str) -> str:
        stderr = self.stderr_text().strip()
        if stderr:
            return f"{message}; stderr:\n{stderr}"
        return message

    def _next_message(self, timeout: float):
        try:
            return self._messages.get(timeout=timeout)
        except queue.Empty:
            if self.proc.poll() is not None:
                return None
            raise

    def _reader_main(self) -> None:
        try:
            while True:
                message = self._read_message()
                if message is None:
                    self._messages.put(None)
                    return
                self._messages.put(message)
        except Exception as exc:
            self._messages.put({"jsonrpc": "2.0", "method": "$/readerError", "params": str(exc)})
            self._messages.put(None)

    def _read_message(self):
        if self.proc.stdout is None:
            return None

        content_length = None
        while True:
            line = self.proc.stdout.readline()
            if line == b"":
                return None
            if line in (b"\r\n", b"\n"):
                break
            name, _, value = line.decode("ascii", errors="replace").partition(":")
            if name.lower() == "content-length":
                content_length = int(value.strip())

        if content_length is None:
            raise JsonRpcError("missing Content-Length header")
        body = self.proc.stdout.read(content_length)
        if len(body) != content_length:
            return None
        return json.loads(body.decode("utf-8"))


def file_uri(path: str | os.PathLike[str]) -> str:
    return "file://" + quote(str(Path(path).resolve()))


def position_of(source: str, needle: str, nth: int = 0, offset: int = 0) -> dict:
    start = -1
    cursor = 0
    for _ in range(nth + 1):
        start = source.find(needle, cursor)
        if start < 0:
            raise ValueError(f"missing needle {needle!r}")
        cursor = start + len(needle)
    index = start + offset
    line = source.count("\n", 0, index)
    line_start = source.rfind("\n", 0, index)
    character = index if line_start < 0 else index - line_start - 1
    return {"line": line, "character": character}


def full_range(source: str) -> dict:
    lines = source.splitlines()
    if not lines:
        return {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}}
    return {
        "start": {"line": 0, "character": 0},
        "end": {"line": len(lines) - 1, "character": len(lines[-1])},
    }


def initialize(client: JsonRpcClient, root_uri: str):
    result = client.request(
        "initialize",
        {
            "processId": None,
            "rootUri": root_uri,
            "workspaceFolders": [{"uri": root_uri, "name": "ora-lsp-bench"}],
            "capabilities": {
                "general": {"positionEncodings": ["utf-16"]},
                "textDocument": {
                    "completion": {"completionItem": {"documentationFormat": ["markdown", "plaintext"]}},
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "signatureHelp": {"signatureInformation": {"documentationFormat": ["markdown", "plaintext"]}},
                },
            },
        },
        timeout=10.0,
    )
    client.notify("initialized", {})
    return result


def did_open(client: JsonRpcClient, uri: str, text: str, version: int = 1) -> None:
    client.notify(
        "textDocument/didOpen",
        {
            "textDocument": {
                "uri": uri,
                "languageId": "ora",
                "version": version,
                "text": text,
            }
        },
    )


def did_change_full(client: JsonRpcClient, uri: str, text: str, version: int) -> None:
    client.notify(
        "textDocument/didChange",
        {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": text}],
        },
    )


def did_change_incremental(client: JsonRpcClient, uri: str, range_: dict, text: str, version: int) -> None:
    client.notify(
        "textDocument/didChange",
        {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"range": range_, "text": text}],
        },
    )


def wait_diagnostics(client: JsonRpcClient, uri: str, timeout: float = 30.0) -> list:
    message = client.wait_for_notification(
        "textDocument/publishDiagnostics",
        lambda params: params is not None and params.get("uri") == uri,
        timeout=timeout,
    )
    return message.get("params", {}).get("diagnostics", [])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
