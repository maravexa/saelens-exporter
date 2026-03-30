"""Unix domain socket protocol for Go control plane communication.

The Go exporter sends JSON commands over the socket, the Python
worker processes them and returns JSON responses. This is the
only IPC boundary — Python never touches the network.

Protocol:
    Request:  {"command": "scan", "prompts": ["...", ...], "scan_id": "..."}
    Response: {"status": "ok", "results": [...]} or {"status": "error", "message": "..."}

Commands:
    scan        — analyze prompts for displacement
    calibrate   — run baseline calibration with provided prompts
    health      — liveness check
    shutdown    — graceful shutdown
"""

import json
import logging
import os
import socket
import struct
from dataclasses import asdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Length-prefixed framing: 4-byte big-endian uint32 + payload
HEADER_SIZE = 4
MAX_MESSAGE_SIZE = 16 * 1024 * 1024  # 16MB — large prompt batches


class SocketServer:
    """Unix domain socket server for receiving commands from Go.

    Uses length-prefixed JSON framing for reliable message boundaries.
    Single-threaded — commands are processed sequentially, which is
    fine because GPU inference is the bottleneck, not IPC.
    """

    def __init__(self, socket_path: str, backlog: int = 5):
        self.socket_path = socket_path
        self.backlog = backlog
        self._handlers: dict[str, Callable] = {}
        self._running = False
        self._sock: socket.socket | None = None

    def register(self, command: str, handler: Callable) -> None:
        """Register a handler function for a command.

        Handler signature: (payload: dict) -> dict
        """
        self._handlers[command] = handler

    def start(self) -> None:
        """Start listening for connections."""
        # Clean up stale socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.socket_path)
        self._sock.listen(self.backlog)
        self._sock.settimeout(1.0)  # allow periodic shutdown checks
        self._running = True

        # Restrict socket permissions — only owner can connect
        os.chmod(self.socket_path, 0o600)

        logger.info("Listening on %s", self.socket_path)

        while self._running:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                self._handle_connection(conn)
            except Exception as exc:
                logger.exception("Error handling connection: %s", exc)
            finally:
                conn.close()

    def stop(self) -> None:
        """Shut down the server."""
        self._running = False
        if self._sock:
            self._sock.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        logger.info("Server stopped")

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single connection: read request, dispatch, respond."""
        request = self._recv_message(conn)
        if request is None:
            return

        command = request.get("command")
        if not command:
            self._send_message(conn, {
                "status": "error",
                "message": "Missing 'command' field",
            })
            return

        handler = self._handlers.get(command)
        if handler is None:
            self._send_message(conn, {
                "status": "error",
                "message": f"Unknown command: {command}",
            })
            return

        logger.info("Processing command: %s", command)
        try:
            result = handler(request)
            self._send_message(conn, {"status": "ok", **result})
        except Exception as exc:
            logger.exception("Command '%s' failed", command)
            self._send_message(conn, {
                "status": "error",
                "message": str(exc),
            })

    @staticmethod
    def _recv_message(conn: socket.socket) -> dict[str, Any] | None:
        """Read a length-prefixed JSON message."""
        header = _recv_exact(conn, HEADER_SIZE)
        if not header:
            return None

        length = struct.unpack(">I", header)[0]
        if length > MAX_MESSAGE_SIZE:
            logger.error("Message too large: %d bytes", length)
            return None

        payload = _recv_exact(conn, length)
        if not payload:
            return None

        return json.loads(payload.decode("utf-8"))

    @staticmethod
    def _send_message(conn: socket.socket, data: dict[str, Any]) -> None:
        """Send a length-prefixed JSON message."""
        payload = json.dumps(data, default=str).encode("utf-8")
        header = struct.pack(">I", len(payload))
        conn.sendall(header + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Receive exactly n bytes from a socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
