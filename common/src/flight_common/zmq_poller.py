from typing import Any

import zmq
from zmq import POLLIN, POLLOUT

from .zmq_socket import Socket


class Poller:
    def __init__(self) -> None:
        self._poller = zmq.Poller()
        self._socket_map: dict[zmq.Socket[Any], Socket] = {}

    def register(self, socket: Socket, flags: int = POLLIN | POLLOUT) -> None:
        self._poller.register(socket.socket, flags)
        self._socket_map[socket.socket] = socket

    def modify(self, socket: Socket, flags: int = POLLIN | POLLOUT) -> None:
        self._poller.modify(socket.socket, flags)  # type: ignore

    def unregister(self, socket: Socket) -> None:
        self._poller.unregister(socket.socket)
        self._socket_map.pop(socket.socket, None)

    def poll(self, timeout: int | None = None) -> list[tuple[Socket | None, int]]:
        raw_events = self._poller.poll(timeout)
        return [(self._socket_map[sock], event) for sock, event in raw_events]
