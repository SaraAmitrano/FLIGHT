import logging
import socket
from ipaddress import ip_address

logger = logging.getLogger(__name__)

IPV6: int = 6

MIN_PORT: int = 1
MAX_PORT: int = 65535


def parse_address(address: str) -> tuple[str, int, bool | None] | None:
    try:
        raw_host, _, raw_port = address.rpartition(":")

        port = int(raw_port)

        if port < MIN_PORT or port > MAX_PORT:
            raise ValueError(f"Invalid port number: {port}")

        try:
            host = raw_host.translate({ord(i): None for i in "[]"})
            version = ip_address(host).version == IPV6
        except ValueError:
            host = raw_host
            version = None
        else:
            pass

    except ValueError:
        return None

    return host, port, version


def is_port_in_use(address: str) -> bool:
    parsed_address = parse_address(address)
    if parsed_address is None:
        return True
    host, port, is_v6 = parsed_address

    protocol = socket.AF_INET6 if is_v6 else socket.AF_INET

    res = False
    with socket.socket(protocol, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.connect((host, port))
        except (ConnectionRefusedError, TimeoutError, OSError):
            res = False
        else:
            res = True
    return res
