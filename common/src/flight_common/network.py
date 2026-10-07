import os
from pathlib import Path
from typing import Literal, get_args

from .models.zmq.network_map import (
    NetworkData,
    NetworkPortData,
    PortMapping,
    RecvSendData,
)

CommunicationType = Literal["SEND", "RECV"]

COMMUNICATION_TYPES = get_args(CommunicationType)


def read_network() -> NetworkData:
    configs_dir = Path(os.getenv("FLIGHT_CONFIGS_DIR", "/configs")).resolve()

    return NetworkData.model_validate_json(
        (configs_dir / "zmq_network_centralized.json").read_text()
    )


def get_send_ports(network_data: NetworkData, service: str) -> PortMapping:
    match service:
        case "api":
            return network_data.api
        case "orchestrator":
            return network_data.orchestrator
        case _:
            return PortMapping.model_validate({})


def get_recv_ports(network_data: NetworkData, service: str) -> PortMapping:
    out: dict[str, int] = {}
    if service in network_data.api.root:
        out.update({"api": network_data.api.root[service]})
    if service in network_data.orchestrator.root:
        out.update({"orchestrator": network_data.orchestrator.root[service]})
    return PortMapping.model_validate(out)


def get_port_data(network_data: NetworkData, service: str) -> NetworkPortData:
    # return {
    #     "SEND": get_send_ports(network_data, service),
    #     "RECV": get_recv_ports(network_data, service),
    # }
    return NetworkPortData(
        SEND=get_send_ports(network_data, service),
        RECV=get_recv_ports(network_data, service),
    )


def get_service_send_address(
    network_data: NetworkData,
    service_req: str,
    service_rep: str,
    protocol: str,
    address_override: str | None = None,
) -> str | None:
    port_data = get_port_data(network_data, service_req)
    rep_send_port_data = port_data.SEND.get(service_rep)

    if rep_send_port_data is None:
        return None

    if address_override is not None:
        return f"{protocol}://{address_override}:3000"

    return f"{protocol}://{service_rep}:{rep_send_port_data}"


def get_service_recv_address(
    network_data: NetworkData,
    service_req: str,
    service_rep: str,
    protocol: str,
    address_override: str | None = None,
) -> str | None:
    port_data = get_port_data(network_data, service_rep)
    req_recv_port_data = port_data.RECV.get(service_req)

    if req_recv_port_data is None:
        return None

    if address_override is not None:
        return f"{protocol}://*:3000"

    return f"{protocol}://*:{req_recv_port_data}"


def get_service_addresses(
    network_data: NetworkData,
    start_service: str,
    end_services: list[str],
    protocol: str,
    service_to_address: dict[str, str] | None = None,
) -> RecvSendData:
    out = RecvSendData(SEND={}, RECV={})

    conn_data: dict[str, str] = {}
    for service in end_services:
        address = get_service_send_address(
            network_data,
            start_service,
            service,
            protocol,
            (service_to_address or {}).get(service),
        )

        if address is not None:
            conn_data[service] = address

    if len(conn_data) > 0:
        out.SEND.update(conn_data)

    conn_data = {}
    for service in end_services:
        address = get_service_recv_address(
            network_data,
            service,
            start_service,
            protocol,
            (service_to_address or {}).get(service),
        )

        if address is not None:
            conn_data[service] = address
    if len(conn_data) > 0:
        out.RECV.update(conn_data)
    return out


__all__ = [
    "NetworkData",
    "get_service_addresses",
    "get_service_recv_address",
    "get_service_send_address",
    "read_network",
]
