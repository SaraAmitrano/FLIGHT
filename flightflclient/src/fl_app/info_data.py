from typing import Any

from flight_common.fl.subprocess.zmq_fl_utils import (
    ClientDetailedStatus,
    check_receiving,
    send_status_update,
)
from flight_common.response_return import ResponseReturn
from flight_common.zmq_client import ZmqClient
from flight_common.zmq_poller import Poller
from flight_common.zmq_socket import Socket


class ClientInfoData:
    def __init__(
        self,
        client_id: int,
        cur_round: int,
        num_rounds: int,
        job_id: str,
        rep_socket: Socket,
        poller: Poller,
        req_client: ZmqClient,
        status: str | None,
        detailed_status: ClientDetailedStatus,
        carbon_intensity_kg_per_kwh: float = 0.279,
        interval_sec: float = 5.0,
        country_code: str = "ITA",
    ) -> None:
        self.client_id = client_id
        self.cur_round = cur_round
        self.num_rounds = num_rounds
        self.job_id = job_id
        self.rep_socket = rep_socket
        self.poller = poller
        self.req_client = req_client
        self.status = status
        self.detailed_status = detailed_status
        self.carbon_intensity_kg_per_kwh = carbon_intensity_kg_per_kwh
        self.interval_sec = interval_sec
        self.country_code = country_code

    def check_receiving_info(self) -> str | None:
        return check_receiving(
            self.job_id,
            self.rep_socket,
            self.poller,
            self.req_client,
            self.status,
            self.detailed_status,
        )

    def should_stop_process(self) -> bool:
        """Check if the training should stop based on the current status."""
        return self.status is None or self.status == "stopping"

    def maybe_send_status_update_info(
        self, results: dict[str, Any] | None = None
    ) -> ResponseReturn | None:
        if self.status is None:
            return None

        return send_status_update(
            self.job_id,
            f"client-{self.client_id}",
            self.req_client,
            self.status,
            results=results,
            detail=self.detailed_status,
        )

    def update_detailed_status(self, msg: str, **status_kwargs: Any) -> None:
        self.detailed_status.update(msg=msg, **status_kwargs)

    def update_and_check(
        self,
        msg: str | None,
        exc: type[Exception] | None,
        status_kwargs: dict[str, Any] | None = None,
        results: dict[str, Any] | None = None,
    ) -> None:
        if msg is not None:
            if status_kwargs is None:
                status_kwargs = {}
            self.update_detailed_status(msg=msg, **status_kwargs)
        self.status = self.check_receiving_info()
        if exc is not None and self.should_stop_process():
            raise exc
        if msg is not None:
            self.maybe_send_status_update_info(results=results)
