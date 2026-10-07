from typing import Any

from flight_common.fl.subprocess.zmq_fl_utils import (
    ErrorDetail,
    ServerDetailedStatus,
    check_receiving,
    send_status_update,
)
from flight_common.response_return import ResponseReturn
from flight_common.zmq_client import ZmqClient
from flight_common.zmq_poller import Poller
from flight_common.zmq_socket import Socket


class ServerInfoData:
    def __init__(
        self,
        cur_round: int,
        num_rounds: int,
        job_id: str,
        rep_socket: Socket,
        poller: Poller,
        req_client: ZmqClient,
        status: str | None,
        detailed_status: ServerDetailedStatus,
    ) -> None:
        self.cur_round = cur_round
        self.num_rounds = num_rounds
        self.job_id = job_id
        self.rep_socket = rep_socket
        self.poller = poller
        self.req_client = req_client
        self.status = status
        self.detailed_status = detailed_status

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
        self,
        results: dict[str, Any] | None = None,
        detail: ErrorDetail | None = None,
    ) -> ResponseReturn | None:
        if self.status is None:
            return None

        return send_status_update(
            self.job_id,
            "server",
            self.req_client,
            self.status,
            results=results,
            detail=detail if detail is not None else self.detailed_status,
        )

    def update_detailed_status(self, msg: str | None, **kwargs: Any) -> None:
        self.detailed_status.update(msg=msg, **kwargs)

    def update_and_check(
        self,
        msg: str,
        status_kwargs: dict[str, Any] | None = None,
        results: dict[str, Any] | None = None,
    ) -> ResponseReturn | None:
        if status_kwargs is None:
            status_kwargs = {}
        self.update_detailed_status(msg=msg, **status_kwargs)
        self.status = self.check_receiving_info()
        if self.should_stop_process():
            return None
        return self.maybe_send_status_update_info(results=results)
