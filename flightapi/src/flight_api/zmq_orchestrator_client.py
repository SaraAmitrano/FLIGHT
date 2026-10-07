from .orchestrator_client import OrchestratorClient
from .zmq_api_utils import send_zeromq_request_to_orchestrator


def get_orchestrator_client() -> OrchestratorClient:
    return OrchestratorClient(send_zeromq_request_to_orchestrator)
