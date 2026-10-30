from typing import Any, override

from flwr.common import Parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager

__all__ = ["ClientInitializedFedOpt"]

# Placeholder for the FedOpt constructor: the real weights arrive from a client
_NO_PARAMETERS = Parameters(tensors=[], tensor_type="numpy.ndarray")


class ClientInitializedFedOpt:
    """
    Let a Flower FedOpt strategy (FedAdam, FedYogi) start from client weights.

    FedOpt needs the initial global weights when it is built, but the FLIGHT
    server receives them from a client only later. The server hands them over
    through set_initial_parameters().
    """

    current_weights: Any

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(initial_parameters=_NO_PARAMETERS, **kwargs)  # type: ignore[call-arg]

    @override
    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        # None: the server asks a client for the initial weights
        return None

    def set_initial_parameters(self, parameters: Parameters) -> None:
        """Start the server optimizer from the initial global weights."""
        self.current_weights = parameters_to_ndarrays(parameters)
