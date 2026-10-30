from flwr.common import Parameters
from flwr.server.strategy.fedavgm import FedAvgM as FlowerFedAvgM

__all__ = ["FedAvgM"]


class FedAvgM(FlowerFedAvgM):
    """
    FedAvg with server momentum (Hsu et al., 2019).

    The server momentum needs the initial global weights, which the FLIGHT
    server receives from a client after building the strategy.
    """

    def set_initial_parameters(self, parameters: Parameters) -> None:
        """Start the server optimizer from the initial global weights."""
        self.initial_parameters = parameters
