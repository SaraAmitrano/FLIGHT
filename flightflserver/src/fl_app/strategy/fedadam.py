from flwr.server.strategy.fedadam import FedAdam as FlowerFedAdam

from .server_optimizer import ClientInitializedFedOpt

__all__ = ["FedAdam"]


class FedAdam(ClientInitializedFedOpt, FlowerFedAdam):
    """Adam on the server (Reddi et al., ICLR 2021)."""
