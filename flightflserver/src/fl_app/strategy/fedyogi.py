from flwr.server.strategy.fedyogi import FedYogi as FlowerFedYogi

from .server_optimizer import ClientInitializedFedOpt

__all__ = ["FedYogi"]


class FedYogi(ClientInitializedFedOpt, FlowerFedYogi):
    """Yogi on the server (Reddi et al., ICLR 2021)."""
