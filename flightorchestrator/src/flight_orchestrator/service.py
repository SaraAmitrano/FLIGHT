from typing import NewType

Service = NewType("Service", str)
FLService = NewType("FLService", Service)


def get_api_service() -> Service:
    """Returns the service name for the API."""
    return Service("api")


def get_orchestrator_service() -> Service:
    """Returns the service name for the orchestrator."""
    return Service("orchestrator")


def get_fl_server_service() -> FLService:
    """Returns the service name for the FL server."""
    return FLService(Service("flserver"))


def get_fl_client_service(client_id: int) -> FLService:
    """Returns the service name for a specific FL client."""
    return FLService(Service(f"flclient-{client_id}"))


def is_fl_client_service(service: Service) -> bool:
    """Checks if the service is a FL client service."""
    return service.startswith("flclient-")


def get_fl_client_id(service: Service) -> int:
    """Extracts the client ID from the FL client service name."""
    if not is_fl_client_service(service):
        raise ValueError(f"Invalid FL client service name: {service}")

    return int(service.split("-")[1])
