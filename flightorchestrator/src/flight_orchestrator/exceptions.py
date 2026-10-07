from flight_common.exceptions import FLIGHTException


class OrchestratorException(FLIGHTException):
    """
    Base class for all exceptions in the flight orchestrator.
    """


class InvalidRequestError(OrchestratorException):
    """
    Exception raised for invalid requests.
    """
