from flight_common.exceptions import FLIGHTException


class APIException(FLIGHTException):
    """
    Base class for all exceptions in the flight orchestrator.
    """
