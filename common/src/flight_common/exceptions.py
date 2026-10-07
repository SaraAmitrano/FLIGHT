class FLIGHTException(Exception):  # noqa: N818
    """
    Base class for all exceptions in the flight orchestrator.
    """


class FLIGHTZMQRequestException(FLIGHTException):
    """
    Exception raised when a ZMQ request fails.
    """
