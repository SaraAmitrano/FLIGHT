from flight_common.exceptions import FLIGHTException


class FLWRServerException(FLIGHTException):
    """
    Base class for all exceptions in the flight orchestrator.
    """


class StopRunning(FLWRServerException):
    """
    Exception to stop the running process.
    """
