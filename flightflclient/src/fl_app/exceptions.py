from flight_common.exceptions import FLIGHTException


class FLWRClientException(FLIGHTException):
    """
    Base class for all exceptions in the flight orchestrator.
    """


class StopRunning(FLWRClientException):
    """Base exception for StopRunning errors."""


class StopTraining(StopRunning):
    """Base exception for StopTraining errors."""


class StopEvaluating(StopRunning):
    """Base exception for StopEvaluating errors."""
