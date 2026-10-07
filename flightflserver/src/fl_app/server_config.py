from dataclasses import dataclass


@dataclass
class ServerConfig:
    """
    Flower server config.

    All attributes have default values which allows users to configure just the ones
    they care about.
    """

    num_rounds: int = 1
    round_timeout: float | None = None
    check_timeout: float | None = None

    def __repr__(self) -> str:
        """Return the string representation of the ServerConfig."""
        round_timeout_string = (
            "no round_timeout"
            if self.round_timeout is None
            else f"round_timeout={self.round_timeout}s"
        )
        check_timeout_string = (
            "no init_timeout"
            if self.check_timeout is None
            else f"init_timeout={self.check_timeout}s"
        )
        return (
            f"num_rounds={self.num_rounds}, "
            f"{round_timeout_string},"
            f"{check_timeout_string}"
        )
