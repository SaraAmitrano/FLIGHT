import logging
import os
from collections import defaultdict
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import ec
from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.client.client import Client
from flwr.client.client_app import ClientApp, LoadClientAppError
from flwr.client.grpc_adapter_client.connection import grpc_adapter
from flwr.client.grpc_client.connection import grpc_connection
from flwr.client.grpc_rere_client.connection import grpc_request_response
from flwr.client.message_handler.message_handler import handle_control_message
from flwr.client.nodestate.nodestate_factory import NodeStateFactory
from flwr.client.run_info_store import DeprecatedRunInfoStore
from flwr.client.typing import ClientFnExt
from flwr.common import (
    GRPC_MAX_MESSAGE_LENGTH,
    Context,
    EventType,
    Message,
    event,  # type: ignore[import]
)
from flwr.common.address import parse_address
from flwr.common.constant import (
    MAX_RETRY_DELAY,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    TRANSPORT_TYPE_GRPC_BIDI,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
    TRANSPORT_TYPES,
    ErrorCode,
)
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.message import Error
from flwr.common.retry_invoker import RetryInvoker, RetryState, exponential
from flwr.common.typing import Fab, Run, RunNotRunningException, UserConfig
from grpc import RpcError  # type: ignore[import]
from datetime import datetime, timezone
from flwr.common import recordset_compat as compat, serde
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
logger = logging.getLogger(__name__)
def _check_actionable_client(
    client: Client | None, client_fn: ClientFnExt | None
) -> None:
    if client_fn is None and client is None:
        raise ValueError(
            "Both `client_fn` and `client` are `None`, but one is required"
        )
    if client_fn is not None and client is not None:
        raise ValueError(
            "Both `client_fn` and `client` are provided, but only one is allowed"
        )
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
def start_client(
    *,
    server_address: str,
    client_fn: ClientFnExt | None = None,
    client: Client | None = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: bytes | str | None = None,
    insecure: bool | None = None,
    transport: str | None = None,
    authentication_keys: (
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None
    ) = None,
    max_retries: int | None = None,
    max_wait_time: float | None = None,
) -> None:
    """
    Start a Flower client node which connects to a Flower server.
    Warning
    -------
    This function is deprecated since 1.13.0. Use :code:`flower-supernode` command
    instead to start a SuperNode.
    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower
        server runs on the same machine on port 8080, then `server_address`
        would be `"[::]:8080"`.
    client_fn : Optional[ClientFnExt]
        A callable that instantiates a Client. (default: None)
    client : Optional[flwr.client.Client]
        An implementation of the abstract base
        class `flwr.client.Client` (default: None)
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : bool (default: True)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        Configure the transport layer. Allowed values:
        - 'grpc-bidi': gRPC, bidirectional streaming
        - 'grpc-rere': gRPC, request-response (experimental)
        - 'rest': HTTP (experimental)
    authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
        Tuple containing the elliptic curve private key and public key for
        authentication from the cryptography library.
        Source: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/
        Used to establish an authenticated connection with the server.
    max_retries: Optional[int] (default: None)
        The maximum number of times the client will try to connect to the
        server before giving up in case of a connection error. If set to None,
        there is no limit to the number of tries.
    max_wait_time: Optional[float] (default: None)
        The maximum duration before the client stops trying to
        connect to the server in case of connection error.
        If set to None, there is no limit to the total time.
    Examples
    --------
    Starting a gRPC client with an insecure server connection:
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client_fn=client_fn,
    >>> )
    Starting an SSL-enabled gRPC client using system certificates:
    >>> def client_fn(context: Context):
    >>>     return FlowerClient().to_client()
    >>>
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client_fn=client_fn,
    >>>     insecure=False,
    >>> )
    Starting an SSL-enabled gRPC client using provided certificates:
    >>> from pathlib import Path
    >>>
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client_fn=client_fn,
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> )
    """
    event(EventType.START_CLIENT_ENTER)
    start_client_internal(
        server_address=server_address,
        node_config={},
        load_client_app_fn=None,
        client_fn=client_fn,
        client=client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
        insecure=insecure,
        transport=transport,
        authentication_keys=authentication_keys,
        max_retries=max_retries,
        max_wait_time=max_wait_time,
    )
    event(EventType.START_CLIENT_LEAVE)
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def start_client_internal(
    *,
    server_address: str,
    node_config: UserConfig,
    load_client_app_fn: Callable[[str, str, str], ClientApp] | None = None,
    client_fn: ClientFnExt | None = None,
    client: Client | None = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: bytes | str | None = None,
    insecure: bool | None = None,
    transport: str | None = None,
    authentication_keys: (
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None
    ) = None,
    max_retries: int | None = None,
    max_wait_time: float | None = None,
    flwr_path: Path | None = None,
) -> None:
    """
    Start a Flower client node which connects to a Flower server.
    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower
        server runs on the same machine on port 8080, then `server_address`
        would be `"[::]:8080"`.
    node_config: UserConfig
        The configuration of the node.
    load_client_app_fn : Optional[Callable[[], ClientApp]] (default: None)
        A function that can be used to load a `ClientApp` instance.
    client_fn : Optional[ClientFnExt]
        A callable that instantiates a Client. (default: None)
    client : Optional[flwr.client.Client]
        An implementation of the abstract base
        class `flwr.client.Client` (default: None)
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : Optional[bool] (default: None)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        Configure the transport layer. Allowed values:
        - 'grpc-bidi': gRPC, bidirectional streaming
        - 'grpc-rere': gRPC, request-response (experimental)
        - 'rest': HTTP (experimental)
    authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
        Tuple containing the elliptic curve private key and public key for
        authentication from the cryptography library.
        Source: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/
        Used to establish an authenticated connection with the server.
    max_retries: Optional[int] (default: None)
        The maximum number of times the client will try to connect to the
        server before giving up in case of a connection error. If set to None,
        there is no limit to the number of tries.
    max_wait_time: Optional[float] (default: None)
        The maximum duration before the client stops trying to
        connect to the server in case of connection error.
        If set to None, there is no limit to the total time.
    flwr_path: Optional[Path] (default: None)
        The fully resolved path containing installed Flower Apps.
    clientappio_api_address : Optional[str]
        (default: `CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS`)
        The SuperNode gRPC server address.
    """
    if insecure is None:
        insecure = root_certificates is None
    if load_client_app_fn is None:
        _check_actionable_client(client, client_fn)
        if client_fn is None:
            # Wrap `Client` instance in `client_fn`
            def single_client_factory(
                context: Context,  # pylint: disable=unused-argument # noqa: ARG001
            ) -> Client:
                if client is None:  # Added this to keep mypy happy
                    raise ValueError(
                        "Both `client_fn` and `client` are `None`, but one is required"
                    )
                return client  # Always return the same instance
            client_fn = single_client_factory
        def _load_client_app(_1: str, _2: str, _3: str) -> ClientApp:
            return ClientApp(client_fn=client_fn)
        load_client_app_fn = _load_client_app
    # At this point, only `load_client_app_fn` should be used
    # Both `client` and `client_fn` must not be used directly
    # Initialize connection context manager
    connection, address, connection_error_type = _init_connection(
        transport, server_address
    )
    def _on_success(retry_state: RetryState) -> None:
        if retry_state.tries > 1:
            logger.debug(
                "Connection successful after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )
    def _on_backoff(retry_state: RetryState) -> None:
        if retry_state.tries == 1:
            logger.warning("Connection attempt failed, retrying...")
        else:
            logger.warning(
                "Connection attempt failed, retrying in %.2f seconds",
                retry_state.actual_wait,
            )
    retry_invoker = RetryInvoker(
        wait_gen_factory=lambda: exponential(max_delay=MAX_RETRY_DELAY),
        recoverable_exceptions=connection_error_type,
        max_tries=max_retries + 1 if max_retries is not None else None,
        max_time=max_wait_time,
        on_giveup=lambda retry_state: (
            logger.warning(
                "Giving up reconnection after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )
            if retry_state.tries > 1
            else None
        ),
        on_success=_on_success,
        on_backoff=_on_backoff,
    )
    # DeprecatedRunInfoStore gets initialized when the first connection is established
    run_info_store: DeprecatedRunInfoStore | None = None
    state_factory = NodeStateFactory()
    state = state_factory.state()
    runs: dict[int, Run] = {}
    if transport not in (None, TRANSPORT_TYPE_GRPC_BIDI):
        logger.warning("FLIGHT T3 audit assumes grpc-bidi, transport=%s", transport)

    # T3: gRPC-bidi application-level payload audit (excludes transport overhead).
    # These counters are local to this client process, not server round IDs.
    t3_round_counters = defaultdict(int)

    def log_t3_bytes(direction: str, msg: Message, round_number: int) -> None:
        """Rebuild Flower 1.15 grpc-bidi Protobuf without mutating the RecordSet."""
        msg_type = msg.metadata.message_type
        has_error = msg.has_error()
        try:
            if has_error:
                # An error reply is not a normal legacy grpc-bidi model update.
                logger.warning(
                    "FLIGHT T3 BYTES | timestamp=%s client=%s direction=%s "
                    "type=%s round=%s run=%s bytes=NA tensor_bytes=NA "
                    "has_error=True error_reason=%r measure=not_applicable",
                    datetime.now(timezone.utc).isoformat(),
                    os.environ.get("CLIENT_ID", "unknown"),
                    direction, msg_type, round_number, msg.metadata.run_id,
                    msg.error.reason,
                )
                return

            records = msg.content
            tensor_bytes = 0
            if direction == "RX":
                if msg_type == "train":
                    ins = compat.recordset_to_fitins(records, keep_input=True)
                    tensor_bytes = sum(len(v) for v in ins.parameters.tensors)
                    proto = ServerMessage(fit_ins=serde.fit_ins_to_proto(ins))
                elif msg_type == "evaluate":
                    ins = compat.recordset_to_evaluateins(records, keep_input=True)
                    tensor_bytes = sum(len(v) for v in ins.parameters.tensors)
                    proto = ServerMessage(evaluate_ins=serde.evaluate_ins_to_proto(ins))
                elif msg_type == "get_parameters":
                    ins = compat.recordset_to_getparametersins(records)
                    proto = ServerMessage(
                        get_parameters_ins=serde.get_parameters_ins_to_proto(ins)
                    )
                elif msg_type == "get_properties":
                    ins = compat.recordset_to_getpropertiesins(records)
                    proto = ServerMessage(
                        get_properties_ins=serde.get_properties_ins_to_proto(ins)
                    )
                elif msg_type == "reconnect":
                    seconds = int(records.configs_records["config"]["seconds"])
                    proto = ServerMessage(
                        reconnect_ins=ServerMessage.ReconnectIns(seconds=seconds)
                    )
                else:
                    raise ValueError(f"Unsupported RX message type: {msg_type}")
            elif direction == "TX":
                if msg_type == "train":
                    res = compat.recordset_to_fitres(records, keep_input=True)
                    tensor_bytes = sum(len(v) for v in res.parameters.tensors)
                    proto = ClientMessage(fit_res=serde.fit_res_to_proto(res))
                elif msg_type == "evaluate":
                    res = compat.recordset_to_evaluateres(records)
                    proto = ClientMessage(evaluate_res=serde.evaluate_res_to_proto(res))
                elif msg_type == "get_parameters":
                    res = compat.recordset_to_getparametersres(records, keep_input=True)
                    tensor_bytes = sum(len(v) for v in res.parameters.tensors)
                    proto = ClientMessage(
                        get_parameters_res=serde.get_parameters_res_to_proto(res)
                    )
                elif msg_type == "get_properties":
                    res = compat.recordset_to_getpropertiesres(records)
                    proto = ClientMessage(
                        get_properties_res=serde.get_properties_res_to_proto(res)
                    )
                elif msg_type == "reconnect":
                    reason = records.configs_records["config"]["reason"]
                    proto = ClientMessage(
                        disconnect_res=ClientMessage.DisconnectRes(reason=int(reason))
                    )
                else:
                    raise ValueError(f"Unsupported TX message type: {msg_type}")
            else:
                raise ValueError(f"Unsupported direction: {direction}")

            logger.info(
                "FLIGHT T3 BYTES | timestamp=%s client=%s direction=%s "
                "type=%s round=%s run=%s group=%s bytes=%d tensor_bytes=%d "
                "has_error=False measure=grpc_bidi_proto",
                datetime.now(timezone.utc).isoformat(),
                os.environ.get("CLIENT_ID", "unknown"),
                direction, msg_type, round_number, msg.metadata.run_id,
                msg.metadata.group_id or "none", proto.ByteSize(), tensor_bytes,
            )
        except Exception:
            logger.exception(
                "FLIGHT T3 BYTES measurement failed direction=%s type=%s round=%s",
                direction, msg_type, round_number,
            )

    while True:
        sleep_duration: int = 0
        with connection(
            address,
            insecure,
            retry_invoker,
            grpc_max_message_length,
            root_certificates,
            authentication_keys,
        ) as conn:
            receive, send, create_node, delete_node, get_run, get_fab = conn
            # Register node when connecting the first time
            if run_info_store is None:
                if create_node is None:
                    if transport not in ["grpc-bidi", None]:
                        raise NotImplementedError(
                            "All transports except `grpc-bidi` require "
                            "an implementation for `create_node()`.'"
                        )
                    # gRPC-bidi doesn't have the concept of node_id,
                    # so we set it to -1
                    run_info_store = DeprecatedRunInfoStore(node_id=-1, node_config={})
                else:
                    # Call create_node fn to register node
                    # and store node_id in state
                    if (node_id := create_node()) is None:
                        raise ValueError(
                            "Failed to register SuperNode with the SuperLink"
                        )
                    state.set_node_id(node_id)
                    run_info_store = DeprecatedRunInfoStore(
                        node_id=state.get_node_id(), node_config=node_config
                    )
            # pylint: disable=too-many-nested-blocks
            while True:
                run_id = -1
                try:
                    # Receive
                    message = receive()
                    if message is None:
                        time.sleep(3)  # Wait for 3s before asking again
                        continue
                    # Count logical operations independently. Round numbering is
                    # inferred from message order, not guaranteed by the server.
                    t3_type = message.metadata.message_type
                    if t3_type in ("train", "evaluate"):
                        t3_round_counters[t3_type] += 1
                        t3_round = t3_round_counters[t3_type]
                    else:
                        t3_round = 0
                    log_t3_bytes("RX", message, t3_round)
                    logger.info(
                        "FLIGHT NET EVENT | event=RECEIVED "
                        "timestamp=%s run=%s group=%s type=%s",
                        datetime.now(timezone.utc).isoformat(),
                        message.metadata.run_id,
                        message.metadata.group_id,
                        message.metadata.message_type,
                    )
                    logger.debug("")
                    if len(message.metadata.group_id) > 0:
                        logger.debug(
                            "[RUN %s, ROUND %s]",
                            message.metadata.run_id,
                            message.metadata.group_id,
                        )
                    logger.debug(
                        "Received: %s message %s",
                        message.metadata.message_type,
                        message.metadata.message_id,
                    )
                    # Handle control message
                    out_message, sleep_duration = handle_control_message(message)
                    if out_message:
                        # Measure before send(): grpc-bidi may consume model tensors.
                        log_t3_bytes("TX", out_message, t3_round)
                        send(out_message)
                        break
                    # Get run info
                    run_id = message.metadata.run_id
                    if run_id not in runs:
                        if get_run is not None:
                            runs[run_id] = get_run(run_id)
                        # If get_run is None, i.e., in grpc-bidi mode
                        else:
                            runs[run_id] = Run.create_empty(run_id=run_id)
                    run: Run = runs[run_id]
                    if get_fab is not None and run.fab_hash:
                        fab = get_fab(run.fab_hash, run_id)
                        # If `ClientApp` runs in the same process, install the FAB
                        install_from_fab(fab.content, flwr_path, skip_prompt=True)
                        fab_id, fab_version = get_fab_metadata(fab.content)
                    else:
                        fab = None
                        fab_id, fab_version = run.fab_id, run.fab_version
                    run.fab_id, run.fab_version = fab_id, fab_version
                    # Register context for this run
                    run_info_store.register_context(
                        run_id=run_id, run=run, flwr_path=flwr_path, fab=fab
                    )
                    # Retrieve context for this run
                    context = run_info_store.retrieve_context(run_id=run_id)
                    # Create an error reply message that will never be used to prevent
                    # the used-before-assignment linting error
                    reply_message = message.create_error_reply(
                        error=Error(code=ErrorCode.UNKNOWN, reason="Unknown")
                    )
                    # Handle app loading and task message
                    try:
                        # Load ClientApp instance
                        client_app: ClientApp = load_client_app_fn(
                            fab_id, fab_version, run.fab_hash
                        )
                        # Execute ClientApp
                        reply_message = client_app(message=message, context=context)
                    except Exception as ex:  # pylint: disable=broad-exception-caught
                        # Legacy grpc-bidi
                        if transport in ["grpc-bidi", None]:
                            # logger.exception(
                            #     "Client raised an exception.", exc_info=True
                            # )
                            # Raise exception, crash process
                            raise
                        # Don't update/change DeprecatedRunInfoStore
                        e_code = ErrorCode.CLIENT_APP_RAISED_EXCEPTION
                        # Ex fmt: "\<class 'ZeroDivisionError'>:<'division by zero'>"
                        reason = str(type(ex)) + ":<'" + str(ex) + "'>"
                        exc_entity = "ClientApp"
                        if isinstance(ex, LoadClientAppError):
                            reason = (
                                "An exception was raised when attempting to load "
                                "`ClientApp`"
                            )
                            e_code = ErrorCode.LOAD_CLIENT_APP_EXCEPTION
                            exc_entity = "SuperNode"
                        logger.exception(
                            "%s raised an exception", exc_entity, exc_info=ex
                        )
                        # Create error message
                        reply_message = message.create_error_reply(
                            error=Error(code=e_code, reason=reason)
                        )
                    else:
                        # No exception, update node state
                        run_info_store.update_context(run_id=run_id, context=context)
                    # Measure before send(): Flower serializes with keep_input=False.
                    log_t3_bytes("TX", reply_message, t3_round)
                    # Send
                    send(reply_message)
                    logger.info(
                    "FLIGHT NET EVENT | event=SENT "
                    "timestamp=%s run=%s group=%s type=%s",
                    datetime.now(timezone.utc).isoformat(),
                    message.metadata.run_id,
                    message.metadata.group_id,
                    message.metadata.message_type,
                )
                    logger.debug("Sent reply")
                except RunNotRunningException:
                    logger.debug("")
                    logger.debug(
                        "SuperNode aborted sending the reply message. "
                        "Run ID %s is not in `RUNNING` status.",
                        run_id,
                    )
                    logger.debug("")
            # pylint: enable=too-many-nested-blocks
            # Unregister node
            if delete_node is not None:
                delete_node()  # pylint: disable=not-callable
        if sleep_duration == 0:
            logger.debug("Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        logger.debug(
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)
def _init_connection(
    transport: str | None, server_address: str
) -> tuple[
    Callable[
        [
            str,
            bool,
            RetryInvoker,
            int,
            bytes | str | None,
            tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None,
        ],
        AbstractContextManager[
            tuple[
                Callable[[], Message | None],
                Callable[[Message], None],
                Callable[[], int | None] | None,
                Callable[[], None] | None,
                Callable[[int], Run] | None,
                Callable[[str, int], Fab] | None,
            ]
        ],
    ],
    str,
    type[Exception],
]:
    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        flwr_exit(
            ExitCode.COMMON_ADDRESS_INVALID,
            f"SuperLink address ({server_address}) cannot be parsed.",
        )
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"
    # Set the default transport layer
    if transport is None:
        transport = TRANSPORT_TYPE_GRPC_BIDI
    # Use either gRPC bidirectional streaming or REST request/response
    if transport == TRANSPORT_TYPE_REST:
        try:
            from flwr.client.rest_client.connection import (  # noqa: PLC0415
                http_request_response,
            )
            from requests.exceptions import (  # noqa: PLC0415
                ConnectionError as RequestsConnectionError,
            )
        except ModuleNotFoundError:
            flwr_exit(ExitCode.COMMON_MISSING_EXTRA_REST)
        if server_address[:4] != "http":
            flwr_exit(ExitCode.SUPERNODE_REST_ADDRESS_INVALID)
        connection, error_type = http_request_response, RequestsConnectionError
    elif transport == TRANSPORT_TYPE_GRPC_RERE:
        connection, error_type = grpc_request_response, RpcError
    elif transport == TRANSPORT_TYPE_GRPC_ADAPTER:
        connection, error_type = grpc_adapter, RpcError
    elif transport == TRANSPORT_TYPE_GRPC_BIDI:
        connection, error_type = grpc_connection, RpcError
    else:
        raise ValueError(
            f"Unknown transport type: {transport} (possible: {TRANSPORT_TYPES})"
        )
    return connection, address, error_type