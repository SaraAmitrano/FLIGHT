#!/usr/bin/env python3

import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict, TypeVar, cast

import yaml

logger = logging.getLogger(__name__)


class ArgParserError(Exception):
    pass


ENVS_SPLIT_LENGTH = 2


def validate_env_file(rel_path: Path, current_dir: Path) -> Path:
    rel_path_s = repr(str(rel_path))

    # Must be a relative path
    if rel_path.is_absolute():
        raise ArgParserError(f"Environment file {rel_path_s} must be a relative path.")

    # Resolve full absolute path under project root
    current_path = current_dir.resolve()
    abs_path = (current_path / rel_path).resolve()

    # Prevent escaping the project directory (e.g. ../ tricks)
    try:
        abs_path.relative_to(current_path)
    except ValueError as e:
        raise ArgParserError(f"{rel_path_s} is outside the project directory.") from e

    out_rel_path = abs_path.relative_to(current_path)

    # File must exist
    if not abs_path.is_file():
        raise ArgParserError(f"Environment file {rel_path_s} does not exist.")

    # Case 1: `.env` in project root
    if abs_path == current_path / ".env":
        return out_rel_path

    # Case 2: in `envs/`, named `.env.*`
    parts = out_rel_path.parts
    if len(parts) == ENVS_SPLIT_LENGTH and parts[0] == "envs":
        name = parts[1]
        if name.startswith(".env.") and name != ".env.":
            return out_rel_path

    raise ArgParserError(
        f"Environment file {rel_path_s} is not allowed. "
        f"Allowed locations are:\n"
        f" - .env in the project root directory\n"
        f" - envs/.env.* files in the envs/ directory"
    )


class CmdArgs(TypedDict):
    centralized: bool
    env_files: list[Path]


def parse_args(args: list[str], main_project_dir: Path) -> CmdArgs:
    if not args:
        raise ArgParserError(
            "Usage: [--centralized] --env-file <env_file> [--env-file <env_file>] ..."
        )

    it = iter(args)
    centralized = False
    env_files: list[Path] = []

    # Support --centralized only as the first token
    first = next(it, None)
    if first == "--centralized":
        centralized = True
        first = next(it, None)

    # Process the rest (including the token we already fetched into `first`)
    def pull_next() -> str:
        nxt = next(it, None)
        if nxt is None:
            raise ArgParserError("--env-file must be followed by a file path.")
        return nxt

    token = first
    while token is not None:
        if token == "--env-file":  # noqa: S105
            env_file_s = pull_next()
            if not env_file_s:
                raise ArgParserError(
                    "--env-file must be followed by a non-empty file path."
                )
            normalized_rel = validate_env_file(Path(env_file_s), main_project_dir)
            env_files.append(normalized_rel)
        else:
            raise ArgParserError(f"Unknown argument {repr(token)}.")
        token = next(it, None)

    if not env_files:
        raise ArgParserError("You must provide at least one --env-file.")

    return CmdArgs(centralized=centralized, env_files=env_files)


def read_envs(env_file: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    with env_file.open() as f:
        env_file_content = f.readlines()
    for line in env_file_content:
        line_s = line.strip()
        if not line_s:
            continue

        if line_s.startswith("#"):
            continue

        if "=" not in line_s:
            logger.info(
                "Warning: Invalid line in environment file %s: %s",
                repr(str(env_file)),
                line_s,
            )
            continue

        env_name, env_value = line_s.split("=", maxsplit=1)

        # add or overwrite
        env[env_name] = env_value
    return env


class ReplacementRule(TypedDict):
    replace_with: Any
    exact_match: bool


T = TypeVar("T")


def deep_replace(
    obj: dict[str, Any], rules: dict[str, ReplacementRule]
) -> dict[str, Any]:
    """
    Recursively walk `obj` and perform replacements according to `rules`.

    Rules format:
        {
            "<needle>": {"replace_with": <any>, "exact_match": <bool>}
        }

    - If exact_match is True and a leaf string equals <needle>, the leaf is replaced
      with the value from `replace_with` (returned as-is; can change type).
    - If exact_match is False and a leaf string contains <needle>, a substring
      replacement is performed with str(replace_with).
    - Dict keys are left untouched (only values are transformed).
    - Supports dicts, lists, tuples, and sets; everything else is treated as a leaf.
    """
    # Split rules up-front for speed and determinism
    exact_rules: dict[str, Any] = {}
    substring_rules: list[tuple[str, str]] = []

    for needle, spec in rules.items():
        if spec.get("exact_match", False):
            exact_rules[needle] = spec["replace_with"]
        else:
            substring_rules.append((needle, str(spec["replace_with"])))

    # To reduce cascading on overlapping needles, apply longer needles first
    substring_rules.sort(key=lambda kv: len(kv[0]), reverse=True)

    def _transform_leaf(x: T) -> T:
        if not isinstance(x, str):
            return x
        # exact match first
        if x in exact_rules:
            return exact_rules[x]
        # then substring replacements
        y = x
        for needle, repl in substring_rules:
            if needle and needle in y:
                y = y.replace(needle, repl)
        return y

    def _walk(o: T) -> T:
        if isinstance(o, dict):
            return {k: _walk(v) for k, v in o.items()}  # values only  # type: ignore
        if isinstance(o, list):
            return [_walk(i) for i in o]  # type: ignore
        if isinstance(o, tuple):
            return tuple(_walk(i) for i in o)  # type: ignore
        if isinstance(o, set):
            return {_walk(i) for i in o}  # type: ignore
        return _transform_leaf(o)

    return _walk(obj)


class MySafeDumper(yaml.SafeDumper):
    def increase_indent(
        self,
        flow: bool = False,  # noqa: FBT001,FBT002
        indentless: bool = False,  # noqa: FBT001,FBT002,ARG002
    ) -> None:
        return super().increase_indent(flow=flow, indentless=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    current_dir = Path().cwd()
    main_project_dir = Path(__file__).resolve().parents[1]
    if current_dir != main_project_dir:
        raise RuntimeError(
            f"This script must be run from the main project directory.\n"
            f"Main project directory: {str(main_project_dir)!r}\n"
            f"Current directory: {str(current_dir)!r}"
        )

    args = sys.argv[1:]

    # args = (
    #     "--env-file envs/.env.base "
    #     "--env-file envs/.env.normal --env-file envs/.env.prod"
    # ).split()
    # args = ["--centralized"] + args

    cmd_args = parse_args(args, main_project_dir)

    centralized = cmd_args["centralized"]
    env_files = cmd_args["env_files"]

    env: dict[str, str] = {}
    for env_file in env_files:
        env.update(read_envs(env_file))

    tz = env["TZ"]

    config_dir = current_dir / "configs" / "compose"

    with (config_dir / "gpu_map.yml").open() as f:
        gpu_map = yaml.safe_load(f)
    if not isinstance(gpu_map, dict):
        raise ValueError("Invalid GPU map format")

    server_gpu = gpu_map["server"]  # type: ignore
    if not isinstance(server_gpu, int):
        raise ValueError("Invalid server GPU ID in GPU map")

    clients_gpus = gpu_map["clients"]  # type: ignore
    if not isinstance(clients_gpus, list) or not all(
        isinstance(gpu, int)
        for gpu in clients_gpus  # type: ignore
    ):
        raise ValueError("Invalid clients GPU IDs in GPU map")
    clients_gpus = cast("list[int]", clients_gpus)

    with (config_dir / "network_map.yml").open() as f:
        network_map = yaml.safe_load(f)
    if not isinstance(network_map, dict):
        raise ValueError("Invalid network map format")

    server_address = network_map["server"]  # type: ignore
    try:
        if not isinstance(server_address, str):
            raise TypeError("Invalid server IP in network map")
    except ValueError as e:
        raise ValueError("Invalid server IP in network map") from e

    clients_addresses = network_map["clients"]  # type: ignore
    if not isinstance(clients_addresses, list) or not all(
        isinstance(ip, str)
        for ip in clients_addresses  # type: ignore
    ):
        raise ValueError("Invalid clients IPs in network map")
    clients_addresses = cast("list[str]", clients_addresses)

    clients = len(clients_gpus) if centralized else len(clients_addresses)

    flserver_portrange = env["FLSERVER_PORTRANGE"]

    replacement_rules_general: dict[str, ReplacementRule] = {
        "{{SERVER_ADDRESS}}": ReplacementRule(
            replace_with=server_address, exact_match=True
        ),
        "{{CLIENTS_ADDRESSES}}": ReplacementRule(
            replace_with=",".join(clients_addresses),
            exact_match=True,
        ),
        "{{SERVER_GPU}}": ReplacementRule(
            replace_with=str(server_gpu),
            exact_match=True,
        ),
        "{{NUM_CLIENTS}}": ReplacementRule(
            replace_with=str(clients),
            exact_match=True,
        ),
        "{{CENTRALIZED}}": ReplacementRule(
            replace_with="-centralized" if centralized else "",
            exact_match=False,
        ),
        "{{API_PORT}}": ReplacementRule(
            replace_with="${API_PORT}"
            if not centralized
            else "${CENTRALIZED_API_PORT}",
            exact_match=False,
        ),
        "{{FLSERVER_PORTRANGE}}": ReplacementRule(
            replace_with=flserver_portrange,
            exact_match=False,
        ),
    }

    with (
        current_dir / "scripts" / "docker" / "docker-compose-template.yml"
    ).open() as f:
        compose = yaml.safe_load(f)

    if not isinstance(compose, dict):
        raise RuntimeError("Invalid docker-compose format.")
    compose = cast("dict[str, Any]", compose)

    # Before replacing the whole content, we need to add the clients logs as mounting
    # volumes to the api service
    api_volumes = cast("list[str]", compose["services"]["api"]["volumes"])
    template_client_volume = (
        "../logs{{CENTRALIZED}}/flclient-{{CLIENT_ID}}:"
        "/app/logs/flclient-{{CLIENT_ID}}:ro"
    )

    if template_client_volume not in api_volumes:
        raise RuntimeError("Template client volume not found in api service volumes.")
    api_volumes.remove(template_client_volume)

    for client in range(clients):
        client_volume = template_client_volume.replace("{{CLIENT_ID}}", str(client))
        api_volumes.append(client_volume)

    # Run the replacements
    compose = deep_replace(compose, replacement_rules_general)

    output_dir = current_dir / ("docker" + ("-centralized" if centralized else ""))
    if not output_dir.is_dir():
        raise RuntimeError(
            f"Output directory {repr(str(output_dir))} does not exist. "
            f"Please create it first."
        )

    logs_dir = current_dir / ("logs" + ("-centralized" if centralized else ""))
    if not logs_dir.is_dir():
        logger.info("Creating logs directory: %s", repr(str(logs_dir)))
        logs_dir.mkdir(parents=True, exist_ok=True)

    models_dir = current_dir / ("models" + ("-centralized" if centralized else ""))
    if not models_dir.is_dir():
        logger.info("Creating model artifact directory: %s", repr(str(models_dir)))
        models_dir.mkdir(parents=True, exist_ok=True)

    datasplit_dir = (
        current_dir / "fldata" / ("datasplit" + ("-centralized" if centralized else ""))
    )
    if not datasplit_dir.is_dir():
        logger.info("Creating datasplit directory: %s", repr(str(datasplit_dir)))
        datasplit_dir.mkdir(parents=True, exist_ok=True)

    # Orchestrator state and exported results are bind-mounted (no suffix)
    for data_dir in [current_dir / "storage", current_dir / "results"]:
        if not data_dir.is_dir():
            logger.info("Creating data directory: %s", repr(str(data_dir)))
            data_dir.mkdir(parents=True, exist_ok=True)

    for service in ["api", "orchestrator", "flserver"]:
        service_logs_dir = logs_dir / service
        if not service_logs_dir.is_dir():
            logger.info(
                "Creating service logs directory: %s", repr(str(service_logs_dir))
            )
            service_logs_dir.mkdir(parents=True, exist_ok=True)
    for client in range(clients):
        client_logs_dir = logs_dir / f"flclient-{client}"
        if not client_logs_dir.is_dir():
            logger.info(
                "Creating client logs directory: %s", repr(str(client_logs_dir))
            )
            client_logs_dir.mkdir(parents=True, exist_ok=True)

    # Containers run as UID/GID from the env files: a bind-mounted directory
    # they cannot write makes status/model/result persistence fail at runtime.
    container_uid = int(env["UID"])
    container_gid = int(env["GID"])
    writable_dirs = [
        current_dir / "storage",
        current_dir / "results",
        models_dir,
        datasplit_dir,
        *(path for path in [logs_dir, *logs_dir.iterdir()] if path.is_dir()),
    ]
    not_writable = [
        path
        for path in writable_dirs
        if not (
            path.stat().st_mode & 0o002
            or (path.stat().st_uid == container_uid and path.stat().st_mode & 0o200)
            or (path.stat().st_gid == container_gid and path.stat().st_mode & 0o020)
        )
    ]
    if not_writable:
        rel_paths = " ".join(str(path.relative_to(current_dir)) for path in not_writable)
        logger.warning(
            "Directories not writable by container UID/GID %d:%d: %s\n"
            "Fix with: sudo chown -R %d:%d %s",
            container_uid, container_gid, rel_paths,
            container_uid, container_gid, rel_paths,
        )

    if centralized:
        if "${API_PORT}:${API_PORT}" in compose["services"]["api"]["ports"]:
            compose["services"]["api"]["ports"].remove("${API_PORT}:${API_PORT}")
        for key in ["SERVER_ADDRESS", "CLIENTS_ADDRESSES"]:
            del compose["services"]["orchestrator"]["environment"][key]
        del compose["services"]["flserver"]["ports"]
        for service in compose["services"].values():
            service["environment"]["CENTRALIZED"] = "1"
    else:
        if (
            "${CENTRALIZED_API_PORT}:${CENTRALIZED_API_PORT}"
            in compose["services"]["api"]["ports"]
        ):
            compose["services"]["api"]["ports"].remove(
                "${CENTRALIZED_API_PORT}:${CENTRALIZED_API_PORT}"
            )
        del compose["networks"]
        for service in compose["services"].values():
            if "networks" in service:
                del service["networks"]

    fl_client_base = compose["services"]["flclient"]
    if not isinstance(fl_client_base, dict):
        raise RuntimeError("Invalid flclient service format.")

    fl_client_base = cast("dict[str, Any]", fl_client_base)

    if centralized:
        for i in range(clients):
            service_name = f"flclient-{i}"
            fl_client_service = deepcopy(fl_client_base)

            replacement_rules_client: dict[str, ReplacementRule] = {
                "{{CLIENT_ID}}": ReplacementRule(replace_with=i, exact_match=False),
                "{{CLIENT_GPU}}": ReplacementRule(
                    replace_with=str(clients_gpus[i]), exact_match=True
                ),
            }

            fl_client_service = deep_replace(
                fl_client_service, replacement_rules_client
            )
            del fl_client_service["ports"]
            del fl_client_service["ipc"]
            del fl_client_service["runtime"]
            del fl_client_service["cap_add"]
            compose["services"][service_name] = fl_client_service
    else:
        fl_client_service = deepcopy(fl_client_base)

        for key in ["deploy", "depends_on"]:
            del fl_client_service[key]

        del fl_client_service["environment"]["CUDA_VISIBLE_DEVICES"]
        del fl_client_service["build"]["dockerfile"]

        replacement_rules_client: dict[str, ReplacementRule] = {
            "{{CLIENT_ID}}": ReplacementRule(
                replace_with="${CLIENT_ID}", exact_match=False
            ),
        }
        fl_client_service = deep_replace(fl_client_service, replacement_rules_client)
        fl_client_service["environment"]["TZ"] = tz
        fl_client_service["volumes"].append("/run/jtop.sock:/run/jtop.sock")

        edge_compose_path = output_dir / "docker-compose-edge.yml"
        with edge_compose_path.open("w") as f:
            logger.info(
                "Writing edge docker-compose to %s ...", repr(str(edge_compose_path))
            )
            yaml.dump(
                {"services": {"flclient": fl_client_service}},
                f,
                sort_keys=False,
                default_flow_style=False,
                indent=2,
                Dumper=MySafeDumper,
            )

    del compose["services"]["flclient"]

    for data in compose["services"].values():
        data["environment"]["TZ"] = tz

    compose_path = output_dir / (
        "docker-compose" + ("-centralized" if centralized else "") + ".yml"
    )

    with compose_path.open("w") as f:
        logger.info("Writing docker-compose to: %s ...", repr(str(compose_path)))
        yaml.dump(
            compose, f, sort_keys=False, default_flow_style=False, Dumper=MySafeDumper
        )

    logger.info("Done.")
