import importlib.resources
import json
from importlib.resources.abc import Traversable
from typing import Any

from jsonschema import Draft7Validator
from referencing import Registry, Resource


def do_load_all_schemas(
    schema_dir: Traversable,
) -> tuple[dict[str, dict[str, Any]], Registry]:
    """Load all JSON schemas from the schema directory."""
    schemas: dict[str, dict[str, Any]] = {}
    registry = Registry()  # type: ignore

    for file in schema_dir.iterdir():
        if file.name.endswith(".json"):
            schema_name = file.name
            schema_path = f"./{file.name}"  # Get absolute URI
            with file.open() as f:
                schema_content = json.load(f)
            schemas[schema_name] = schema_content
            registry = registry.with_resource(  # type: ignore
                schema_path, Resource.from_contents(schema_content)
            )

    return schemas, registry  # type: ignore


def get_schema_dir(model_name: str) -> Traversable:
    """Get the path to the schema directory."""
    return importlib.resources.files("flight_api").joinpath(
        "resources", "schemas", model_name
    )


def load_all_schemas(model_name: str) -> tuple[dict[str, dict[str, Any]], Registry]:
    """Load all JSON schemas from the schema directory."""
    schema_dir = get_schema_dir(model_name)
    return do_load_all_schemas(schema_dir)


def get_schema_validator(
    schemas: dict[str, dict[str, Any]], registry: Registry
) -> Draft7Validator:
    """Get the main schema validator."""
    main_schema = schemas.get("main-schema.json")
    if main_schema is None:
        raise ValueError("Main schema not found.")
    return Draft7Validator(main_schema, registry=registry)
