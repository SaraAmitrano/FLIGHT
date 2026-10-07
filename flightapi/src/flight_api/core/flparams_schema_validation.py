from collections.abc import Iterable
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError
from pydantic import BaseModel


class ValidationErrorBase(BaseModel):
    message: str
    key: str
    validator: Any
    validator_value: Any
    instance: Any


def extract_error_info(errors: Iterable[ValidationError]) -> list[ValidationErrorBase]:
    results: list[ValidationErrorBase] = []
    for error in errors:
        path_parts: list[str] = []
        for part in error.path:
            if isinstance(part, int):
                # Index in array — append as [index] to the last path part
                if path_parts:
                    path_parts[-1] = f"{path_parts[-1]}[{part}]"
                else:
                    path_parts.append(f"[{part}]")
            else:
                path_parts.append(str(part))
        full_path = ".".join(path_parts)

        results.append(
            ValidationErrorBase(
                message=error.message,
                key=full_path,
                validator=error.validator,
                validator_value=error.validator_value,
                instance=error.instance,
            )
        )
    return results


def validate_flparams(
    params: dict[str, Any], schema_validator: Draft7Validator
) -> list[ValidationError]:
    return sorted(
        schema_validator.iter_errors(params),  # type: ignore
        key=lambda e: e.path,
    )
