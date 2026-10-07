import json
import zlib
from typing import Any


def compress_json(data: dict[str, Any]) -> bytes:
    """
    Compress JSON data using zlib.

    Args:
        data (dict[str]): The JSON data to compress.

    Returns:
        bytes: The compressed JSON data.

    """
    json_data = json.dumps(data, separators=(",", ":"))
    return zlib.compress(json_data.encode("utf-8"))


def decompress_json(data: bytes) -> dict[str, Any]:
    """
    Decompress JSON data using zlib.

    Args:
        data (bytes): The compressed JSON data to decompress.

    Returns:
        dict[str]: The decompressed JSON data.

    """
    decompressed_data = zlib.decompress(data)
    return json.loads(decompressed_data.decode("utf-8"))
