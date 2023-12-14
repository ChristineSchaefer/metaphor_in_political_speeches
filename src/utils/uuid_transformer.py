import base64
import uuid


def convert_to_valid_uuid(uuid_str: str) -> uuid.UUID:
    return uuid.UUID(bytes=base64.b64decode(uuid_str))
