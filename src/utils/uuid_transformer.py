import base64
import uuid


def convert_to_valid_uuid(uuid_str: str) -> uuid.UUID:
    """
        Converts a UUID string to a valid uuid object.

        @param uuid_str: uuid string

        @returns uuid object
    """
    return uuid.UUID(bytes=base64.b64decode(uuid_str))
