import base64
import uuid


def convert_to_valid_uuid(uuid_str: str) -> uuid.UUID:
    """
        Converts a UUID string to a valid uuid object.

        @param uuid_str: uuid string

        @returns uuid object
    """
    return uuid.UUID(bytes=base64.b64decode(uuid_str))


def oid_to_uuid(oid) -> uuid.UUID:
    # Convert ObjectId to bytes
    oid_bytes = oid.binary

    # Append 4 bytes for timestamp and 8 bytes for ObjectId
    padded_bytes = oid_bytes.ljust(16, b'\x00')

    # Convert bytes to UUID
    uuid_result = uuid.UUID(bytes=padded_bytes)

    return uuid_result


def bson_to_uuid(bson_object: dict) -> uuid.UUID:
    # Extract base64-encoded binary string and subType
    base64_str = bson_object["$binary"]["base64"]
    sub_type = bson_object["$binary"]["subType"]

    # Decode base64 string to bytes
    decoded_bytes = base64.b64decode(base64_str)

    # Check if subType is "04" (ObjectId)
    if sub_type == "04":
        # Pad the bytes to make it 16 bytes long
        padded_bytes = decoded_bytes.ljust(16, b'\x00')

        # Convert bytes to UUID
        uuid_result = uuid.UUID(bytes=padded_bytes)

        return uuid_result
    else:
        raise ValueError("Invalid subType. Expected '04' for ObjectId.")