import datetime
import uuid


def generate_request_id() -> str:
    now = datetime.datetime.now(datetime.UTC)
    dt_str = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    uid_str = str(uuid.uuid4())[:8]
    return f"{dt_str}-{uid_str}"
