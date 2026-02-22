from datetime import datetime

def generate_datetime_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"