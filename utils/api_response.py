def format_response(statusCode: int, message: str, payload) -> dict:
    return {"message": message, "statusCode": statusCode, "payload": payload}
