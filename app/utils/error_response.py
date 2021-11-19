from app import app
from app.util.api_response import response_object


@app.errorhandler(400)
def handle_bad_request(e):
    return response_object(status=False, code=400), 400
