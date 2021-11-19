from datetime import datetime

from . import response_message


def response_object(status=True, message=response_message.SUCCESS, data=[], pagination=None):
    if not pagination:
        return {
            'status': status,
            'message': message,
            'data': data,
        }
    else:
        return {
            'status': status,
            'message': message,
            'data': data,
            'pagination': pagination
        }


def json_serial(obj):
    if isinstance(obj, datetime):
        # return obj.isoformat()
        return obj.__str__()


def date_to_json(obj):
    return obj.strftime("%Y-%m-%d %H:%M:%S")
