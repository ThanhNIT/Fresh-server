from flask_restx import Namespace, fields

from app.dto.base_dto import base
from werkzeug.datastructures import FileStorage
from app.utils.auth_parser_util import get_auth_required_parser, get_auth_not_required_parser


class FreshDto:
    api = Namespace('Fresh', description="Fresh")
    __base = api.model("base", base)

    """request"""
    fresh_request = api.parser()
    fresh_request.add_argument("file", type=FileStorage, location="files", required=True)

    get_file_request = api.parser()
    get_file_request.add_argument('filename', type=str, location='json', required=True)

    """response"""
    _login_data = api.inherit('login_data', {
        'id': fields.Integer(required=False),
        'name': fields.String(required=False),
        'address': fields.String(required=False),
        'email': fields.String(required=False)
    })

    login_response = api.inherit('login_response', base, {
        'data': fields.Nested(_login_data)
    })

    message_response = api.inherit('message_response', base, {
        'data': fields.String,
    })
