from flask_restx import Namespace, fields

from app.dto.base_dto import base
from app.utils.auth_parser_util import get_auth_required_parser, get_auth_not_required_parser


class DemoDto:
    api = Namespace('User', description="User")
    __base = api.model("base", base)

    """request"""
    login_request = api.parser()
    login_request.add_argument("email", type=str, location="json", required=True)
    login_request.add_argument("password", type=str, location="json", required=True)

    create_user_request = api.parser()
    create_user_request.add_argument('email', type=str, location='json', required=True)
    create_user_request.add_argument('password', type=str, location='json', required=True)
    create_user_request.add_argument('first_name', type=str, location='json', required=True)
    create_user_request.add_argument('last_name', type=str, location='json', required=True)
    create_user_request.add_argument('is_admin', type=bool, location='json', required=True)

    logout_request = get_auth_required_parser(api)
    logout_request.add_argument("Authorization", type=str, location='headers', required=False)

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
