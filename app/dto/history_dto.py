from bson.json_util import default
from flask_restx import Namespace, fields
import datetime
from app.dto.base_dto import base
from app.utils.auth_parser_util import get_auth_required_parser, get_auth_not_required_parser


class HistoryDto:
    api = Namespace('History', description="History")
    __base = api.model("base", base)

    """request"""
    get_all_request = api.parser()

    get_all_request.add_argument(
        "skip", type=int, location="json", required=True)
    get_all_request.add_argument(
        "limit", type=int, location="json", required=True)
    get_all_request.add_argument(
        "Authorization", type=str, location='headers', required=False)

    get_by_duration_request = api.parser()

    get_by_duration_request.add_argument(
        "skip", type=int, location="json", required=True)
    get_by_duration_request.add_argument(
        "limit", type=int, location="json", required=True)
    get_by_duration_request.add_argument(
        "start_date", type=str, location="json", required=True, default='2021-12-05')
    get_by_duration_request.add_argument(
        "end_date", type=str, location="json", required=True, default='2021-12-05')
    get_by_duration_request.add_argument(
        "Authorization", type=str, location='headers', required=False)

    rating_request = api.parser()
    rating_request.add_argument(
        "_id", type=str, location="json", required=True)
    rating_request.add_argument(
        "rate", type=int, location="json", required=True)
    rating_request.add_argument(
        "Authorization", type=str, location='headers', required=False)

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
