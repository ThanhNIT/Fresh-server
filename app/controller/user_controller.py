from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Resource

from app import response_object
from app.dto.demo_dto import DemoDto
from app.model.user_model import User
from app.service import demo_service
from app.utils.auth_parser_util import get_auth_required_parser
from app.utils.jwt_util import admin_required, user_required

api = DemoDto.api

_login_request = DemoDto.login_request
_create_user_request = DemoDto.create_user_request
_logout_request = DemoDto.logout_request


@api.route('/login')
class Login(Resource):
    @api.doc('login')
    @api.expect(_login_request, validate=True)
    def post(self):  # post put delete get
        args = _login_request.parse_args()
        return demo_service.login(args)


@api.route('/create')
class CreateUser(Resource):
    @api.doc('create')
    @api.expect(_create_user_request, validate=True)
    def post(self):
        args = _create_user_request.parse_args()
        return demo_service.create_user(args)


@api.route('/logout')
class Logout(Resource):
    @api.doc('logout')
    @api.expect(_logout_request, validate=True)
    @jwt_required()
    def post(self):
        auth_token = request.headers['Authorization'].split(" ")[1]
        return demo_service.logout(auth_token)


@api.route('/test-admin')
class TestAdmin(Resource):
    @api.doc('test-admin')
    @api.expect(get_auth_required_parser(api), validate=True)
    @jwt_required()
    @admin_required()
    def post(self):
        print('aaa')
        user_id = get_jwt_identity()['user_id']
        print(user_id)
        user = User.query.get(user_id)
        data = user.to_json()
        return response_object(data=data), 200


@api.route('/test-user')
class TestUser(Resource):
    @api.doc('test-user')
    @api.expect(get_auth_required_parser(api), validate=True)
    @jwt_required()
    @user_required()
    def post(self):
        user_id = get_jwt_identity()['user_id']
        user = User.query.get(user_id)
        data = user.to_json()
        return response_object(data=data), 200
