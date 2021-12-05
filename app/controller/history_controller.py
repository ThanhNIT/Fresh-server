from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Resource
from pymongo.message import delete

from app import response_object, db
from app.dto.history_dto import HistoryDto
from app.model.user_model import User
from app.service import history_service
from app.utils.auth_parser_util import get_auth_required_parser
from app.utils.jwt_util import admin_required, user_required
from bson import ObjectId
from app.utils.encoder import toJson
api = HistoryDto.api

_get_all_request = HistoryDto.get_all_request
_get_by_duration_request = HistoryDto.get_by_duration_request
_rating_request = HistoryDto.rating_request


@api.route('/')
class Get_All(Resource):
    @api.doc('get all')
    @jwt_required()
    @api.expect(_get_all_request, validate=True)
    def post(self):  # post put delete get
        args = _get_all_request.parse_args()
        user_id = get_jwt_identity()['user_id']
        args['user_id'] = user_id
        print(args)
        return history_service.get_all_by_user(args)


@api.route('/duration')
class Get_by_duration(Resource):
    @api.doc('get by duration')
    @jwt_required()
    @api.expect(_get_by_duration_request, validate=True)
    def post(self):  # post put delete get
        args = _get_by_duration_request.parse_args()
        user_id = get_jwt_identity()['user_id']
        args['user_id'] = user_id
        return history_service.get_with_duration(args)


@api.route('/rating')
class Rate_Detection(Resource):
    @api.doc('rate detection')
    @jwt_required()
    @api.expect(_rating_request, validate=True)
    def post(self):  # post put delete get
        args = _rating_request.parse_args()
        user_id = get_jwt_identity()['user_id']
        args['user_id'] = user_id
        return history_service.rating(args)


@api.route('/get-by-id/<id>')
class Get_By_id(Resource):
    @api.doc('get by id')
    @api.expect(get_auth_required_parser(api), validate=True)
    @jwt_required()
    def get(self, id):  # post put delete get
        user_id = get_jwt_identity()['user_id']

        args = {
            'user_id': user_id,
            '_id': id
        }
        return history_service.get_by_id(args)


@api.route('/<id>')
class Delete_By_id(Resource):
    @api.doc('delete by id')
    @api.expect(get_auth_required_parser(api), validate=True)
    @jwt_required()
    def delete(self, id):  # post put delete get
        user_id = get_jwt_identity()['user_id']

        args = {
            'user_id': user_id,
            '_id': id
        }
        return history_service.delete_history(args)

# @api.route('/change-password')
# class Login(Resource):
#     @api.doc('change password')
#     @api.expect(_change_password_request, validate=True)
#     def post(self):  # post put delete get
#         args = _change_password_request.parse_args()
#         return user_service.change_password(args)

# @api.route('/reset-password')
# class Login(Resource):
#     @api.doc('reset password')
#     @api.expect(_reset_password_request, validate=True)
#     def post(self):  # post put delete get
#         args = _reset_password_request.parse_args()
#         return user_service.reset_password(args)


# @api.route('/create')
# class CreateUser(Resource):
#     @api.doc('create')
#     @api.expect(_create_user_request, validate=True)
#     def post(self):
#         args = _create_user_request.parse_args()
#         return user_service.create_user(args)


# @api.route('/logout')
# class Logout(Resource):
#     @api.doc('logout')
#     @api.expect(_logout_request, validate=True)
#     @jwt_required()
#     def post(self):
#         auth_token = request.headers['Authorization'].split(" ")[1]
#         return user_service.logout(auth_token)


# @api.route('/test-admin')
# class TestAdmin(Resource):
#     @api.doc('test-admin')
#     @api.expect(get_auth_required_parser(api), validate=True)
#     @jwt_required()
#     @admin_required()
#     def post(self):
#         user_id = get_jwt_identity()['user_id']
#         user = db.users.find_one({'_id':ObjectId(user_id)})
#         data = toJson(user)
#         return response_object(data=data), 200


# @api.route('/test-user')
# class TestUser(Resource):
#     @api.doc('test-user')
#     @api.expect(get_auth_required_parser(api), validate=True)
#     @jwt_required()
#     @user_required()
#     def post(self):
#         user_id = get_jwt_identity()['user_id']
#         user = User.query.get(user_id)
#         data = user.to_json()
#         return response_object(data=data), 200
