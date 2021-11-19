# imports for PyJWT authentication
from functools import wraps

from flask import request, jsonify
from flask_jwt_extended import get_jwt_identity

from app.model.user_model import User
from app.utils.api_response import response_object
from app.utils.response_message import UNAUTHORIZED_401, NOT_FOUND_404


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header

        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        # return 401 if token is not passed
        print('aaaaaaaaaaaaaa')
        if not token:
            return jsonify({'message': 'Token is missing !!'}), 401
        payload = User.decode_auth_token(token)

        try:
            user = User.query.filter(User.id == payload['user_id']).first()

        except:
            return response_object(status=False, message=UNAUTHORIZED_401), 401
        return f(*args, user, **kwargs)

    return decorated


def admin_required():
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            identity = get_jwt_identity()
            user_id = identity['user_id']

            try:
                user = User.query.filter(User.id == user_id).first()
                if user.is_admin:
                    return function(*args, **kwargs)
                else:
                    return response_object(status=False, message=UNAUTHORIZED_401), 401
            except:
                return response_object(status=False, message=NOT_FOUND_404), 404

        return wrapper

    return decorator

def user_required():
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            identity = get_jwt_identity()
            user_id = identity['user_id']

            try:
                user = User.query.filter(User.id == user_id).first()
                if not user.is_admin:
                    return function(*args, **kwargs)
                else:
                    return response_object(status=False, message=UNAUTHORIZED_401), 401
            except:
                return response_object(status=False, message=NOT_FOUND_404), 404

        return wrapper

    return decorator