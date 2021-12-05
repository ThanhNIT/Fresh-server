# imports for PyJWT authentication
from functools import wraps

from flask import request, jsonify
from flask_jwt_extended import get_jwt_identity

from app.model.user_model import User
from app.utils.api_response import response_object
from app.utils.response_message import UNAUTHORIZED_401, NOT_FOUND_404
from app import app, db
from app.utils.encoder import JSONEncoder
import jwt
import json
from bson import ObjectId
from bson.json_util import dumps, loads

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header

        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        # return 401 if token is not passed
        if not token:
            return jsonify({'message': 'Token is missing !!'}), 401
        payload = decode_auth_token(token)

        try:
            user = db.users.find_one({'_id':payload['user_id']})
            user = json.loads(JSONEncoder().encode(user))
            print(user)
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

            # try:
            print(user_id)
            user = db.users.find_one({'_id':ObjectId(user_id)})
            print(json.loads(JSONEncoder().encode(user)))
            if not user:
                return response_object(status=False, message=UNAUTHORIZED_401), 401
            user = json.loads(JSONEncoder().encode(user))
            if user['is_admin']:
                return function(*args, **kwargs)
            else:
                return response_object(status=False, message=UNAUTHORIZED_401), 401
            # except:
            #     return response_object(status=False, message=NOT_FOUND_404), 404

        return wrapper

    return decorator

def user_required():
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            identity = get_jwt_identity()
            user_id = identity['user_id']

            try:
                user = db.users.find_one({'_id':payload['user_id']})
                if not user:
                    return response_object(status=False, message=UNAUTHORIZED_401), 401
                user = json.loads(JSONEncoder().encode(user))
                if not user['is_admin']:
                    return function(*args, **kwargs)
                else:
                    return response_object(status=False, message=UNAUTHORIZED_401), 401
            except:
                return response_object(status=False, message=NOT_FOUND_404), 404

        return wrapper

    return decorator


def decode_auth_token(auth_token):
    try:
        payload = jwt.decode(auth_token, app.config.get('SECRET_KEY'), algorithms=["HS256"])

        # token = BlackListToken.query.filter(BlackListToken.token == auth_token).first()

        # if not token:
        #     return 'Token blacklisted. Please log in again.'
        # else:
        tz_London = pytz.timezone('Asia/Saigon')
        if (token.created_date + app.config.get("TOKEN_EXPIRED_TIME")) < datetime.now():
            return 'Signature expired. Please log in again.'
        return payload
    except jwt.ExpiredSignatureError:
        return 'Signature expired. Please log in again.'
    except jwt.InvalidTokenError:
        return 'Invalid token. Please log in again.'