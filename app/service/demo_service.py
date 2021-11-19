from sqlalchemy import func
import datetime
import random
import re
import string
from operator import or_, and_
from app import db, app
from app.model.user_model import User
from app.model.black_list_token import BlackListToken
from app.utils.api_response import response_object
import app.utils.response_message as message
from flask_jwt_extended import create_access_token, get_jwt
import detect


def login(args):
    detect.run(source='C:/Users/Thanh/Documents/Downloads/Server/Fresh-Fruit-Server/images/images (2).jpg',save_txt=True,weights='C:/Users/Thanh/Documents/Downloads/Server/Fresh-Fruit-Server/weights/best.pt')
    user = User.query.filter(func.lower(User.email) == func.lower(args['email'])).first()
    if not user:
        return response_object(status=False, message=message.EMAIL_NOT_EXISTS), 401
    if not user.is_active:
        return response_object(status=False, message=message.ACCOUNT_IS_NOT_ACTIVATED), 401
    if not user.verify_password(args['password']):
        return response_object(status=False, message=message.PASSWORD_WRONG), 401
    auth_token = create_access_token(identity=user.to_payload(), expires_delta=app.config['TOKEN_EXPIRED_TIME'])
    if auth_token:
        data = user.to_json()
        data['token'] = auth_token
        return response_object(data=data), 200
    return response_object(status=False, message=message.UNAUTHORIZED_401), 401


def test(args):
    pass


def create_user(args):
    if User.query.filter(func.lower(User.email) == func.lower(args['email'])).first():
        return response_object(status=False, message=message.CONFLICT_409), 409

    user = User(
        email=args['email'],
        password=args['password'],
        first_name=args['first_name'],
        last_name=args['last_name'],
        is_active=True,
        is_admin=args['is_admin']
    )
    db.session.add(user)
    db.session.commit()

    return response_object(), 201


def logout(token):
    jti = get_jwt()["jti"]
    if token:
        black_list = BlackListToken(token=jti)
        db.session.add(black_list)
        db.session.commit()

    return response_object(), 200
