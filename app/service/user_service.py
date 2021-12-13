from sqlalchemy import func
import datetime
import random
import re
import string
from operator import or_, and_
from app import db, app, bcrypt
from app.model.user_model import User
# from app.model.black_list_token import BlackListToken
from app.utils.api_response import response_object
import app.utils.response_message as message
from flask_jwt_extended import create_access_token, get_jwt
import detect
from app.utils.encoder import UserEncoder, JSONEncoder
import json
from bson import ObjectId
from bson.json_util import dumps, loads
from flask import jsonify
from app.service import mail_service



def login(args):
    user  = db.users.find_one({"email":args['email'].lower()})
    if not user:
        return response_object(status=False, message=message.EMAIL_NOT_EXISTS), 401
    user = json.loads(JSONEncoder().encode(user))

    if not user['is_active']:
        return response_object(status=False, message=message.ACCOUNT_IS_NOT_ACTIVATED), 401
    if not bcrypt.check_password_hash(user['password'], args['password']):
        return response_object(status=False, message=message.PASSWORD_WRONG), 401
    auth_token = create_access_token(identity=to_payload(user), expires_delta=app.config['TOKEN_EXPIRED_TIME'])
    if auth_token:
        data = user
        data['token'] = auth_token
        del data['password']
        return response_object(data=data), 200
    return response_object(status=False, message=message.UNAUTHORIZED_401), 401

def change_password(args):
    user  = db.users.find_one({"email":args['email'].lower()})
    if not user:
        return response_object(status=False, message=message.EMAIL_NOT_EXISTS), 401
    user = json.loads(JSONEncoder().encode(user))

    if not user['is_active']:
        return response_object(status=False, message=message.ACCOUNT_IS_NOT_ACTIVATED), 401
    if not bcrypt.check_password_hash(user['password'], args['password']):
        return response_object(status=False, message=message.PASSWORD_WRONG), 200
    
    db.users.update_one({'_id': ObjectId(user['_id'])}, {"$set": {'password': bcrypt.generate_password_hash(args['new_password']).decode('utf-8')}})

    return response_object(status=True, message=message.SUCCESS), 200


def test(args):
    pass

def create_user(args):


    user  = db.users.find_one({"email":args['email'].lower()})
    if user:
        return response_object(status=False, message=message.CONFLICT_409), 409

    user = {
        'email':args['email'],
        'password': bcrypt.generate_password_hash(args['password']).decode('utf-8'),
        'first_name':args['first_name'],
        'last_name':args['last_name'],
        'is_active':True,
        'is_admin':args['is_admin']
    }
    
    # active_code = {
    #     'email':user['email'],
    #     'code':''.join(random.choice(string.ascii_letters) for i in range(20))
    # }

    

    # try:s
    # if not mail_service.send_mail_active_user(active_code=active_code):
    #     return response_object(status=False, message=message.CREATE_FAILED), 500
    # db.codes.insert_one(active_code)
    db.users.insert_one(user)

    # except Exception as e:
    #     print(e)
    #     return response_object(status=False, data=str(e)), 500

    return response_object(), 201


def reset_password(args):


    user  = db.users.find_one({"email":args['email'].lower()})
    if not user:
        return response_object(status=False, message=message.EMAIL_NOT_EXISTS), 401
    email = args['email'].lower()
    # try:
    password = ''.join(random.choice(string.ascii_letters) for i in range(6))
    if not mail_service.send_mail_reset_password(email=email,password=password):
        return response_object(status=False, message=message.CREATE_FAILED), 500
    # db.codes.insert_one(active_code)
    db.users.update_one({'email':email }, {"$set": {'password': bcrypt.generate_password_hash(password).decode('utf-8')}})

    # except Exception as e:
    #     print(e)
    #     return response_object(status=False, data=str(e)), 500

    return response_object(), 200

def to_payload(user):
        return {
            'user_id': user['_id'],
            'is_admin': user['is_admin'],
            'is_active': user['is_active'],
        }

# def logout(token):
#     jti = get_jwt()["jti"]
#     if token:
#         black_list = BlackListToken(token=jti)
#         db.session.add(black_list)
#         db.session.commit()

#     return response_object(), 200
