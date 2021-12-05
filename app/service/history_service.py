from sqlalchemy import func
import datetime
import random
import re
import string
from operator import or_, and_
from app import db, app, bcrypt
from app.model.user_model import User
from app.utils import encoder
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
from flask_pymongo import DESCENDING
from datetime import datetime


def get_all_by_user(args):
    histories = db.histories.find({"user_id": args['user_id'], "is_deleted": False}).sort(
        '_id', DESCENDING).skip(args['skip']).limit(args['limit'])
    histories = json.loads(JSONEncoder().encode(
        [history for history in histories]))
    return response_object(data=histories), 200


def get_with_duration(args):
    from_date = datetime.fromisoformat(args['start_date'])
    to_date = datetime.fromisoformat(args['end_date'])
    histories = db.histories.find({"user_id": args['user_id'], "date": {"$gte": from_date, "$lt": to_date}, "is_deleted": False}).sort(
        '_id', DESCENDING).skip(args['skip']).limit(args['limit'])
    histories = json.loads(JSONEncoder().encode(
        [history for history in histories]))
    return response_object(data=histories), 200


def rating(args):

    result = db.histories.update_one({"_id": ObjectId(
        args['_id']), "is_deleted": False}, {"$set": {'user_id': args['user_id'], 'rate': args['rate']}})

    if result.matched_count == 0:
        return response_object(status=False, message=message.NOT_FOUND_404), 404

    return response_object(status=True, message=message.SUCCESS), 200


def get_by_id(args):

    history = {}
    try:
        history = db.histories.find_one(
            {"_id": ObjectId(args['_id']), "user_id": args['user_id'], "is_deleted": False})
    except:
        response_object(status=True, message=message.BAD_REQUEST_400), 200
    if not history:
        return response_object(status=False, message=message.NOT_FOUND_404), 404
    history = json.loads(JSONEncoder().encode(history))
    return response_object(status=True, message=message.SUCCESS, data=encoder.toJson(history)), 200


def delete_history(args):
    history = {}
    try:
        history = db.histories.update_one({"_id": ObjectId(
            args['_id']), "user_id": args['user_id'], "is_deleted": False}, {"$set": {'is_deleted': True}})
    except:
        response_object(status=True, message=message.BAD_REQUEST_400), 200
    if history.matched_count == 0:
        return response_object(status=False, message=message.NOT_FOUND_404), 404
    return response_object(status=True, message=message.SUCCESS), 200


def to_payload(user):
    return {
        'user_id': user['_id'],
        'is_admin': user['is_admin'],
        'is_active': user['is_active'],
    }
