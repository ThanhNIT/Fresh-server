import json
from datetime import datetime

import jwt
import pytz

from app import db, bcrypt, app
from app.model.black_list_token import BlackListToken
# from app.model.follow import follow_table
from app.utils.api_response import json_serial, date_to_json


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(255), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(255), nullable=True)
    last_name = db.Column(db.String(255), nullable=True)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=False, nullable=False)

    created_date = db.Column(db.DateTime, nullable=True)
    updated_date = db.Column(db.DateTime, nullable=True)

    def __init__(self, email, password, first_name, last_name, is_admin=False, is_active=False):
        self.email = email
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')
        self.first_name = first_name
        self.last_name = last_name
        self.is_admin = is_admin
        self.is_active = is_active
        self.created_date = datetime.now()
        self.updated_date = datetime.now()

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

    # @staticmethod
    # def encode_auth_token(user_id, is_admin, is_tutor):
    #     try:
    #         payload = {
    #             "expired_time": json.dumps((datetime.now() + app.config['TOKEN_EXPIRED_TIME']), default=json_serial),
    #             "issued_at": json.dumps(datetime.now(), default=json_serial),
    #             "user_id": user_id,
    #             "is_admin": is_admin
    #         }
    #         auth_token = jwt.encode(
    #             payload,
    #             app.config.get('SECRET_KEY'),
    #             algorithm="HS256"
    #         )
    #
    #         token = Token(token=auth_token)
    #         db.session.add(token)
    #         db.session.commit()
    #         return auth_token
    #     except Exception as e:
    #         return e

    @staticmethod
    def decode_auth_token(auth_token):
        try:
            payload = jwt.decode(auth_token, app.config.get('SECRET_KEY'), algorithms=["HS256"])

            token = BlackListToken.query.filter(BlackListToken.token == auth_token).first()

            if not token:
                return 'Token blacklisted. Please log in again.'
            else:
                tz_London = pytz.timezone('Asia/Saigon')
                if (token.created_date + app.config.get("TOKEN_EXPIRED_TIME")) < datetime.now():
                    db.session.delete(token)
                    db.session.commit()
                    return 'Signature expired. Please log in again.'
                return payload
        except jwt.ExpiredSignatureError:
            return 'Signature expired. Please log in again.'
        except jwt.InvalidTokenError:
            return 'Invalid token. Please log in again.'

    def verify_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

    def to_json(self):
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_admin': self.is_admin,
            'is_active': self.is_active,
            'created_date': date_to_json(self.created_date),  # json.dumps(self.created_date, default=json_serial),
            'updated_date': date_to_json(self.updated_date)  # json.dumps(self.created_date, default=json_serial),
        }

    def to_payload(self):
        return {
            'user_id': self.id,
            'is_admin': self.is_admin,
            'is_active': self.is_active,
        }
