import datetime
from app import db


class BlackListToken(db.Model):
    __tablename__ = 'token'

    token = db.Column(db.String(500), primary_key=True)
