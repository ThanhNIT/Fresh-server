import json
from bson import ObjectId
from json import JSONEncoder
from bson.json_util import dumps, loads
from datetime import datetime


class UserEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return (str(o))
        return json.JSONEncoder.default(self, o)


def toJson(o):
    return json.loads(JSONEncoder().encode(o))
