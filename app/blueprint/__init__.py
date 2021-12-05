import werkzeug
from flask import Blueprint
from flask_restx import Api

from app import response_object
from app.controller.user_controller import api as user_api
from app.controller.fresh_controller import api as fresh_api
from app.controller.history_controller import api as history_api
blueprint = Blueprint('api', __name__, url_prefix="/api", template_folder='templates')
api = Api(blueprint,
          title='API DOCUMENT',
          version='1.0'
          )

api.add_namespace(user_api, path='/user')
api.add_namespace(history_api, path='/histories')
api.add_namespace(fresh_api, path='/fresh')

@api.errorhandler(werkzeug.exceptions.BadRequest)
def bad_request(error):
    return response_object(status=False, message='Bad Request'), 400


@api.errorhandler(werkzeug.exceptions.Unauthorized)
def bad_request(error):
    return response_object(status=False, message='Unauthorized'), 401


@api.errorhandler(werkzeug.exceptions.Forbidden)
def forbidden(error):
    return response_object(status=False, message='Forbidden'), 403


@api.errorhandler(werkzeug.exceptions.NotFound)
def page_not_found(error):
    return response_object(status=False, message='Page Not Found'), 404


@api.errorhandler(werkzeug.exceptions.InternalServerError)
def internal_server_error(error):
    return response_object(status=False, message='Server Error'), 500
