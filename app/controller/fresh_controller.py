from flask import request,send_from_directory
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Resource
from app import response_object
from app.dto.fresh_dto import FreshDto
from app.model.user_model import User
from app.service import demo_service
from app.service import fresh_service
from app.utils.auth_parser_util import get_auth_required_parser
from app.utils.jwt_util import admin_required, user_required
from werkzeug.utils import secure_filename
import os
api = FreshDto.api

_fresh_request = FreshDto.fresh_request
_get_file_request = FreshDto.get_file_request
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@api.route('/detect')
class Detect(Resource):
    @api.doc('fresh detect')
    @api.expect(_fresh_request, validate=True)
    def post(self):  # post put delete get
         # check if the post request has the file part
          
        if 'file' not in _fresh_request.parse_args():
            flash('No file part')
            return redirect(request.url)
        file = _fresh_request.parse_args()['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            # print(filename)
            # file.save(os.path.join('images', filename))
            return fresh_service.detectFromImage(file)


        
        return response_object(status=False, message='Invalid Image'), 400

@api.route('/download')
class Download(Resource):
    @api.doc('download')
    @api.expect(_get_file_request, validate=True)
    def post(self):  # post put delete get
        args = _get_file_request.parse_args()
        print(args['filename'])
        return download_file(args['filename'])

def allowed_file(filename):
    print(filename.rsplit('.', 1)[1].lower())
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




