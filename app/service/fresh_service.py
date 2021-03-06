from sqlalchemy import func
import datetime
import random
import re
import string
from operator import or_, and_
from app import db, app
from app.model.user_model import User
from app.utils import encoder
# from app.model.black_list_token import BlackListToken
from app.utils.api_response import response_object
import app.utils.response_message as message
from flask_jwt_extended import create_access_token, get_jwt
import torch
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import datetime
from flask import request, send_from_directory
import os
import detect
from werkzeug.utils import secure_filename
import json
from datetime import date


# model = torch.hub.load(os.path.abspath("weights"), 'custom', path=os.path.abspath("best.pt"), source='local')  # local repo
best = "/home/thanhnguyen_it_work/best.pt"
best_all = "/home/thanhnguyen_it_work/all_best.pt"
# best = 'H:/Fresh-server/weights/best.pt'


def detectFromImage(img):

    UPLOAD_FOLDER = 'app/static'
    now = datetime.datetime.now()
    date = str(now)
    filename = 'admin_'+date+'.jpg'
    filename = secure_filename(filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    img.save(os.path.join(UPLOAD_FOLDER, filename))

    # return download_file(filename)
    url, result, time = detect.run(
        save_txt=True, save_conf=True, save_crop=True, weights=best, source=path)
    parsed = json.loads(result)
    accepted =0
    rejected =0
    for e in parsed:
        if int(e['level'])>=6:
            accepted+=1
        else:
            rejected+=1
    
    history = {'url': url, 'result': parsed, 'time': time,
               'user_id': '', 'is_deleted': False, 'date': now, 'accepted':accepted, 'rejected':rejected}

    db.histories.insert_one(history)
    return encoder.toJson(history)


def download_file(name):
    return send_from_directory(os.path.abspath("images"), name)
