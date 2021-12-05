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
# best = "/home/thanhnguyen_it_work/best.pt"
best = 'H:/Fresh-server/weights/best.pt'


def detectFromImage(img):
    # boxes = model(img).pandas().xyxy[0]
    # for i in boxes.index:
    #     start_point = (int(boxes['xmin'][i]), int(boxes['ymin'][i]))
    #     end_point = (int(boxes['xmax'][i]), int(boxes['ymax'][i]))
    #     img = cv2.rectangle(img, start_point, end_point, color, thickness)
    #     point = (int((boxes['xmin'][i]+boxes['xmax'][i])/2)-30,int((boxes['ymin'][i]+boxes['ymax'][i])/2))
    #     img = cv2.putText(img,"apple - 5", point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    # Displaying the image
    # cv2_imshow(image)

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
    history = {'url': url, 'result': parsed, 'time': time,
               'user_id': '', 'is_deleted': False, 'date': now}

    db.histories.insert_one(history)
    return encoder.toJson(history)


def download_file(name):
    return send_from_directory(os.path.abspath("images"), name)
