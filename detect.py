# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

from matplotlib import cm
from skimage import transform
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as BK
import pandas as pd
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.plots import Annotator, colors
from utils.general import (
    check_img_size,
    check_requirements,
    colorstr,
    is_ascii,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.datasets import LoadStreams, LoadImages
from models.experimental import attempt_load
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path


def mapping_to_target_range(x, target_min=0, target_max=10):
    x02 = BK.tanh(x) + 1  # x in range(0,2)
    scale = (target_max - target_min) / 2.0
    return x02 * scale + target_min


# filename = "H:/Fresh-server/weights/apple_vgg16.h5"
# banana_h5 = "H:/Fresh-server/weights/banana_vgg16.h5"
# orange_h5 = "H:/Fresh-server/weights/orange_vgg16.h5"

filename = "/home/thanhnguyen_it_work/apple_vgg16.h5"
banana_h5 = "/home/thanhnguyen_it_work/banana_vgg16.h5"
orange_h5 = "/home/thanhnguyen_it_work/orange_vgg16.h5"

appleModel = ""
bananaModel = ""
orangeModel = ""

appleModel = load_model(
    filename, custom_objects={
        "mapping_to_target_range": mapping_to_target_range}
)
bananaModel = load_model(
    banana_h5, custom_objects={
        "mapping_to_target_range": mapping_to_target_range}
)
orangeModel = load_model(
    orange_h5, custom_objects={
        "mapping_to_target_range": mapping_to_target_range}
)
orangeModel.summary()

models = [appleModel, bananaModel, orangeModel]
host = "http://34.133.76.56"


@torch.no_grad()
def run(
    weights="yolov5s.pt",  # model.pt path(s)
    source="data/images",  # file/dir/URL/glob, 0 for webcam
    imgsz=640,  # inference size (pixels)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project="runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
):
    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (
        suffix == x for x in [".pt", ".onnx", ".tflite", ".pb", ""]
    )  # backend
    stride, names = 64, [f"class{i}" for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = (
            model.module.names if hasattr(model, "module") else model.names
        )  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name="resnet50", n=2)  # initialize
            modelc.load_state_dict(
                torch.load("resnet50.pt", map_location=device)["model"]
            ).to(device).eval()
    elif onnx:
        check_requirements(("onnx", "onnxruntime"))
        import onnxruntime

        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(("tensorflow>=2.4.1",))
        import tensorflow as tf

        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(
                    lambda: tf.compat.v1.import_graph_def(gd, name=""), []
                )  # wrapped import
                return x.prune(
                    tf.nest.map_structure(x.graph.as_graph_element, inputs),
                    tf.nest.map_structure(x.graph.as_graph_element, outputs),
                )

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, "rb").read())
            frozen_func = wrap_frozen_graph(
                gd=graph_def, inputs="x:0", outputs="Identity:0"
            )
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(
                model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # is TFLite quantized uint8 model
            int8 = input_details[0]["dtype"] == np.uint8
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    #     bs = len(dataset)  # batch_size
    # else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    # vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != "cpu":
        # run once
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if onnx:
            img = img.astype("float32")
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = (
                increment_path(save_dir / Path(path).stem, mkdir=True)
                if visualize
                else False
            )
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(
                session.run(
                    [session.get_outputs()[0].name], {
                        session.get_inputs()[0].name: img}
                )
            )
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]["quantization"]
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]["index"], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]["index"])
                if int8:
                    scale, zero_point = output_details[0]["quantization"]
                    pred = (pred.astype(np.float32) -
                            zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        data_frame = []
        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            # if webcam:  # batch_size >= 1
            #     p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            # else:
            p, s, im0, frame = path, "", im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # img.txt
            s += "%gx%g " % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(
                im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                data = []  # loop over the indexes
                columns = [
                    "cls",
                    "cf",
                    "imw",
                    "imh",
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                ]

                for *xyxy, conf, cls in reversed(det):
                    array = []
                    if save_txt:  # Write to file
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        # label format
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        line = ("%g " * len(line)).rstrip() % line
                        line = line.split()
                        array = line

                        # print(('%g ' * len(cls)).rstrip() % cls )
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # if save_img or save_crop or view_img:  # Add bbox to image
                    if save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = (
                            None
                            if hide_labels
                            else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        )
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        im0 = annotator.result()
                        array.append(im0.shape[1])
                        array.append(im0.shape[0])
                        record = []
                        record.append(array[0])
                        record.extend(array[5:])
                        record.extend(
                            convertToCoordinate(
                                float(array[1]),
                                float(array[2]),
                                float(array[3]),
                                float(array[4]),
                                array[6],
                                array[7],
                            )
                        )
                        if len(record) == 8:
                            data.append(record)

                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                data_frame = pd.DataFrame(data, columns=columns)

            # Print time (inference + NMS)
            print(f"{s}Done. ({t2 - t1:.3f}s)")

            # Stream results
    #         im0 = annotator.result()
    #         print(reversed(det))
    #         if view_img:
    #             cv2.imshow(str(p), im0)
    #             cv2.waitKey(1)  # 1 millisecond

    #         # Save results (image with detections)
    #         if save_img:
    #             if dataset.mode == 'image':
    #                 cv2.imwrite(save_path, im0)
    #             else:  # 'video' or 'stream'
    #                 if vid_path[i] != save_path:  # new video
    #                     vid_path[i] = save_path
    #                     if isinstance(vid_writer[i], cv2.VideoWriter):
    #                         vid_writer[i].release()  # release previous video writer
    #                     if vid_cap:  # video
    #                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                     else:  # stream
    #                         fps, w, h = 30, im0.shape[1], im0.shape[0]
    #                         save_path += '.mp4'
    #                     vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #                 vid_writer[i].write(im0)

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    img = cv2.imread(source)

    className = ["apple", "banana", "orange"]
    color = (0, 0, 255)
    thickness = 2

    if len(data_frame) == 0:
        return f"{host}{source.replace('app','')}", str([]), time.time() - t0
    levels = []
    for i in data_frame.index:
        predicted_class = int(data_frame["cls"][i])
        height = data_frame["y_max"][i] - data_frame["y_min"][i]
        width = data_frame["x_max"][i] - data_frame["x_min"][i]
        img_crop = img[
            data_frame["y_min"][i]: data_frame["y_min"][i] + height,
            data_frame["x_min"][i]: data_frame["x_min"][i] + width,
        ]
        if predicted_class == 2:
            cv2.imwrite("cam.jpg", img_crop)
            img_crop = load_image2("cam.jpg")
        else:
            img_crop = load_image(img_crop)
        # predict level
        level = round(
            float(models[predicted_class].predict(img_crop)[0][0]), 1)
        levels.append(level)
        if level < 6:
            color = (0, 0, 255)
        else:
            color = (0, 204, 0)
        # write results
        start_point = (int(data_frame["x_min"][i]),
                       int(data_frame["y_min"][i]))
        end_point = (int(data_frame["x_max"][i]), int(data_frame["y_max"][i]))
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        minus = 7

        point = (
            int((data_frame["x_min"][i] + data_frame["x_max"][i]) / 2) - minus,
            int((data_frame["y_min"][i] + data_frame["y_max"][i]) / 2),
        )
        img = cv2.putText(
            img,
            f"{className[predicted_class]}",
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
            False,
        )
        point = (
            int((data_frame["x_min"][i] + data_frame["x_max"][i]) / 2) - minus,
            int((data_frame["y_min"][i] + data_frame["y_max"][i]) / 2) + 30,
        )
        img = cv2.putText(
            img,
            f"{level}",
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
            False,
        )

    cv2.imwrite(source, img)
    data_frame["level"] = levels

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f"Done. ({time.time() - t0:.3f}s)")
    return (
        f"{host}{source.replace('app','')}",
        data_frame.to_json(orient="records"),
        time.time() - t0,
    )


def load_image(np_array):
    np_image = np.array(np_array).astype("float32") / 255
    np_image = transform.resize(np_image, (120, 120, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def load_image2(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype("float32") / 255
    np_image = transform.resize(np_image, (120, 120, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def convertToCoordinate(
    x_rect_mid=0.0,
    y_rect_mid=0.0,
    width_rect=0.0,
    height_rect=0.0,
    img_width=0.0,
    img_height=0.0,
):

    x_min_rect = ((2 * x_rect_mid * img_width) - (width_rect * img_width)) / 2
    x_max_rect = ((2 * x_rect_mid * img_width) + (width_rect * img_width)) / 2
    y_min_rect = ((2 * y_rect_mid * img_height) -
                  (height_rect * img_height)) / 2
    y_max_rect = ((2 * y_rect_mid * img_height) +
                  (height_rect * img_height)) / 2

    return int(x_min_rect), int(y_min_rect), int(x_max_rect), int(y_max_rect)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov5s.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true",
                        help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true",
                        help="augmented inference")
    parser.add_argument("--visualize", action="store_true",
                        help="visualize features")
    parser.add_argument("--update", action="store_true",
                        help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp",
                        help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr("detect: ") +
          ", ".join(f"{k}={v}" for k, v in vars(opt).items()))
    check_requirements(exclude=("tensorboard", "thop"))
    run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
