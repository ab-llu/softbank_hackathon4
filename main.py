from flask import Flask, request, render_template, send_from_directory, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

import torch, detectron2
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, uuid

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No image selected for uploading"
    if file:
        _, file_extension = os.path.splitext(file.filename)
        filename = f"{uuid.uuid4()}{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(os.path.join('static', filepath))
        
        # ここで画像加工処理を呼び出す
        [processed_image_url, plaque_pro, inflammation] = process_image(os.path.join('static', filepath))
        
        return render_template('image_display.html', image_url=processed_image_url, plaque_pro=plaque_pro, inflammation=inflammation)
    else:
        return "Something went wrong"

cfg = get_cfg()
cfg.merge_from_file("config/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

rcnn_model_path = 'model/tooth_and_plaque.pth'
# ファイルが存在するかチェック
if os.path.exists(rcnn_model_path):
    print(f"{rcnn_model_path} exists.")
else:
    print(f"Error: {rcnn_model_path} does not exist.")
cfg.MODEL.WEIGHTS = rcnn_model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
thing_classes = ["plaque", "plaque", "tooth"]
predictor = DefaultPredictor(cfg)

loaded_model = load_model("model/model.h5")

def process_image(image_path):
    from detectron2.utils.visualizer import ColorMode
    import cv2

    im = cv2.imread(image_path)
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    plaque_instances = instances[instances.pred_classes == 1]

    visualized_image = im.copy()
    for i in range(len(plaque_instances)):
        # マスクを取得し、描画条件を満たすピクセルを赤色で塗りつぶす
        mask = plaque_instances.pred_masks[i].numpy()
        green_mask = np.zeros_like(visualized_image)
        green_mask[:, :, 1] = 255  # 赤色チャネルを最大に
        visualized_image[mask] = cv2.addWeighted(visualized_image[mask], 0.5, green_mask[mask], 0.5, 0)
    
    base_name, ext = os.path.splitext(image_path)
    masked_im_path = f"{base_name}_masked{ext}"
    cv2.imwrite(masked_im_path, visualized_image)

    #面積を計算
    area_per_class = {}
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()  # セグメンテーションマスクをnumpy配列に変換
        classes = instances.pred_classes.numpy()  # 予測されたクラスIDをnumpy配列に変換
    for mask, cls in zip(masks, classes):
        # クラスIDをキーとして面積を合計
        area = mask.sum()  # マスク内のTrueの数が面積に相当
        if cls in area_per_class:
            area_per_class[cls] += area
        else:
            area_per_class[cls] = area
    plaque_pro = np.round(area_per_class[1] / area_per_class[2] * 100)
    
    #炎症かどうか判定
    img_i = image.load_img(image_path, target_size=(640, 640))
    img_array = image.img_to_array(img_i)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_labels = {0: '炎症あり', 1: '炎症なし'}
    predicted_label = class_labels[predicted_class]

    return [masked_im_path, plaque_pro, predicted_label]

if __name__ == "__main__":
    app.run(debug=True)

