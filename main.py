from flask import Flask, request, render_template, send_from_directory, url_for, redirect, session, render_template_string
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import json

# Some basic setup:
# Setup detectron2 logger
import torch, detectron2
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

# FlaskとFlask-Loginの初期化
app = Flask(__name__)
app.secret_key = "jfwei938n329fn"
login_manager = LoginManager()
login_manager.init_app(app)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 仮のユーザー
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# ユーザーローダーの設定
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/')
def index():
    return redirect(url_for('login'))

# ログインページ
@app.route('/login', methods=['GET', 'POST'])
def login():
    session['points'] = {}
    if request.method == 'POST':
        # ログイン処理（ここでは仮の処理を行います）
        user_id = request.form.get('user_id')
        session['user'] = user_id
        user = User(user_id)
        login_user(user)
        return redirect(url_for('home'))
    return render_template('login.html')

# ログアウト
@app.route('/logout')
@login_required
def logout():
    logout_user()
    # アップロードディレクトリ内のファイルを全て削除
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return redirect(url_for('login'))

# ホームページ
@app.route('/home')
@login_required
def home():
    now = datetime.now()
    date_string = now.strftime("%Y年%m月%d日")
    return render_template('home.html', date=date_string, user=session['user'])

#記録
@app.route('/record_list')
@login_required
def record_list():
    now = datetime.now()
    date_string = now.strftime("%Y年%m月%d日")
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    day_before_yesterday = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    display = {} #ホーム画面表示用データ格納
    for day in [today, yesterday, day_before_yesterday]:
        display[day] = {}
        for time_of_day in ['morning', 'afternoon', 'evening']:
            time_label = day + "_" + time_of_day
            if time_label in session['points'].keys():
                display[day][time_of_day] = session['points'][time_label]
                display[day][time_of_day]['json'] = json.dumps(session['points'][time_label])
            else:
                display[day][time_of_day] = None
    return render_template('record_list.html', points = session['points'], date=date_string, display=display, user=session['user'])

#歯科予約
@app.route('/reserve')
@login_required
def reserve():
    now = datetime.now()
    date_string = now.strftime("%Y年%m月%d日")
    return render_template('reserve.html', date=date_string, user=session['user'])

#チャット
@app.route('/chat')
@login_required
def chat():
    now = datetime.now()
    date_string = now.strftime("%Y年%m月%d日")
    return render_template('chat.html', date=date_string, user=session['user'])

#歯周病を知る
@app.route('/learn')
@login_required
def learn():
    now = datetime.now()
    date_string = now.strftime("%Y年%m月%d日")
    return render_template('learn.html', date=date_string, user=session['user'])

#設定
@app.route('/setting')
@login_required
def setting():
    now = datetime.now()
    date_string = now.strftime("%Y年%m月%d日")
    return render_template('setting.html', date=date_string, user=session['user'])

#画像アップロードページ
@app.route('/upload_folder')
@login_required
def upload_form():
    day = request.args.get('day', default=None, type=str)
    time = request.args.get('time', default=None, type=str)
    print(day, time)

    now = datetime.now()
    if day is None:
        day = now.strftime("%Y-%m-%d")
    if time is None:
        hour = now.hour
        if 4 < hour < 11: time = "morning"
        elif 10 < hour < 17: time = "afternoon"
        else: time = "evening"

    return render_template('upload.html', day=day, time=time)

#記録表示ページ
@app.route('/record', methods=['GET'])
@login_required
def show_record():
    data_str = request.args.get('data', default=None, type=str)
    if data_str is None:
        return "データがありません"
    try:
        data_dict = json.loads(data_str)
    except json.JSONDecodeError:
        return "JSON形式のデータの解析に失敗しました"
    return render_template('image_display.html', data=data_dict)

#結果表示ページ(画像投稿時)
@app.route('/upload', methods=['POST'])
@login_required
def upload_image():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No image selected for uploading"
    if file:
        date = request.form['date']
        time_of_day = request.form['time_of_day']
        time_label = date + "_" + time_of_day
        _, file_extension = os.path.splitext(file.filename)
        filename = time_label + "_" + f"{uuid.uuid4()}{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # ここで画像加工処理を呼び出す
        [processed_image_url, plaque_pro, predicted_class] = process_image(filepath)
        class_labels = {0: '炎症あり', 1: '炎症なし'}
        inflammation_label = class_labels[predicted_class]
        tooth_point = int(50 * predicted_class + 50 - plaque_pro/2)
        print(type(plaque_pro), type(predicted_class), type(tooth_point))  # データ型を確認
        session['points'][time_label] = {'plaque': plaque_pro, 'inflammation': predicted_class, 'point': tooth_point, 'url': processed_image_url}
        session.modified = True

        print(session['points'][time_label])

        return render_template('image_display.html', data=session['points'][time_label])
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
    # with open(rcnn_model_path, 'r') as file:
    #     first_line = file.readline()
    #     print(first_line)
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
    masks, classes = [], [] 
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()  # セグメンテーションマスクをnumpy配列に変換
        classes = instances.pred_classes.numpy()  # 予測されたクラスIDをnumpy配列に変換
    if 2 not in classes:
        plaque_pro = int(0) #将来的には、「歯が検出できませんでした」というエラーメッセージを書く
    elif 1 not in classes:
        plaque_pro = int(0)
    else:
        for mask, cls in zip(masks, classes):
            # クラスIDをキーとして面積を合計
            area = mask.sum()  # マスク内のTrueの数が面積に相当
            if cls in area_per_class:
                area_per_class[cls] += area
            else:
                area_per_class[cls] = area
    plaque_pro = int(np.round(area_per_class[1] / area_per_class[2] * 100))
    
    #炎症かどうか判定
    img_i = image.load_img(image_path, target_size=(640, 640))
    img_array = image.img_to_array(img_i)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = loaded_model.predict(img_array)
    predicted_class = int(np.argmax(predictions)) #0: '炎症あり', 1: '炎症なし'

    return [masked_im_path, plaque_pro, predicted_class]

if __name__ == "__main__":
    app.run(debug=True)

