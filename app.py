from flask import Flask, jsonify ,request, render_template,send_file, make_response
from werkzeug.utils import secure_filename
import pandas as pd
from Api import Api
import sys, os
import time
import os.path

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

obj = Api()

UPLOAD_FOLDER = 'static/tempData/'
ALLOWED_EXTENSIONS = set(['png','jpg','csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classification')
def classification():
    path = request.path
    return render_template('classification.html', data=path)

@app.route('/')
def home():
    path = request.path
    return render_template('home.html', data=path)

@app.route('/train')
def train():
    path = request.path
    return render_template('train.html', data=path)

@app.route('/train_process', methods=['POST'])
def train_process():
    path = request.path

    model = obj.modelCNN()
    # obj.train(model,'images_after_prepro/train','images_after_prepro/validation')
    print("last modified: %s" % time.ctime(os.path.getmtime('data/model_new.h5')))
    print("created: %s" % time.ctime(os.path.getctime('data/model_new.h5')))
    try:
        return jsonify({ 'code':200, 'message' : 'Success' ,'data':'success','updated': "Last Model Updated: %s" % time.ctime(os.path.getmtime('data/model_new.h5'))}), 200
    except e:
        return jsonify({ 'code':500, 'message' : 'Success', 'error': str(e) }), 500

@app.route('/process',methods=['POST'])
def classification_process():
    path = request.path
   
    error = ""
    if 'file' in request.files:
        filetxt = request.files["file"]
        if filetxt and allowed_file(filetxt.filename):
            filename = secure_filename(filetxt.filename)
            print(filename,filetxt.filename)
            filetxt.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            error = "Format file salah"
    
    img = obj.read_img('static/tempData/'+filename)

    imgScale = obj.scale_img(img)
    imgGray = obj.grayscale(imgScale)
    imgHsv = obj.rgb_hsv(imgScale)
    imgThresh = obj.threshold(imgScale)

    obj.save_img(imgScale,'scaled_image.'+filename.split('.')[1])
    obj.save_img(imgGray,'grayscale_image.'+filename.split('.')[1])
    obj.save_img(imgHsv,'rgbToHsv_image.'+filename.split('.')[1])
    obj.save_img(imgThresh,'treshold_image.'+filename.split('.')[1])
    
    model = obj.loadModel('data/model.h5')
    predictImage = obj.predict(model, 'static/tempData/treshold_image.'+filename.split('.')[1]) 

    try:
        return jsonify({ 'code':200, 'message' : 'Success' ,'data':{
            'classes': predictImage,
            'image': 'static/tempData/'+filename,
            'image_after_preprocessing': 'static/tempData/'+'treshold_image.'+filename.split('.')[1]
        }}), 200
    except e:
        return jsonify({ 'code':500, 'message' : 'Success', 'error': str(e) }), 500

@app.route('/download')
def downloadFile():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    data = "data/model.h5"
    resp = make_response(data)
    resp.headers["Content-Disposition"] = "attachment; filename=model.h5"
    resp.headers["Content-Type"] = "application/x-hdf;subtype=bag"
    return resp

if __name__ == "__main__":
    # app.run(debug=True, port=1337)
    port = int(os.environ.get("PORT",1337))
    app.run(host="0.0.0.0",port=port)