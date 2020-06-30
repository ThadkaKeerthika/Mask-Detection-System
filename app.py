import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from flask import Flask, request,render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model=load_model("mask.h5")
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/', methods=['GET','POST'])

def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("Current Path")
        basepath = os.path.dirname(__file__)
        print("Current Path",basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload floder is ",filepath)
        f.save(filepath)
        
        img = image.load_img(filepath, target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            print("Prediction",preds)
        index = ['without mask','with mask']
        text = "the prediction is : "+str(index[preds])
       
    return text

if __name__=='__main__':
    app.run(debug=True,threaded = False)