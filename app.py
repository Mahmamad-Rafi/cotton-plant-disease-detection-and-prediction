from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
#import tensorflow as tf
#import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, render_template, request
 
#from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH ='C:/Users/rafis/OneDrive/Desktop/project/cotton_disease/model/cotton_plant.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        return "diseased Cotton leaf", 'disease_plant.html'
    elif preds==1:
        return 'Diseased Cotton Plant', 'disease_plant.html'
    elif preds==2:
        return 'fresh Cotton leaf', 'healthy_plant_leaf.html'
    else:
        return "fresh Cotton Plant", 'healthy_plant.html'
    
    

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        #user_image = "path_to_user_image.jpg"  
        return render_template('index.html')
     
  
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('C:/Users/rafis/OneDrive/Desktop/project/cotton_disease/static/uploads',filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred, output_page = model_predict(img_path=file_path)
               
        return render_template(output_page, pred_output = pred, user_image = file_path[54:])
     
      #  preds = model_predict(file_path, model)
        
if __name__ == '__main__':
    app.run(port=5010, debug=True)
    
    