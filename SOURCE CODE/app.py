import re
import numpy as np
import os
from flask import Flask, app,request,render_template
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
#Loading the model
model=load_model(r"C:\Users\ADMIN\Desktop\ibmpro\models.h5")

app=Flask(__name__)


#default home page or route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/index.html')
def home():
    return render_template("index.html")

@app.route('/result',methods=["GET","POST"])
def res():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__) #getting the current path i.e where app.py is present
        #print("current path",basepath)
        filepath=os.path.join(basepath,'uploads',f.filename) #from anywhere in the system we can give image but we want that image later  to process so we are saving it to uploads folder for reusing
        #print("upload folder is",filepath)
        f.save(filepath)

        new_img=image.load_img(filepath,target_size=(224,224,3))
        img = np.array(new_img) / 255.0
        img = np.expand_dims(img, axis=0)


        prediction = model.predict(img)

# Define the class labels in the same order as the model's output
        li = ['Anthracnose','algal leaf','bird eye spot','white spot','gray light','healthy','red leaf spot','brown blight']

        d = prediction.flatten()
        j  = d.max()
        for index, item in enumerate(d):
            if item == j:
                class_name = li[index]

        
    
        
        return render_template('prediction.html',prediction=class_name)
        



""" Running our application """
if __name__ == "__main__":
    app.run(debug=True,port=8000)