from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.applications import vgg16
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import imageio

app=Flask(__name__)

loaded_model = load_model('transferlearningmodel.h5')
loaded_model._make_predict_function()

@app.route('/')
def home():
    return render_template("home.html")

def ClassPredictor(file):
    #preprocess file
    # use 
    result = loaded_model.predict(file)
    return result[0]

def process_image(img):
    images = []
    # Convert the image to a numpy array
    image_array = image.img_to_array(img)
    
    # Add the image to the list of images
    images.append(image_array)
    X = np.array(images)
    print(X.shape)
    X = vgg16.preprocess_input(X)
    result = ClassPredictor(X)
    return result

@app.route('/result',methods = ['POST'])
def result():
    prediction=''
    if request.method == 'POST':
        file = request.files['file']
        file = imageio.imread(file)
        result = process_image(file)       
        print("result from model", result)
        if float(result)<0.5:
            prediction = 'This is not a dog'
        else:
            prediction = 'This is a dog'
        print(prediction)
        return render_template("result.html",prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
