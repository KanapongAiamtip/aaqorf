from flask import Flask, render_template, request, jsonify  # เพิ่ม jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2

app = Flask(__name__)

@app.route('/')
def home(): 
   return render_template('index.html')

@app.route('/about')
def about(): 
   return render_template('about.html')


# @app.route('/good')
# def good(): 
#    return render_template('good.html')

# @app.route('/bad')
# def bad(): 
#    return render_template('bad.html')

# @app.route('/normal')
# def normal(): 
#    return render_template('normal.html')


# Load the pomelo detection model
model = load_model('pomelo.h5', compile=True)

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize to (224, 224)
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define a function to predict emotion
def predict_emotion(image_path):
    img = preprocess_image(image_path)
    result = model.predict(img)
    emotion_labels = ['Good','Normal','Bad']
    emotion = emotion_labels[np.argmax(result)]
    return emotion

@app.route('/detect_org', methods=['POST'])
def detect_org():
    if 'image' in request.files:
        uploaded_image = request.files['image']
        if uploaded_image.filename != '':
            image_path = 'static/uploaded_image.png'
            uploaded_image.save(image_path)
            emotion = predict_emotion(image_path)
            
            if emotion == "Good":
                return render_template('good.html', image=image_path)
            elif emotion == "Normal":
                return render_template('normal.html', image=image_path)
            elif emotion == "Bad":
                 return render_template('bad.html', image=image_path)
            else:
                # หากไม่ใช่  ให้กลับไปที่หน้าหลัก
                return render_template('index.html', result=emotion, image=image_path)
    return render_template('index.html')

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5000, debug=True)