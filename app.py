
from flask import Flask, render_template, Response
import numpy as np
import cv2
import tensorflow 
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("C:/Users/hp/OneDrive/Documents/Christ University/Trimester_4/DATASET/model.h5")
model.load_weights("C:/Users/hp/OneDrive/Documents/Christ University/Trimester_4/DATASET/weights.weights.h5")

frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.5         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)



# Preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def get_class_name(classNo):
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons', 
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 
        'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 
        'No entry', 'General caution', 'Dangerous curve to the left', 
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 
        'Slippery road', 'Road narrows on the right', 'Road work', 
        'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 
        'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 
        'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 
        'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_names[classNo]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        threshold = 0.5

        while True:
            success, img_original = cap.read()
            if not success:
                break

            img = np.asarray(img_original)
            img = preprocessing(img)
            img = cv2.resize(img, (32, 32))
            img = img.reshape(1, 32, 32, 1)

            predictions = model.predict(img)
            classNo = np.argmax(predictions)
            probabilityValue = np.amax(predictions)

            # Add fixed text annotations to the image
            cv2.putText(img_original, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img_original, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

            # Add dynamic text annotations if the probability is above the threshold
            if probabilityValue > threshold:
                cv2.putText(img_original, f"{get_class_name(classNo)}", 
                            (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img_original, f"{round(probabilityValue * 100, 2)}%", 
                            (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

            # Encode the image as JPEG
            ret, buffer = cv2.imencode('.jpg', img_original)
            if not ret:
                continue
            frame = buffer.tobytes()

            # Yield the frame in the format expected by Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
    
