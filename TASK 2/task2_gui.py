import cv2
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
from keras.preprocessing import image
# from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

model = load_model("TASK 2/best_model.h5")

face_haar_cascade = cv2.CascadeClassifier('TASK 2\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = sr.Recognizer()

while True:
    ret, test_img = cap.read() 
    if not ret:
        continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h] 
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = roi_gray.astype('float32')
        img_pixels /= 255.0

        pred = model.predict(np.expand_dims(img_pixels, axis=0))
        max_index = np.argmax(pred[0])
        emotions = ('Sad', 'Surprise', 'Neutral', 'Angry', 'Disgust', 'Fear', 'Happy')
        predicted_emotion = emotions[max_index]
        print(predicted_emotion)

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Emotion Detector with Voice by Gurditt', resized_img)

    with sr.Microphone() as source:
        print("Say something...")
        audio_data = recognizer.listen(source)

    try:    
        text = recognizer.recognize_google(audio_data)
        print("Emotion:", text)
  
        if 'quit' in text.lower():
            break

    except sr.UnknownValueError:
        print("Speech Recognition couldn't understand Audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()