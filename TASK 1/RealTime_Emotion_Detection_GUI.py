import numpy as np
import cv2

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier(
    "TASK 1\haarcascade_frontalface_default.xml"
)

def EmotionDetectionModel(json_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    return model


classifier = EmotionDetectionModel(
    "TASK 1\model_RED.json"
)

emotionLabels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray)

    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            pred = classifier.predict(roi)[0]
            label = emotionLabels[pred.argmax()]
            labelPosition = (x, y)
            cv2.putText(
                frame,
                label,
                labelPosition,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No Face Detected",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
    cv2.imshow("Emotion Detector by Gurditt", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
