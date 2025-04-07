import streamlit as st
import cv2
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os
from pyngrok import ngrok
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av

face_detector = MTCNN()
emotion_model = load_model('C:/Users/cheto/Desktop/SSS/1.3_emotion_input_output/output/emotion_model_pretrained.h5')
age_model = load_model('C:/Users/cheto/Desktop/SSS/1.1_age_input_output/output/age_model_pretrained.h5')
gender_model = load_model('C:/Users/cheto/Desktop/SSS/1.2_gender_input_output/output/gender_model_pretrained.h5')



# Labels on Age, Gender and Emotion to be predicted
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
emotion_ranges = ['positive', 'negative', 'neutral']

class_labels = emotion_ranges
gender_labels = gender_ranges
face_detector = MTCNN()


def predict_age_gender_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detect_faces(image)

    i = 0
    for face in faces:
        if len(face['box']) == 4:
            i = i + 1
            x, y, w, h = face['box']
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Crop the face ROI from the grayscale image
            roi_gray = gray[y:y + h, x:x + w]

            # Resize the ROI to 48x48 pixels and apply histogram equalization
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_gray = cv2.equalizeHist(roi_gray)

            # Get the ROI ready for prediction by scaling it between 0 and 1
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)

            # Use the emotion model to predict the emotion label of the ROI
            output_emotion = class_labels[np.argmax(emotion_model.predict(roi))]

            # Use the gender model to predict the gender label of the ROI
            gender_img = cv2.resize(roi_gray, (100, 100), interpolation=cv2.INTER_AREA)
            gender_image_array = np.array(gender_img)
            gender_input = np.expand_dims(gender_image_array, axis=0)
            output_gender = gender_labels[np.argmax(gender_model.predict(gender_input))]

            # Use the age model to predict the age range of the ROI
            age_image = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_AREA)
            age_input = age_image.reshape(-1, 200, 200, 1)
            output_age = age_ranges[np.argmax(age_model.predict(age_input))]

            # Build the output string with the predicted age, gender, and emotion labels
            output_str = str(i) + ": " + output_gender + ', ' + output_age + ', ' + output_emotion

            # Draw a rectangle and the output string on the original image
            col = (0, 255, 0)
            cv2.putText(image, output_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), col, 2)

    # Return the annotated image with the predicted labels
    return image


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = MTCNN()

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        # Convert the frame to an RGB numpy array
        img = frame.to_ndarray(format="rgb24")

        # Detect faces in the image
        faces = self.detector.detect_faces(img)

        # Draw bounding boxes around the detected faces
        for face in faces:
            x, y, w, h = face["box"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img[:, :, ::-1]



def take_photo():
    # Define webrtc_streamer params
    cap = cv2.VideoCapture(0)

    # Wait for the user to click the "Take a photo" button
    if st.button("Take a photo"):
        # Capture a frame from the video stream
        ret, frame = cap.read()

        # If a frame was successfully captured, use the predict_age_gender_emotion function to predict the age, gender, and emotion labels
        if ret:
            # Pass the captured image to the predict_age_gender_emotion function
            annotated_img = predict_age_gender_emotion(frame)

            # Display the annotated image to the user in the Streamlit app
            st.image(annotated_img, channels="BGR")

            # Release the video capture object and close all windows
            cap.release()
            cv2.destroyAllWindows()

            # Return the annotated image
            return annotated_img


def app():
    st.title("Age, Gender, and Emotion Prediction")

    # Call take_photo function to capture and process an image
    take_photo()

# Start the Streamlit app
if __name__ == "__main__":
    app()