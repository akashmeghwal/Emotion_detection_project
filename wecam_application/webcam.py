import cv2
import numpy as np
from keras.models import load_model

# Load the emotion recognition model
model = load_model('final_model_file.h5')

# Load the Haar Cascade classifier for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Open the webcam for capturing video
cap = cv2.VideoCapture(0)

# Create a title bar at the top of the frame
title_bar_height = 50
title_bar_width = int(cap.get(3))
title_bar = np.zeros((title_bar_height, title_bar_width, 3), dtype=np.uint8)
title_text = "Emotion Detector By Akash"
text_color = (255, 255, 255)

# Get the size of the title text
text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

# Calculate the position to center the text
text_x = (title_bar_width - text_size[0]) // 2
text_y = (title_bar_height + text_size[1]) // 2

# Put the title text in the title bar
cv2.putText(title_bar, title_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        # Create a black rectangle with the same size as the resized image
        black_rect = np.zeros_like(resized, dtype=np.uint8)
        res = cv2.addWeighted(resized, 0.77, black_rect, 0.23, 0)


        # Add emotion label text to the processed face image
        emotion_label = labels_dict[label]
        color=(0,0,255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label_position = (x, y - 10)
        cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    # Concatenate the title bar, the processed frame, and the black bar vertically
    final_frame = np.vstack((title_bar, frame))
    
    # Display the combined frame with title, rectangles, and emotion labels
    cv2.imshow("Emotion Recognition", final_frame)
    
    # Check for the 'q' key press to exit the loop
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
