import cv2
import sys
import json
import time
import numpy as np
from keras.models import model_from_json


emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
cascPath = sys.argv[1]

emojis = []
for emote in emotion_labels:
    img = cv2.imread(emote+".png", -1)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)

    emojis.append(img)


faceCascade = cv2.CascadeClassifier(cascPath)

# load json and create model arch
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('model.h5')

def predict_emotion(face_image_gray): # a single cropped face
    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    # cv2.imwrite(str(index)+'.png', resized_img)
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    return [angry, fear, happy, sad, surprise, neutral]

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY,1)


    faces = faceCascade.detectMultiScale(img_gray, 1.2, 5)

    emotions = []
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:

        face_image_gray = img_gray[y:y+h, x:x+w]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        emotions = predict_emotion(face_image_gray)
        ind = emotions.index(max(emotions))
        actual = emotion_labels[ind]
        cv2.putText(frame, actual, (y+h, x+w), cv2.FONT_HERSHEY_SIMPLEX, 2, (56, 15, 28), 2)
        with open('emotion.txt', 'a') as f:
            f.write('{},{},{},{},{},{},{}\n'.format(time.time(), emotions[0], emotions[1], emotions[2], emotions[3], emotions[4], emotions[5]))

        cv2.imshow("Emoji", emojis[ind])
       # frame[y:y+h, x:x+w] = np.array(cv2.split(emojis[ind])).reshape(frame[y:y+h, x:x+w].shape)

    # Display the resulting frame
    cv2.imshow('Video', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
