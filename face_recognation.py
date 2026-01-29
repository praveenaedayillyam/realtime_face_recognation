from keras.models import load_model 
import cv2  # Install opencv-python
import numpy as np

np.set_printoptions(suppress=True)

# Loading the model
model = load_model("keras_Model.h5", compile=False)

# Loading the labels
class_names = open("labels.txt", "r").readlines()

# default camera of computer
camera = cv2.VideoCapture(0)

# window to full-screen mode
cv2.namedWindow("Webcam Image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Webcam Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, image = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resizing the raw image to (224-height, 224-width) pixels
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Converting the image into a numpy array and reshape it to match the model's input shape
    image_array = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalizing the image array
    image_array = (image_array / 127.5) - 1

    # prediction
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip() 
    confidence_score = prediction[0][index] * 100 

    # To Display prediction on the camera feed
    text = f"{class_name}: {confidence_score:.2f}%"
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image in full-screen mode
    cv2.imshow("Webcam Image", image)

    #'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
