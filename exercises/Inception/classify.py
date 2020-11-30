import inception
import time
from datetime import timedelta
import cv2


inception.download()

model = inception.Inception()

def classify(image_path):
    start_time = time.time()
    pred = model.classify(image_path = image_path)
    model.print_scores(pred = pred, k = 10, only_first_name = True)
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: ", timedelta(seconds = int(round(time_dif))))

classify(image_path = 'images/1.jpg')

#CLASSIFYING IMAGE WITH CAMERA
"""camera = cv2.VideoCapture(0)
while(True):
    return_value, image = camera.read()
    cv2.imwrite('images/0.jpg', image)
    classify(image_path = 'images/0.jpg')"""