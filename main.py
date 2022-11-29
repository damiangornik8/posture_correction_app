
import numpy as np
import cv2


# haarscdoe data
# https://github.com/kipr/opencv/tree/master/data/haarcascades


# def convertToRGB(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

import winsound
duration = 200  # milliseconds
freq = 440  # Hz

# Haar Cascade is a machine learning-based approach where a lot of positive and 
# negative images are used to train the classifier. Positive images – These images contain the 
# images which we want our classifier to identify. 
# Negative Images – Images of everything else, which do not contain the object we want to detect.

haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')



# haar cascade face + test image + 
# scaleFactor – Parameter specifying how much the image size is reduced at each image scale.


import time

def current_milli_time():
    return round(time.time() * 1000)

time_now = current_milli_time()
time_to_work = current_milli_time() + 10000



def detect_faces(cascade, camera_view, scaleFactor = 1.1):
    global camera_view_copy 
    camera_view_copy = camera_view.copy()

    global x_var 
    global y_var 
    global w_var 
    global h_var 

    

  

    

    # OpenCV facedetector user grayimages
    gray_image = cv2.cvtColor(camera_view_copy, cv2.COLOR_BGR2GRAY)
    # Gray image to numbers
    gray_image = np.array(gray_image, dtype='uint8')

    
    # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    # In other words, this parameter will affect the quality of the detected faces. H
    # igher value results in less detections but with higher quality.
    # We use v2.CascadeClassifier.detectMultiScale() to find faces or eyes, and it is defined like this:


    # here we apply the haar classifier to detect faces on our live cam
    face_rectangle = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in face_rectangle:
        draw = cv2.rectangle(camera_view_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)

        x_var = x 
        y_var = y
        w_var = w
        h_var = h

        # Use that code when finding the most ergonomic position
        # print( "X: " + str(x_var) )
        # print( "Y" + str(y_var) )
        # print( "W" + str(w_var) )
        # print( "H" + str(h_var) )


    return camera_view_copy

import time

# def detect_correct_posture():
#     time.sleep(10)


# ret is a boolean variable that returns true if the frame is available.
# frame is an image array vector captured based on the default frames per second defined explicitly or implicitly




cap = cv2.VideoCapture(0)

ret, frame = cap.read()
face_detect = detect_faces(haar_cascade_face, frame)




# font = cv2.FONT_HERSHEY_SIMPLEX

# draw3 = cv2.putText(img=image_copy, text='Hello', org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)

# print("Prosze usiasc w komfortowej pozycji i byc w niej przez 3 sekundy")
# draw3 = cv2.putText(img=image_copy, text='22', org=(150, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)

# time.sleep(3)
# print("Kalibracja urzadzenia")
# time.sleep(1)
# print("Urzadzenie gotowe do dzialania")

calibrated = False


if calibrated == False:
    x = x_var 
    y = y_var
    w = w_var
    h = h_var

    print("Calibrated")

    calibrated = True

cap = cv2.VideoCapture(0)

while(True):
    time_now = current_milli_time()
    # Capture frame-by-frame
    ret, frame = cap.read()

    

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #do face detection
    face_detect = detect_faces(haar_cascade_face, frame)

    


    




  


    


             
    

        


    draw2 = cv2.rectangle(camera_view_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)


    # Windsound lib for sounds


    

    # When right?
    if ( (x - 50) >= x_var):
      
      
        if (time_now > time_to_work):
            winsound.Beep(freq, duration)
            
            time_to_work = current_milli_time() + 1000
            draw = cv2.rectangle(camera_view_copy, (x_var, y_var), (x_var+w_var, y_var+h_var), (16, 0, 235), 3)


    # left?
    if ( (x + 80) <= x_var):


        if (time_now > time_to_work):
            winsound.Beep(freq, duration)
            time_to_work = current_milli_time() + 1000
            draw = cv2.rectangle(camera_view_copy, (x_var, y_var), (x_var+w_var, y_var+h_var), (16, 0, 235), 3)
        




    # When top?
    if ( (y - 50) >= y_var):
       
        if (time_now > time_to_work):
            winsound.Beep(freq, duration)
            time_to_work = current_milli_time() + 1000
            draw = cv2.rectangle(camera_view_copy, (x_var, y_var), (x_var+w_var, y_var+h_var), (16, 0, 235), 3)
   


    # bottom:
    if ( (y + 60) <= y_var):
      
        if (time_now > time_to_work):
            winsound.Beep(freq, duration)
            time_to_work = current_milli_time() + 1000
            draw = cv2.rectangle(camera_view_copy, (x_var, y_var), (x_var+w_var, y_var+h_var), (16, 0, 235), 3)
 

    # When left??
    # if ( (w - 10) <= w_var):
    #     winsound.Beep(freq, duration)

    # When bottom?
    # if ( (h + 5) >= h_var):
    #     winsound.Beep(freq, duration)


    

   




    # Display the resulting frame
    cv2.imshow('frame',face_detect)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()












img = cv2.imread('test_image.jpg')

face_test = detect_faces(haar_cascade_face, img)
cv2.imshow('test', face_test)

cv2.waitKey()

