# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
# import the necessary packages
from imutils.face_utils import FaceAligner, rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
import json
import urllib
from urllib import request, parse

def extract_face(faceAligner, detector, size_threshold, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
    rects = detector(gray, 0)
    faceAligned = []
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        if w * h > size_threshold:
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=96)
            faceAligned = faceAligner.align(image, gray, rect)

            # display the output images
            cv2.imshow("Original", faceOrig)
            cv2.imshow("Aligned", faceAligned)
    return faceAligned


def display_id(image, current_id):
    if current_id == "None":
        display_text = 'Sorry I do not recognize you'
    else:
        display_text = "Hi " + str(current_id)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    cv2.putText(image, display_text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType) 

def display_text(image,text):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,10)
    fontScale              = 0.6
    fontColor              = (255,255,255)
    lineType               = 2
    cv2.putText(image, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType) 

def determine_id(result, tracker):
    current_id = result['name']
    if current_id == tracker['last_id']:
        tracker['last_id_cnt'] = tracker['last_id_cnt'] + 1
        if tracker['last_id_cnt'] > 3:
            tracker['current_id'] = current_id
            tracker['current_cnt'] = tracker['last_id_cnt']
            tracker['last_id_cnt'], tracker['oscillating_cnt'] = 0,0
    elif current_id != tracker['current_id']:
        tracker['last_id'] = current_id
        tracker['oscillating_cnt'] = tracker['oscillating_cnt'] + 1
        if tracker['oscillating_cnt'] > 3:
            tracker['current_id'] = None
    else:
        tracker['current_id_cnt'] = tracker['current_id_cnt'] + 1
    return tracker['current_id'], tracker


def send_image(image, url):
    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    _, img_encoded = cv2.imencode('.jpg', image)
    data = img_encoded.tostring()
    try:
         # send http request with image and receive response
        r = request.Request(url, data=data, headers=headers, method='POST')
        resp = request.urlopen(r, timeout=10)
        result = json.loads(resp.read().decode('utf-8'))
    except urllib.error.URLError as err:
        print("Error accessing URL " + url )
        raise err
    return resp, result
        


# default URL to our face detection API
url = "http://127.0.0.1:5000"

#
# Data structure to track various variables to smooth out oscillating detections
tracker = { 'current_id'       : None,      # keep track of the id currently detected
            'current_id_cnt'   : 0,         # count of consecutive detections
            'last_id'          : None,      # id which has been detected last time
            'last_id_cnt'      : 0,         # count how often the id has been seen, consecutively
            'oscillating_cnt'  : 0 }        # counter tracking how the last id and the presented id differed consecutively


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-x", "--url", required=False,
    help="url to submit image for face detection (default=http://127.0.0.1:5000/)")
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
ap.add_argument("-u", "--user", required=False, 
    help="username to be registered, only if cmd == register")
ap.add_argument('-c', "--cmd", required=False, metavar="{detect | register}", default="detect",
                    help='command to execute. Supported commands: detect to start face detection, register to add a new user')
args = vars(ap.parse_args())

cmd = args['cmd']
if cmd != "detect" and cmd != 'register':
    ap.print_help()
    exit()

if args["url"]:
    url = args["url"]

if cmd == "detect":
    url = url + "/detect"
else:
    if "user" not in args:
        print("Username required for command register")
        ap.print_help()
        exit()
    username = args["user"]
    url = url + "/user/" + username.replace(" ","_")


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=96)

current_id = None

if args["image"]:
    image = cv2.imread(args["image"])
    height, width = image.shape[:2]
    faceAligned = extract_face(fa, detector,height*width*0.05,image)
    resp, result = send_image(faceAligned,url)
    print("Name: %s , distance: %s" % (result['name'], result['dist']) )
    display_id(image, result["name"])
    cv2.imshow("Input", image)
    cv2.imshow("Face", faceAligned)
    key = cv2.waitKey(0)

else:
    video = cv2.VideoCapture(0)
    if video.isOpened():
        while True:
            check, frame = video.read()
        
            if check:
                image = frame
                height, width = image.shape[:2]
                faceAligned = extract_face(fa, detector,height*width*0.05,image)
                if len(faceAligned) > 0 and cmd == 'detect':
                    resp, result = send_image(faceAligned,url)
                    print("Name: %s , distance: %s" % (result['name'], result['dist']) )
                    current_id, tracker = determine_id(result, tracker)
                    display_id(frame, current_id)      
                    cv2.putText(frame, "REC", (width-55,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255), 2) 
                if cmd == 'register':
                    display_text(frame, "press <space> to capture, q to quit") 
                cv2.imshow("Input", frame)
                key = cv2.waitKey(20)
                if key == ord('q'):
                    break
                elif key == ord(' ') and cmd == 'register' and len(faceAligned) > 0:
                    resp, result = send_image(faceAligned,url)
                    if (resp.status == 200):
                        cv2.imshow('Registered image', faceAligned)
                        print("Image successful registered")
                    else:
                        print("Server returned error status %d" % resp.status)
 

            
