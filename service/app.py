# create user
# update user with a picture
# who is it

from flask import Flask,render_template, request, Response
import jsonpickle
import numpy as np
import cv2
from fr_model import *
from os import listdir
from os.path import isfile, join
import time
import re

app = Flask(__name__)

# register a new user
@app.route("/user/<username>",methods=['POST'])
def newUser(username):
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # TODO: Database need to be factored out
    database[username] = FRModel.img_to_encoding(img)
    FRModel.setDatabase(database)
    cv2.imwrite(mypath + '/' + username + '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),'name': str(username)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# detect face 
@app.route('/detect', methods=['POST'])
def detect():
    start = time.time()
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    dist, id = FRModel.who_is_it(img)
    match = re.search('(\w+)(_\d+)',str(id))
    if match:
        id = match.group(1)
    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),'name': id, 'dist': str(dist)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    print( time.time()-start)    
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
 
    FRModel = FaceRecognitionModel()
 
    # initialize database 
    # TODO: needs to be refactored into separate class
    database = {}
#    database["danielle"] = FRModel.img_to_encoding_from_path("images/danielle.png")
#    database["younes"] = FRModel.img_to_encoding_from_path("images/younes.jpg")
#    database["tian"] = FRModel.img_to_encoding_from_path("images/tian.jpg")
#    database["andrew"] = FRModel.img_to_encoding_from_path("images/andrew.jpg")
#    database["kian"] = FRModel.img_to_encoding_from_path("images/kian.jpg")
#    database["dan"] = FRModel.img_to_encoding_from_path("images/dan.jpg")
#    database["sebastiano"] = FRModel.img_to_encoding_from_path("images/sebastiano.jpg")
#    database["bertrand"] = FRModel.img_to_encoding_from_path("images/bertrand.jpg")
#    database["kevin"] = FRModel.img_to_encoding_from_path("images/kevin.jpg")
#    database["felix"] = FRModel.img_to_encoding_from_path("images/felix.jpg")
#    database["benoit"] = FRModel.img_to_encoding_from_path("images/benoit.jpg")
#    database["arnaud"] = FRModel.img_to_encoding_from_path("images/arnaud.jpg")
    mypath='db'
    for f in listdir(mypath):
        if (isfile(join(mypath, f))):
            print(str(f))
            person_id = os.path.splitext(f)[0]
            print("Loading file for %s" % str(person_id))
            database[person_id] = FRModel.img_to_encoding_from_path(join(mypath,f))
 
    FRModel.setDatabase(database)

    # TODO: init and load the model
    # app.run(host='127.0.0.1')
    # Accept connections from everywhere
    app.run(host='0.0.0.0')