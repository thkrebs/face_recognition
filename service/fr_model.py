from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

np.set_printoptions(threshold=np.nan)


class FaceRecognitionModel:
    """Face Recognition Model based on a pre-trained FaceNet model"""
    
    def __init__(self):
        FRmodel = faceRecoModel(input_shape=(3, 96, 96))
        FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
        load_weights_from_FaceNet(FRmodel)
        self.model = FRmodel 

    def setDatabase(self,database):
        self.database = database

    def verify(self, image, identity):
        """
        Function that verifies if the person on the "image_path" image is "identity".
        
        Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be registered in the database
        Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
        """
       
        # Compute the encoding for the image. Use img_to_encoding() see example above.
        encoding = self.img_to_encoding(image_path, self.model)
        
        # Compute distance with identity's image 
        dist = np.linalg.norm(self.database[identity] - encoding)
        
        # Open the door if dist < 0.7, else don't open
        if (dist < 0.7):
            print("It's " + str(identity) + ", welcome home!")
            door_open = True
        else:
            print("It's not " + str(identity) + ", please go away")
            door_open = False
            
        return dist, door_open

    def who_is_it(self, image):
        """
        Implements face recognition for the finding who is the person on the image_path image.
        
        Arguments:
        image_path -- path to an image
        
        Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
        """
        ## Compute the target "encoding" for the image. Use img_to_encoding() see example above. 
        encoding = self.img_to_encoding(image)
        
        ## Find the closest encoding
        # Initialize "min_dist" to a large value, say 100
        min_dist = 100
        
        # Loop over the database dictionary's names and encodings.
        # Not the most efficient approach. A more efficient implementation would require an appropriate datastructure to find the
        # closest encoding 
        for (name, db_enc) in self.database.items():
            
            # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
            dist = np.linalg.norm(db_enc - encoding)

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if (dist < min_dist):
                min_dist = dist
                identity = name
   
        if min_dist > 0.7:
            print("Not in the database.")
            identity = None
        else:
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))
            
        return min_dist, identity

    def img_to_encoding_from_path(self,image_path):
        img = cv2.imread(image_path, 1)
        return self.img_to_encoding(img)

    def img_to_encoding(self, image):
        img = image[...,::-1]
        img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
        x_train = np.array([img])
        embedding = self.model.predict_on_batch(x_train)
        return embedding
    
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    # Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    # subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss