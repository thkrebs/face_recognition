# Face Recognition

Demonstrator and playground for experiments with OpenCV and deep learning based face recognition.

The server uses a pre-trained deep neural network which was trained using the triplet loss function.
The client uses OpenCV and dlib to capture the video stream from the integrated WebCam, detect and align faces.

![Architecture Overview][(https://github.com/thkrebs/face_recognition/blob/master/fr_architecture.png "Architecture Overview")]

You have three options to run the backend service:
- directly in your installed python interpreter
- as docker image
- in kubernetes on minikube or e.g. in the Google Cloud

## Run service in Python directly
In order to run the server directly through Python, install all pre-requisites from service\requirements.txt, and then

Run the backend service....
```console
$ cd service
$ python app.py
```

You should see the following output:

```console
Using TensorFlow backend.
2018-09-03 15:58:57.079567: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
thomas_2.jpg
Loading file for thomas_2
thomas_1.jpg
Loading file for thomas_1
 * Serving Flask app "app" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```
Depending on the machine you are running on loading the saved Keras model might take a couple of minutes.

## Run service in docker image
To run the backend service as docker image (I have tested and used docker version 18.05.)

```console
$ docker run -p 5000:5000 thkrebs/fr-service
```

You should see the same output as above.

## Run service in Minikube

Install minikube on your system. I have used version v0.25.0
Start minikube...

```console
$ minikube start
```

After minikube has been started

```console
$ cd <your project dir>/service/kubernetes
$ kubectl apply -f fr_deployment.yml
$ ./create_service.sh
```

You can reach the application on the IP address minikube is running on. 

```console
$ minikube ip
```

## Run service on Google cloud

Make sure you have setup an account and created a project on Google Cloud Platform.
Set your context (project, compute zone etc.) according to [Google Cloud Quickstart](https://cloud.google.com/kubernetes-engine/docs/quickstart)


If you have setup contexts already you can view your contexts with
```console
$ kubectl config get-contexts
```
Change your contexts
```console
$  kubectl config use-context <context name>
```

Create cluster and start service
```console
$ ./create_gcloud_cluster.sh
$ ./create_service.sh
```
Wait until the external IP for your service is available
```console
ubectl get service fr-svc
NAME      TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
fr-svc    LoadBalancer   10.23.254.250   35.242.198.62   5000:30307/TCP   5m
```

Check whether your service is reachable (make sure you waited long enough to let your image start up and load the Keras model which might take a couple of minutes)

```console
$ curl http://<external-ip of your service:5000/test
{"message": "test success"}
```

## Run client program

In order to run the client you need to install python 3 and the following core packages  
- OpenCV (-> pip install opencv-python)
- Dlib, e.g. follow the instructions [here](https://www.learnopencv.com/install-opencv-3-and-dlib-on-windows-python-only/)


Further you need to install the python packages imported in the client program `face_recognizer.py`

The programm accesses the integrated webcam through OpenCV. I didn't test it with an external webcam, like a USB cam.

The client application supports basically two functions:
- add a face
- detect faces


## Adding a faces

You have basically two options to add known faces:
- run the client (See above)
- manually copying image to the folder service/db

If you copy images manually you need to make sure that the image is a JPG 96x96 pixels wide.

Naming conventions are:

`<name of the person>_<nr>.jpg`

You can have multiple images from the same person which are distinguished by the number suffix.
The name of the person recognized is extracted from the filename. I know that is pretty basic but works for the purpose.

In order to use the client program to add a face 

```console
$ cd client 
$ python face_recognizer.py --shape-predictor shape_predictor_68_face_landmarks.dat -x http://<url:port> -u <user-name> -c register
```
As URL use the url, the service is running on, if you run it - on Google Cloud use the external IP address of the service and port 5000
- on Minikube use the Minikube address and port 32000
- as Docker image or directly use the localhost and port 5000

For the username follow the naming conventions above without the file extension .jpg

Once the program has been started it opens two window, the videostream as accessed from the webcam and the small window which holds the detected face cropped and aligned.

Pressing space will send one frame to the server storing it under the provided username. The registered image is shown in a another small window. Once you are satisfied with the registered image you can leave the program by pressing q on the keyboard.

The face detection is sensitive to tilting of the face, so not to tilt your face.


Note: The simplistic storage mechanism is not cluster capable.


## Detecing faces

In order to start face detection

```console
$ cd client 
$ python face_recognizer.py --shape-predictor shape_predictor_68_face_landmarks.dat -x http://<url:port> -u <user-name> -c detect
```

As URL use the url, the service is running on, if you run it - on Google Cloud use the external IP address of the service and port 5000
- on Minikube use the Minikube address and port 32000
- as Docker image or directly use the localhost and port 5000


Once the program is running it shows the captured video stream and a small window with the cropped and aligned face.
Once your face fills 5% of the frame, it starts to extract faces and send them for detection to the service. This mode is signaled by displaying "REC" in the upper right corner.
