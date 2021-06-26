# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from dotenv import dotenv_values

#adding new imports
import yagmail

#adding mailing function

"""
Note to self:
Google turns off sign in for less secure apps so this setting has to re enabled to use the sms feature with this script
https://support.google.com/mail/answer/7126229?p=BadCredentials&visit_id=637592815026254557-1661076137&rd=2#cantsignin&zippy=%2Ci-cant-sign-in-to-my-email-client
"""
config = dotenv_values(".env")

def mail_owner():
	detection_email = config['DETECTION_EMAIL']
	detection_pass = config['DETECTION_PASS']
	recipient_email = config['RECIPIENT_EMAIL']
	yag = yagmail.SMTP(detection_email, detection_pass)
	contents = [
		"There is an intruder in your home"
		]
	yag.send(recipient_email, 'Intruder Alert!!!!', contents)


#adding main structure
def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--detector", required=True, help="path to OpenCV's deep learning face detector")
	ap.add_argument("-m", "--embedding-model", required=True, help="path to OpenCV's deep learning face embedding model")
	ap.add_argument("-r", "--recognizer", required=True, help="path to model trained to recognize faces")
	ap.add_argument("-l", "--le", required=True, help="path to label encoder")

	#adding new arguments for human body detection
	ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-x", "--model", required=True, help="path to Caffe pre-trained model")

	ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
	IGNORE = set(["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "pottedplant", "sheep","sofa", "train", "tvmonitor"])

	#COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized face detector from disk
	print("[INFO] loading face detector...")
	protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
	modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
	# load our serialized face embedding model from disk
	print("[INFO] loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
	# load the actual face recognition model along with the label encoder
	recognizer = pickle.loads(open(args["recognizer"], "rb").read())
	le = pickle.loads(open(args["le"], "rb").read())

	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# initialize the video stream, then allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	# start the FPS throughput estimator
	fps = FPS().start()
	# loop over frames from the video file stream

	start_time = time.time()	

	while True:
		detection = False
		detection_me_in_frame = False
		# grab the frame from the threaded video stream
		frame = vs.read()
		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]
		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		net.setInput(blob)
		detector.setInput(imageBlob)
		detections = detector.forward()
		detection_body = net.forward()
	
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections
			if confidence > args["confidence"]:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue
				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()
				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				#change this to the owner folder name containing the owner images in the dataset dir
				if str(name) == "suhesna":
					detection_me_in_frame = True
				#print(name)
				#print(type(name))
				# draw the bounding box of the face along with the
				# associated probability
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				if detection_me_in_frame:
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
					cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
				
				else:
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						(0, 0, 255), 2)
					cv2.putText(frame, text, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		for i in np.arange(0, detection_body.shape[2]):

			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detection_body[0, 0, i, 2]
			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# `detections`
				idx = int(detection_body[0, 0, i, 1])
				# if the predicted class label is in the set of classes
				# we want to ignore then skip the detection
				if CLASSES[idx] in IGNORE:
					continue
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detection_body[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				# draw the prediction on the frame
				label = "Person in frame!! {}: {:.2f}%".format(CLASSES[idx], confidence * 100)
				if detection_me_in_frame:
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				else:
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				
				detection = True
				# if start_counter == 0:
				# 	mail_owner()
				# 	start_counter += 1
				# diff = start_time - time.time()
				# if diff > 5.0:
				# 	print("Inside second if")
				# 	mail_owner()
				# 	start_time = time.time()

				# show the output frame
		#muting this
		#cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		
		# if (time.time() - start_time) > 5:
		if (time.time() - start_time) > 10:
			if detection == True:
				if detection_me_in_frame == False:
					start_time = time.time()
					mail_owner()
		# if the `q` key was pressed, break from the loop
		# if key == ord("q"):
		# 	break
		# update the FPS counter
		fps.update()
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

if __name__ == "__main__":
	main()
	
# python recognize_video.py --detector ./face_detection_model --embedding-model ./nn4.small2.v1.t7 --recognizer ./output/recognizer.pickle --le ./output/le.pickle

# command to run this script
# python recognize_video.py --detector ./face_detection_model --embedding-model ./nn4.small2.v1.t7 --recognizer ./output/recognizer.pickle --le ./output/le.pickle -p ./MobileNetSSD_deploy.prototxt.txt -x ./MobileNetSSD_deploy.caffemodel