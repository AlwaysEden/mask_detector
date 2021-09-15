import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import subprocess

#Opencv의 DNN을 통하여 facenet(얼굴추출) 객체 읽어오기
#마스크 착용 유무를 판별하는 학습된 모델 저장
facenet = cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('model.h5')

#라즈베리파이의 카메라 사용
cap = cv2.VideoCapture(0)

#video save
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('video.avi',fourcc,5,(640,480))



flag = 0 #with mask state = 0, without mask state = 1 

if not cap.isOpened():

        print("Could not open cam")

        exit()
# loop through frames

while cap.isOpened():
	faces = []
	locs = []
	ret, frame = cap.read()
	#카메라 상하좌우 반전
	frame = cv2.flip(frame,-1)
	if not ret: #만약 프레임의 ret이 false라면 루프 중지
		print("Could not read frame")
		exit()

	h, w = frame.shape[:2] #frame의 세로 가로 길이 할당
 	blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.)) #blob을 통해서 프레임 전처리 진행
	facenet.setInput(blob) #facenet에 전처리된 blob을 input
	detections = facenet.forward() #facenet의 추론 시작
	confidence = detections[0, 0, 0, 2] #얼굴 하나의 정확도를 confidence에 대입

	if confidence > 0.5: #조건 : 정확도 50% 이상
		box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
		face = frame[startY:endY, startX:endX] # facenet을 통해 읽은 얼굴인 dectections의 왼쪽 위 가장자리 좌표와 오른쪽 아래 가장자리 좌표를 가져옴
		if face.any(): #조건 : 얼굴이 있을 시
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) #OpenCV는 BGR 포멧을 사용하기 때문에 우리에게 익숙한 얼굴 값을 가져오기 위해서는 RGB로 전환시켜주어야한다.
			face = cv2.resize(face, (224, 224)) #얼굴을 추출해냈다면 그 얼굴 이미지를 224 x 224로 만들어준다. 제각각 다른 얼굴 사이즈를 가져온다면 추후 predict할때 문제가 생길 수도 있다.
			face = img_to_array(face) #RGB값으로 존재하는 face 변수를 array 자료형으로 바꿔줌으로써 처리를 더욱 수월하게 만들어줌.
			face = preprocess_input(face) #preprocess_input 함수가 존재하는 ResNet50은 50개 계층으로 구성된 컨벌루션 신경망이다. 이 함수는 이미지배치를 인코딩하는 tensor 또는 Numpy 배열을 전처리한다.
			faces.append(face)
			locs.append((startX, startY, endX, endY))
			faces = np.array(faces, dtype="float32") #faces변수를 Numpy array 자료형으로 변환시킨다.
			preds = model.predict(faces, batch_size=32) #예측시작, 이 모델은 0~100%를 예측하는 모델인데, 100%에 가까울수록 no mask이고 0에 가까울수록 mask를 쓴 상태이다.

		for (box, no_mask) in zip(locs, preds): #얼굴의 위치 값 locs와 예측값 preds를 zip으로 묶어서 box와 no_mask 변수에 대입
			(startX, startY, endX, endY) = box #box 변수는 원소가 4개이기에 각각 변수에 대입할 수 있다.
			if no_mask > 0.6: #조건 : 마스크를 쓰고 있지 않다는 preds가 0.6 이상일 시
				if flag == 0: #조건 : 마스크를 쓰고 있는 상태일때
					cv2.imwrite('find.jpg',frame) #현재 돌아가고있는 프레임 저장
					name = subprocess.check_output("python3 naver_OCR_api.py -i find.jpg", shell = True) #저장한 프레임과 Naver_OCR.py를 main.py에서 실행시킨다.
					p = subprocess.Popen(['python3','kakao_TTS_api.py','-n',name]) #Naver_OCR에서 받아온 name 값을 가지고 Kakao_TTS과 같이 실행시킨다.
				flag = 1 #마스크를 안쓰고 있는 상태일 때
				color = (0,0,255)
				label = "No Mask ({:.2f}%)".format(no_mask[0]*100)
				cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color,2)

			else:
				flag = 0
				color = (0,255,0)
				label = "Mask ({:.2f}%)".format( (1-no_mask[0]) * 100)
				cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow('mask',frame)
	out_video.write(frame)
	if cv2.waitKey(1) & 0xFF == 27:
		break

cap.release()
out_video.release()
cv2.destroyAllWindows()
