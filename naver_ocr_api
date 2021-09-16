import argparse
import json
import base64
import requests

# Naver Clova OCR API
#input filename, output korean text
def naver_OCR_api(frame):
	with open(frame, "rb") as f: #main.py에서 돌고있는 Frame을 가져와서 'rd'모드 파일로 열어준다.
		img= base64.b64encode(f.read()) #아래에서 decode하기 위해서는 여기서 encode를 해줘야한다.
		
		#URL와 KEY는 Naver Clova OCR API를 사용하고자 할 때 웹사이트에서 알려준다.
		URL = "your url"
		KEY = "your key"

		#JSON형태는 데이터를 송수신할 때 주로 쓰이는데 headers와 body부분으로 나뉜다. 이 형태는 인터넷에서 찾을 수 있고 원하는 방식으로 수정시켜주면된다.
		headers = {
			"Content-Type": "application/json",
			"X-OCR-SECRET": KEY
		}

		data = {
			"version": "V1",
			"requestId": "test",
			"timestamp": 0,
			"images": [
				{
				"name": "127777",
				"format": "jpg",
				"data": img.decode('utf-8')
				}
			]
		}
		data = json.dumps(data) #python 객체를 json으로 변환시켜주기 위해서는 dumps()함수를 사용해야한다고 한다.
		response = requests.post(URL, data=data, headers=headers) #변화시켜준 data를 requests.post()를 통하여 URL에 요청하고 응답값을 받는다.
		res = json.loads(response.text) #response는 json형태일테니 다시 python객체로 만들어주어야한다.
		name = ''
		temp = []
		for dic in res['images' ][0]['fields'] : #한글을 제외한 문자들을 삭제시키는 과정
			name = name + dic['inferText']
			temp = re.compile('[가-힣]+').findall(name)
			name = ''.join(temp)
	return name


if __name__ ==  "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="image file")
	args = vars(ap.parse_args())

	img = args["image"]
	name = naver_OCR_api(img)
	print(name)
