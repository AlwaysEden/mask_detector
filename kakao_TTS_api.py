import time
import argparse
import requests
import pygame

# KaKao Speech Synthesis API
# input = name, output = make "warning.wav" in same directory
def kakao_Speech_Synthesis_api(name):
	file_name = 'warning.mpeg' #음성을 저장할 파일이름
	url = "https://kakaoi-newtone-openapi.kakao.com/v1/synthesize" #Kakao TTS api 사이트에서 가져온 내 URL과 KEY
	key = '3cc9239a60f93d908dabe029c94c2213'
  
  #xml형태로 보내기 위한 headers와 body
	headers = {
		"Content-Type": "application/xml",
		"Authorization": "KakaoAK " + key,
	}
	talk = "{}님, 올바른 마스크 착용 부탁드려요".format(name).encode('utf-8').decode('latin-1')
	data = '<speak><prosody rate="1.2" volume="loud">{}</prosody></speak>'.format(talk)

	res = requests.post(url, headers=headers, data=data) #Kakao TTS에 xml파일을 보내서 응답받기
	f= open(file_name, 'wb') #위에서 정하였던 file을 'wb'(이진 읽기모드)로 열기
	f.write(res.content) #응답내용 file에 저장
	f.close() #파일종료
	return file_name

if __name__ == "__main__": #다른 파일에서 이 파일을 import해서 사용했을 때는 __name__이 kakao_TTS_api이기 때문에 실행이 안되는데 이 파일내에서 자체적으로 실행시킨다면 __name__은 main이 될 것이고 이 코드가 실핼될 것이다.
	pygame.mixer.init()

	ap = argparse.ArgumentParser()
	ap.add_argument("-n", "--name", required=True, help="input name")
	args = vars(ap.parse_args())

	name = args["name"]
	file_name = kakao_Speech_Synthesis_api(name)
	time.sleep(1.5)
	pygame.mixer.music.load(file_name)
	pygame.mixer.music.play()
	while pygame.mixer.music.get_busy()==True:
		continue
	#sound = pygame.mixer.Sound(file_name)
	#sound.play()
