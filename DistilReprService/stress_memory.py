import requests
import threading
import os
from time import sleep
import random
from random_words import RandomWords
rw = RandomWords()

exitapp=False

def doGet():
	while not exitapp:
		try:
			text = ""
			for i in range(100):
				text += " " + rw.random_word()
			#print(text)		
			requests.post("http://localhost:9820/", json={"title":"title", "id":"0", "text": text})
		except Exception as e:
			print(e)

			
def main():
	while True:
		doGet()
	
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exitapp=True
