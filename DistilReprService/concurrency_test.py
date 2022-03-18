import requests
import threading
import os
from time import sleep, time
import random
import urllib3
import json
from nltk.corpus import words
from random import sample


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

exitapp = False


def doGet():
    i = 0
    acc = 0
    while not exitapp:
        try:
            i += 1
            req = "http://localhost:8000"
            distil_body = {"id": str(i), "text": ' '.join(sample(words.words(), 50)), "title": "Razzia der türkischen Polizei sorgt für Schockwelle an der Börse", "main_repr": "True"}
            st_time = time()
            print(i, req, requests.post(req, data=json.dumps(distil_body), verify=False))
            acc += time() - st_time
            print(acc / i)
            print()
        except Exception as e:
            print(e)


def main():
    threads = []

    for i in range(32):
        thread = threading.Thread(target=doGet)
        thread.start()
        threads.append(thread)

    for threads in threads:
        thread.join()

    print("threads finished...exiting")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exitapp = True
        os.exit()