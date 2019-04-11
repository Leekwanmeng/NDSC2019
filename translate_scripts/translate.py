import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from io import StringIO
import time

df = pd.read_csv('./data/train_beauty_image.csv')
f = open('translated_titles.txt', 'w')
for i in range(df.shape[0] // 500):
	sub = df.iloc[500 * i : 500 * (i + 1)]
	l = open("lines.txt", "w")
	for index, row in sub.iterrows():
		l.write(row["title"] + "\n")
	l.close()
    # lines += len(open('splits/' + str(i) + '.txt', 'r').readlines())
	files = {'file': open('lines.txt', 'rb')}
	params = {'tl': 'en', 'sl': 'id', 'prev': '_t', 'js': 'y', 'ie': 'UTF_8', 'hl': 'en'}
	headers = {'Host': 'translate.googleusercontent.com', 'Referer': 'https://translate.google.com/?tr=f&hl=en', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}
	url = "https://translate.googleusercontent.com/translate_f"

	response = requests.post(url, files=files, data=params, headers=headers)
	soup = BeautifulSoup(response.content)
	f.write(soup.get_text())
	f.flush()
	print(len(soup.get_text().split("\n")))
	time.sleep(5)

if df.shape[0] % 500 > 0:
	sub = df.iloc[500 * (df.shape[0] // 500):]
	l = open("lines.txt", "w")
	for index, row in sub.iterrows():
		l.write(row["title"] + "\n")
	l.close()
    # lines += len(open('splits/' + str(i) + '.txt', 'r').readlines())
	files = {'file': open('lines.txt', 'rb')}
	params = {'tl': 'en', 'sl': 'id', 'prev': '_t', 'js': 'y', 'ie': 'UTF_8', 'hl': 'en'}
	headers = {'Host': 'translate.googleusercontent.com', 'Referer': 'https://translate.google.com/?tr=f&hl=en', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}
	url = "https://translate.googleusercontent.com/translate_f"

	response = requests.post(url, files=files, data=params, headers=headers)
	soup = BeautifulSoup(response.content)
	f.write(soup.get_text())
	f.flush()
	print(len(soup.get_text().split("\n")))
	
f.close()