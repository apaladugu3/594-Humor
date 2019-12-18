import io
import os
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
max = 0
min = 0
#arr = ['negative0.csv']
arr = os.listdir('/home/axp1147/Humor/Anotherbert/new')
#word_tokenize accepts a string as an input, not a file. 
for f in arr:
  file1 = open('/home/axp1147/Humor/Anotherbert/new/'+f) 
  count = 0
  line = file1.read()# Use this to read file content as a stream: 
  lines = line.split('\n')
  if 'positive' in f: 
    appendFile = open('/home/axp1147/Humor/Anotherbert/positive.txt','a')
    appendFile.write(lines[len(lines)-2] + "\n")
    appendFile.close()
  else:
    appendFile = open('/home/axp1147/Humor/Anotherbert/negative.txt','a')
    appendFile.write(lines[len(lines)-2] + "\n")
    appendFile.close()
    

