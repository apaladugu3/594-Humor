import io
import os
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
count = 1
#arr = ['negative0.csv']
arr = os.listdir('/home/axp1147/Humor/Anotherbert/new')
#word_tokenize accepts a string as an input, not a file. 
for f in arr:
  file1 = open('/home/axp1147/Humor/Anotherbert/new/'+f) 
  line = file1.read()# Use this to read file content as a stream: 
  lines = line.split('\n')
  sent = ''
  if 'positive' in f: 
    l = lines[len(lines)-2].split()
    for i in range(0, len(lines)-3):
      sent = sent + str(count) + " "
      appendFile = open('/home/axp1147/Humor/Anotherbert/vocabcnn.txt','a')
      appendFile.write(str(count) + "*:*" + l[i] + "*:*"+ lines[i] + "\n")
      appendFile.close()
      count+=1
    sent = sent + str(count)
    appendFile = open('/home/axp1147/Humor/Anotherbert/vocabcnn.txt','a')
    appendFile.write(str(count) + "*:*" + l[len(lines)-3] + "*:*" + lines[len(lines)-3] + "\n" )
    appendFile.close()
    count+=1
    appendFile = open('/home/axp1147/Humor/Anotherbert/positiveid.txt','a')
    appendFile.write(str(sent) + "\n")
    appendFile.close()
    
  else:
    l = lines[len(lines)-2].split()
    for i in range(0, len(lines)-3):
      sent = sent + str(count) + " "
      appendFile = open('/home/axp1147/Humor/Anotherbert/vocabcnn.txt','a')
      appendFile.write(str(count) + "*:*" + l[i] + "*:*"+ lines[i] + "\n")
      appendFile.close()
      count+=1
    sent = sent + str(count)
    appendFile = open('/home/axp1147/Humor/Anotherbert/vocabcnn.txt','a')
    appendFile.write(str(count) + "*:*" + l[len(lines)-3] + "*:*" + lines[len(lines)-3] + "\n")
    appendFile.close()
    count+=1
    appendFile = open('/home/axp1147/Humor/Anotherbert/negativeid.txt','a')
    appendFile.write(str(sent) + "\n")
    appendFile.close()
    
