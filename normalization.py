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
  for l in lines:
    if not (l == lines[len(lines)-1] or l==lines[len(lines)-2]):
      check = l.split(',')
      for c in check:
        if min > float(c):
          min = float(c)
        if max<float(c):
          max = float(c)
for f in arr:
  file1 = open('/home/axp1147/Humor/Anotherbert/new/'+f) 
  count = 0
  line = file1.read()# Use this to read file content as a stream: 
  lines = line.split('\n')
  appendFile = open('/home/axp1147/Humor/Anotherbert/normalized/' + f,'a')
  appendFile.write(lines[len(lines)-2] + "\n")
  appendFile.close()
  for l in lines:
    if not (l == lines[len(lines)-1] or l==lines[len(lines)-2]):
      check = l.split(',')
      for c in check:
        if c != check[len(check)-1]: 
          appendFile = open('/home/axp1147/Humor/Anotherbert/normalized/' + f,'a')
          appendFile.write(str((float(c)-min)/(max-min)) + ",")
          appendFile.close()
        else:
          appendFile = open('/home/axp1147/Humor/Anotherbert/normalized/' + f,'a')
          appendFile.write(str((float(c)-min)/(max-min)) + "\n")
          appendFile.close()
print(min)
print(max)
