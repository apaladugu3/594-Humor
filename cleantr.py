import io
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
var1 = 'train.csv'
var = 'rand'
#word_tokenize accepts a string as an input, not a file. 
stop_words = set(stopwords.words('english')) 
file1 = open(var1) 
accuracy = 0.0
loss = 0.0
count = 0
appendFile = open('/home/axp1147/Humor/Anotherbert/grapht.csv','a')
line = file1.read()# Use this to read file content as a stream: 
lines = line.split('\n')
for l in lines:
  next = l.split(' ')
  loss = loss + float(next[1])
  accuracy = accuracy + float(next[2])
  if(int(next[0])%100 == 0):
    appendFile.write(next[0] + "," + str(loss/100) + "," + str(accuracy/100) + "\n")
    print (loss/100)
    print (accuracy/100)
    print (next[0])
    loss=0.0
    accuracy=0.0
appendFile.close()
  
    	