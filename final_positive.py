import io
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
var = 'positive'
var1 = 'output.json'
stop_words = set(stopwords.words('english')) 
file1 = open(var1) 
count = 0
temp1 = ''
temp2 = ''
lines = ''
    
def filewrite(string, count):
  f = ''
  check = string.split('"values": [')
  next = check[1].split('"token": "')
  next1 = next[1].split('", "layers":')
  f = f + next1[0]
  for c in check:
    if not (c == check[0] or c == check[1] or c == check[len(check)-2] or c == check[len(check)-1]):
      next = c.split('"token": "')
      next1 = next[1].split('", "layers":')
      f = f + ' ' + next1[0]
      new = next[0].split(']}]}')
      appendFile = open('/home/axp1147/Humor/Anotherbert/new/'+ var + str(count) + '.csv','a')
      appendFile.write(new[0]+ "\n")
      appendFile.close()
  next = check[len(check)-2].split('"token": "')
  next1 = next[1].split('", "layers":')
  new = next[0].split(']}]}')
  appendFile = open('/home/axp1147/Humor/Anotherbert/new/'+ var+ str(count) + '.csv','a')
  appendFile.write(new[0]+ "\n")
  appendFile.close()
  appendFile = open('/home/axp1147/Humor/Anotherbert/new/'+ var+ str(count) + '.csv','a')
  appendFile.write(f + "\n")
  appendFile.close()
  count += 1
  return count;
  
  
  
#word_tokenize accepts a string as an input, not a file. 

#line = file1.read()# Use this to read file content as a stream: 
#lines = line.split('{"linex_index":')
for line in file1:
  if '{"linex_index":' in line:
    temp1 = line.split('{"linex_index":')
    lines = lines + temp1[0]
    if(lines != ''):
      count = filewrite(lines, count)
    lines = temp1[1]
  else:
    lines = lines + line
    
