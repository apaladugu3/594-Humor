import io
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
var = 'positive'
var1 = 'output.json'
#word_tokenize accepts a string as an input, not a file. 
stop_words = set(stopwords.words('english')) 
file1 = open(var1) 
count = 0
line = file1.read()# Use this to read file content as a stream: 
lines = line.split('{"linex_index":')
for l in lines:
	f = ''
	if l != lines[0]:
		check = l.split('"values": [')
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
