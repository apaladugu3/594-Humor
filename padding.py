import io
import os
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
#arr = ['negative0.csv']
max = 0
min = 0
file1 = open('/home/axp1147/Humor/Anotherbert/positiveid.txt') 
line = file1.read()# Use this to read file content as a stream: 
lines = line.split('\n')
for l in lines:
  check = l.split(' ')
  if min > len(check):
    min = len(check)
  if max<len(check):
    max = len(check)  
file1 = open('/home/axp1147/Humor/Anotherbert/negativeid.txt') 
line = file1.read()# Use this to read file content as a stream: 
lines = line.split('\n')
for l in lines:
  check = l.split(' ')
  if min > len(check):
    min = len(check)
  if max<len(check):
    max = len(check)

file1 = open('/home/axp1147/Humor/Anotherbert/positiveid.txt') 
line = file1.read()# Use this to read file content as a stream: 
lines = line.split('\n')
for l in lines[0:len(lines)-1]:
  check = l.split(' ')
  for i in range (len(check), max):
    l = l + " " + str(0)
  appendFile = open('/home/axp1147/Humor/Anotherbert/positiveidpad.txt','a')
  appendFile.write(l + "\n")
  appendFile.close()


file1 = open('/home/axp1147/Humor/Anotherbert/negativeid.txt') 
line = file1.read()# Use this to read file content as a stream: 
lines = line.split('\n')
for l in lines:
  check = l.split(' ')
  for i in range (len(check), max):
    l = l + " " + str(0)
  appendFile = open('/home/axp1147/Humor/Anotherbert/negativeidpad.txt','a')
  appendFile.write(l + "\n")
  appendFile.close()

