import io
import numpy as np 
import tensorflow as tf
f = 'vocabcnn.txt'
file1 = open('/home/axp1147/Humor/Anotherbert/'+f)
line = file1.read()
lines = line.split('\n')
input_x = tf.placeholder(tf.int32, [None, 200], name="input_x")
W = tf.Variable( tf.random_uniform([len(lines)+1, 1024], -1.0, 1.0), name="W")

a = np.empty((len(lines)+1,1024 ),dtype=np.float32)
b = np.empty(len(lines)+1,dtype=np.int32)
for l in lines:
  if(l != lines[len(lines)-1]):
    check = l.split("*:*")
    a[int(check[0].strip())] = np.array(check[2].split(',')).astype(np.float)
    b[int(check[0].strip())] = int(check[0].strip())
emb = np.array(a)
tf_embedding = tf.constant(emb, dtype=tf.float32)
x=tf.nn.embedding_lookup(tf_embedding, input_x)
print(x)
print(x)


