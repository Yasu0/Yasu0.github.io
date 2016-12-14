import tensorflow as tf # tensorflow에 내장된 함수들을 불러옴.
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784]) # 계산할 때 입력할 값. 각 이미지는 784차원의 벡터로 단조화됨.
W = tf.Variable(tf.zeros([784, 10])) # 초기값을 만들고 0으로 채워진 텐서들로 초기화
b = tf.Variable(tf.zeros([10])) # 초기값을 만들고 0으로 채워진 텐서들로 초기화
y = tf.nn.softmax(tf.matmul(x, W) + b) # x와 W를 곱하고 b를 더한 뒤 tf.nn.softmax 적용

y_ = tf.placeholder(tf.float32, [None, 10]) # 교차 엔트로피를 구현하기 위해 새로운 정답을 입력할 placeholder

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # 교차 엔트로피, y의 로그값을 구하고 y_의 원소들과 곱한 뒤 텐서의 모든 원소를 더함.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 학습도를 0.01로 준 경사 하강법 알고리즘으로 교차 엔트로피 최소화 

# Session
init = tf.initialize_all_variables() # 실행 전 만들었던 변수 초기화

sess = tf.Session()
sess.run(init)

# Learning
for i in range(1000): # 학습을 1000번 시킴.
  batch_xs, batch_ys = mnist.train.next_batch(100) # 학습세트로 부터 100개의 무작위 데이터를 일괄처리
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # placeholders를 대체하기 위한 일괄 처리 데이터에 train_step 피딩

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # 맞는 라벨을 예측했는지 확인
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 부정 소수점으로 캐스팅한 후 평균값 구하기.

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # 프린트하여 정확도 확인