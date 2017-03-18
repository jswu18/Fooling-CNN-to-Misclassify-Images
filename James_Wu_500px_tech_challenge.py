###################################################################################################
###################################################################################################
#500px Machine Learning Engineer Intern - Tech Challenge, 2017
#Submission by James Wu
#Email: js.wu@mail.utoronto.ca
#3rd year Engineering Science Robotics student at the University of Toronto

#Final output image of original, noise, and adversarial images are stored as "final_image" variable

#Ideas from:
#http://karpathy.github.io/2015/03/30/breaking-convnets/

#Using code from:
#https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/#deep-mnist-for-experts
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
###################################################################################################
###################################################################################################
from pylab import *
import numpy as np
from scipy import*
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

###################################################################################################
###################################################################################################
#TRAINING NEURAL NET
#Code used from:
#https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/#deep-mnist-for-experts
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py
#Creates the classifier model
def train_neural_net(images_dictionary, test_images, test_labels, iterations):
    '''
    Function: trains a neural net to classify image vectors as a digit between 0 and 9
    Input:
        images_dictionary: mnist dictionary flattened image vectors
        test_images: flattened image vector of images of twos which will later be used to generate adversarial images (10x784)
        test_labels: actual classification for the image (10x10)
        iterations: number of iterations to take during training (int)
    Output:
        weights: list of weights and biases for the trained neural net
    '''
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    print ("Training Neural Net")
    for i in range(iterations):
      batch = images_dictionary.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    #test neural net on images of twos which will later be used to generate adversarial images
    #this should output an accuracy of 100% because the test labels are the actual labels of the images
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: test_images, y_: test_labels, keep_prob: 1.0}))

    weights = sess.run([W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2])
    return weights


###################################################################################################
###################################################################################################
#MODIFYING IMAGES TO "FOOL" NEURAL NET
#The neural net training code above was modified
#Weights were held constant while the input images were defined as variables
#Performing cost minimization instead changes the input images
def generate_adversarial_image(original_image, desired_label, weights):
    '''
    Function: generates an adversarial image of the original image which the neural net will incorrectly classify
    Input:
        original_image: flattened image vector of original image (1x784)
        desired_label: desired classification for the image (1x10)
        weights: weights of neural net (list of weights)
    Output:
        adversarial_image: generated adversarial image which will "fool" the neural net to incorrectly classify
    '''

    #define input images as a variable which can be adjusted during cost minimization
    x_images = tf.Variable(original_image)
    x_images = tf.cast(x_images, tf.float32)
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x_images, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])


    def conv2d(x_images, W):
      return tf.nn.conv2d(x_images, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2_n32(x_images):
      return tf.nn.max_pool(x_images, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    #define weights and biases as constants which cannot be changed during cost minimization
    W_conv1 = tf.constant(weights[0])
    b_conv1 = tf.constant(weights[1])

    x_image = tf.reshape(x_images, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2_n32(h_conv1)

    W_conv2 = tf.constant(weights[2])
    b_conv2 = tf.constant(weights[3])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2_n32(h_conv2)

    W_fc1 = tf.constant(weights[4])
    b_fc1 = tf.constant(weights[5])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.constant(weights[6])
    b_fc2 = tf.constant(weights[7])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    print ("Generating Adversarial Image...")
    for i in range(10000):
        train_accuracy = accuracy.eval(feed_dict={
        y_: desired_label, keep_prob: 1.0})
        if train_accuracy > 0.9: #check if neural net is "fooled" to classify two as six
            break
        train_step.run(feed_dict={y_: desired_label, keep_prob: 0.5})
    print ("Number of Iterations Used to Generate Adversarial Image:", i)
    adversarial_image = sess.run(x_images)
    return adversarial_image


###################################################################################################
###################################################################################################
#GETTING TEST IMAGES AND OTHER FUNCTIONS FOR MODIFYING IMAGES
def get_adversarial_image(original_image, wrong_label, weights):
    '''
    Function: Generates an adversarial image which the neural net will incorrectly classify
    Inputs:
        original_image: flattened image vector (1x784)
        wrong_label: incorrect desired label vector for the image (1x10)
        weights: weights for the neural network
    Output:
        adversarial_image: adversarial image that the neural net will incorrectly identify
    '''
    image = np.zeros((1,784))
    image[0,:] = original_image
    desired_label =  np.zeros((1,10))
    desired_label[0,:] = wrong_label
    #call function which generates adversarial image
    adversarial_image = generate_adversarial_image(image, desired_label, weights)
    return adversarial_image

def get_original_images(images_dictionary):
    '''
    Function: creates matrix of flattened image vectors of "twos" and label matricies
    Input:
        images_dictionary: mnist dictionary of flattened image vectors
    Output:
        original_images: matrix of flattened images of "twos" (10x784)
        correct_labels: label vector that labels images as "two" (10x10)
        wrong_labels: label vector that labels images as "six" (10x10)
    '''
    original_images = np.zeros((10,784))
    correct_labels = np.zeros((10,10))
    wrong_labels = np.zeros((10,10))
    correct_labels[:,2] = np.ones((1,10))
    wrong_labels[:,6] = np.ones((1,10))
    j = 0
    for i in range(len(images_dictionary.test.images)): #iterate through dictionary of images
        if j == 10:
            break #we have aquired 10 images
        if images_dictionary.test.labels[i][2] == 1:
            original_images[j,:] = images_dictionary.test.images[i]
            j +=1
    return original_images, correct_labels, wrong_labels

def reshape_images(image_matrix):
    '''
    Function: reshapes images to a 28x28 matrix which can be saved as an image
    Input:
        image_matrix: matrix of flattened image vectors (10x784)
    Output:
        images: matrix of image matricies  (10x28x28)
    '''
    images = []
    for i in range(len(image_matrix)):
        images.append(image_matrix[i].reshape((28,28)))
    return images

def save_images(original_images, noise, adversarial_images):
    '''
    Function: saves images on computer for visualization
    '''
    adversarial_images = reshape_images(adversarial_images)
    noise = reshape_images(noise)
    original_images = reshape_images(original_images)
    for i in range(10):
        original_image_name = str(i)+"th_original_image.png"
        noise_name = str(i)+"th_noise_image.png"
        adversarial_image_name = str(i)+"th_adversarial_image.png"
        imsave(original_image_name, original_images[i], cmap=plt.cm.gray)
        imsave(noise_name, noise[i], cmap=plt.cm.gray)
        imsave(adversarial_image_name, adversarial_images[i], cmap=plt.cm.gray)
    return

def reorder_images(original_images, noise, adversarial_images):
    '''
    Function: returns matrix of original images, noise, and adversarial image in desired format
    '''
    final_image = np.zeros((10,3,784))
    final_image[:,0,:] = original_images
    final_image[:,1,:] = noise
    final_image[:,2,:] = adversarial_images
    return final_image


###################################################################################################
###################################################################################################
if __name__ == "__main__":
    num_train_iterations = 10000
    adversarial_images = np.zeros((10, 784))

    #get dictionary of images
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    #get test images of twos
    original_images, correct_labels, wrong_labels = get_original_images(mnist)

    #train neural network, saving the weights
    weights = train_neural_net(mnist, original_images, correct_labels, num_train_iterations)

    #generate an adversarial image for each test image
    for i in range(10):
        adversarial_images[i] = get_adversarial_image(original_images[i], wrong_labels[i],weights)

    #subtract orignal from adversarial image for noise added to generate adversarial
    noise = adversarial_images-original_images

    #generate final format of desired images
    final_image = reorder_images(original_images, noise, adversarial_images)

###################################################################################################
    #uncomment if a visualization of the images are desired, images will be saved on computer
    save_images(original_images, noise, adversarial_images)
