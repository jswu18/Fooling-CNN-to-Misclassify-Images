# Fooling-CNN-to-Misclassify-Images
# 500px Machine Learning Engineer Intern - Tech Challenge, 2017
# Submission by James Wu
# Email: js.wu@mail.utoronto.ca
# 3rd year Engineering Science Robotics student at the University of Toronto

# Training a Neural Network in tensorflow on mnist data to classify images as a digit between 0 and 9. 
# The program then generating Adversary images from images of "two" digit that the neural network will incorrectly identify as "six".
# This was achieved by adjusting the original image instead of the weights of the neural net using cost minimization with a target classification of "six" instead of "two". 
# The final adversary images generated "fool" the neural net to classify them as "six" instead of "two". 
# Upon visualization of the adversary images, it is clear that they still display a picture of a two.
