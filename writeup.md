# Traffic Sign Classification

In the context of this project, traffic sign classification is taking images of traffic signs and matching them to the information that they are trying to convey to human drivers. By using machine learning we are able use examples of traffic signs that have been pre-matched to their meaning and train a system to automatically identify new images.

## Dataset

The [Real-Time Computer Vision research group](http://www.ini.rub.de/research/groups/rtcv/index.html.en) at the [Institut für Neuroinformatik](http://www.ini.rub.de/index.html.en) provide a dataset of German traffic sign images labeled with their appropriate classification. This dataset is called the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=about) (GTSRB)

![image](https://cloud.githubusercontent.com/assets/712014/24229485/e0d7be46-0f36-11e7-9ff7-3d7d869e7b94.png)

### Summary

```python
# Load pickled data
import pickle

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

n_train = len(train["features"])
n_valid = len(valid["features"])
n_test = len(test["features"])
width, height = len(test["features"][0]), len(test["features"][0][0])
image_shape = (width, height)
n_classes = len(set(test["labels"]))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32)
    Number of classes = 43
    
### Visualization

The training set is the largest of the sets since it its the set of samples that will actually build the model. The test samples are never used for training the model and kept completely separate until the very end when checking the accuracy of the classifier.



![output_8_1](https://cloud.githubusercontent.com/assets/712014/24229748/64400c92-0f38-11e7-822c-f529fbcf4b44.png)

- Blue: Train
- Orange: Validation
- Green: Test
- Y-Axis: Number of samples
- X-Axis: Identifier/Index for the type of sign

## The Classifier

The goal of the classifier is to take images of traffic signs that it has never seen before and be able to accurately classify them. This was accomplished using machine learning by applying a large training data set to a convolutional neural network.

### Preprocessing

For the neural network to accept the dataset, some transformations needed to be applied to the image. I went with the simplest transformation: converting the image to grayscale. Since I was using an architecture adapted from the LeNet architecture, the initial shape of the sample needed to be a single channeled 32 by 32 image.

```python
import tensorflow as tf

features_placeholder = tf.placeholder(tf.float32, (None, height, width, None), name='features_placeholder')
features = tf.image.rgb_to_grayscale(features_placeholder)
```

### Architecture

I used the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) architecture as a starting point and added on top of it to produce better results. I found that if I added 1x1 convolution as the first layer it increased the model size enough to better capture the classifier. Adding the dropout operation also improved the performance by making the neural network more generalized.

**LeNet Architecture** (initial approach *Testing Accuracy: 0.879*)

| Layer Number  | Type  | Input Shape  | Output Shape  | Activation  |
|---|---|---|---|---|
| 1 | Convolutional (5x5) | 32x32x1 | 28x28x6 | Rectified Linear Unit (ReLU) |
| 2 | Pooling (2x2) | 28x28x6 | 14x14x6 | - |
| 3 | Convolutional (5x5) | 14x14x6 | 10x10x16 | ReLU |
| 4 | Pooling (2x2) | 10x10x16 | 5x5x16 | - |
| 5 | Fully-Connected | 120 | 84 | ReLU |
| 6 | Fully-Connected | 84 | 43 | ReLU |

### Solution

The LeNet architecture was choosen as a starting point since it is an effective architecture for classifying images in the MNIST dataset. By itself, the architecture was able to reach a 0.879 validation accuracy however was limited by a couple factors. The LeNet architecture was too small to model the traffic sign concept since the MNIST data has much less features than you would find in traffic sign images. By adding a 1x1 convolutional layer I was able to increase the size of the network to make it better at modeling the concept.

**Modified LeNet Architecture** (*Testing Accuracy: 0.941*)

| Layer Number  | Type  | Input Shape  | Output Shape  | Activation  |
|---|---|---|---|---|
| 1 | Convolutional (1x1) | 32x32x1 | 32x32x1 | Rectified Linear Unit (ReLU) |
| 2 | Convolutional (5x5) | 32x32x1 | 28x28x6 | Rectified Linear Unit (ReLU) |
| 3 | Pooling (2x2) | 28x28x6 | 14x14x6 | - |
| 4 | Convolutional (5x5) | 14x14x6 | 10x10x16 | ReLU |
| 5 | Pooling (2x2) | 10x10x16 | 5x5x16 | - |
| 6 | Fully-Connected + Dropout | 120 | 84 | ReLU |
| 7 | Fully-Connected + Dropout | 84 | 43 | ReLU |

### Training

```python
# The number of passes the training set is ran through the neural network
EPOCHS = 20
# The number of samples ran through the neural network at one time
BATCH_SIZE = 256
# The constant used in the generating the weight delta during back-propagation
LEARNING_RATE = 0.001
# The probability that a neuron will be zero'd out in the fully connected layers
DROPOUT = 0.60
```

For training the neural network, I used the [AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) provided by TensorFlow. As for the hyper-parameters, they were chosen by experimenting with different values.

The number of epochs were increased because using dropout required more iterations to create a more generalized model. Batch size is typically memory limited so if more memory is available, increasing the batch size is beneficial for training the network. Since I am using dropout on one of the layers, a new hyper parameter is introduced.

## Outside the Dataset

As an exercise of showing the effectiveness of the classifier on data outside the dataset, I found several images of German traffic signs on the internet and ran them through the classifier.

### New Images

I used a couple approaches to grab traffic sign images. I first started with grabbing the top images off of Google for the phrase "German traffic signs". This produced some SVG based images which are less realistic but an interesting test case.

The other approach I used was taking advantage of Google Street View to produce images in a real life setting. I did a screenshot and some quick processing to get it in a format that was suitable for the classifier.

Although the new images had were able to be classified successfully, I could see cases where images could not be classified correctly. Since the dataset has under-represented labels, I could see images in those classes that have unusual features (blurry, occluded, rotated) being classified incorrectly.

![image](https://cloud.githubusercontent.com/assets/712014/24281902/5f79b91a-1019-11e7-9b53-d0668726537c.png)

### Performance

```python
# Grab the largest logit, the index corresponds to the image's predicted class
classify_operation = tf.argmax(logits, 1)
new_y = np.zeros(n_classes)

new_x = [plt.imread(filename) for filename in filenames]

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    # Dropout probability is set to 1 to make the neural networks behavior deterministic
    index = sess.run(classify_operation, feed_dict={features_placeholder: new_x, logits_placeholder: new_y, dropout_prob: 1.0})
    print(filenames)
    print(index)
```

Running the images through the classifier worked well with a perfect score of classifying the images. Considering the test set was reporting a 0.94 accuracy, this didn't appear to be too unusual.

#### Certainty

```python

# Grab the top 3 probabilities produced by running the logits through a softmax function
top_operation = tf.nn.top_k(tf.nn.softmax(logits), k=3)
new_y = np.zeros(n_classes)

new_x = [plt.imread(filename) for filename in filenames]

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    top = sess.run(top_operation, feed_dict={features_placeholder: new_x, logits_placeholder: new_y, dropout_prob: 1.0})

    print(top)
```

**Output**
```
[
       [  1.00000000e+00,   2.83870527e-08,   1.58895819e-09],
       [  9.74874496e-01,   1.66735873e-02,   6.44520996e-03],
       [  1.00000000e+00,   5.54841106e-09,   1.71303116e-09],
       [  9.99998569e-01,   7.89253022e-07,   3.30958187e-07],
       [  9.99987125e-01,   6.46488252e-06,   4.42701048e-06]
]
```

The top probabilities for each image were surprisingly high. Some of the probabilities were essentially 1.00 indicating that the classifier was highly confident that the provided image belong to the corresponding class.
