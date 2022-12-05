# AI-application-systems - Timur Khairulov (12224993)

# ![Week 3](Weekly_sessions/week3/week_3.ipynb "Go to code")
### Goals of week 3:
- [x] Learn how does a single neuron work
- [x] Learn how does a backpropagation algorithm work
- [x] Implement a simple neural network without frameworks
- [x] Train a network to solve the XOR problem

### XOR Problem

For solving a XOR problem the following structure of neural network was developed

![](images/6.png "Network")

It has 2 inputs, 1 hidden layer containing 2 neurons and 1 output neuron

### Backpropagation
The backpropagation algorithm consists of 3 steps:
1. forward_pass

```python
def forward_pass(x):
  """
    First step of back propagation algorithm
  """
  global n_y
  n_y[0] = np.tanh(np.dot(n_w[0],x))
  n_y[1] = np.tanh(np.dot(n_w[1],x))

  n2_inputs = np.array([1.0, n_y[0], n_y[1]])
  z2=np.dot(n_w[2], n2_inputs)
  n_y[2] = 1.0/(1.0 + np.exp(-z2))
 ```
 2. backward_pass
 
 ```python
 def backward_pass(y_truth):
  """
    Second step of back propagation algorithm
  """
  global n_error
  error_prime = -(y_truth - n_y[2])
  derivative = n_y[2] * (1.0 - n_y[2])
  n_error[2] = error_prime * derivative
  derivative = 1.0 - n_y[0]**2
  n_error[0] = n_w[2][1] * n_error[2] * derivative
  derivative = 1.0 - n_y[1]**2
  n_error[1] = n_w[2][2] * n_error[2] * derivative
```
3. adjusting weights
```python
def adjust_weights(x):
  """
    Process of weights adjusting
  """
  global n_w
  n_w[0] -= (x * LEARNING_RATE * n_error[0])
  n_w[1] -= (x * LEARNING_RATE * n_error[1])
  n2_inputs = np.array([1.0, n_y[0], n_y[1]]) # 1.0 is bias
  n_w[2] -= (n2_inputs * LEARNING_RATE * n_error[2])
```

### Result
Eventually, the neural network of 3 neurons learnt to solve the XOR problem



# ![Week 4](Weekly_sessions/week4/week_4.ipynb "Go to code")
### Goals of week 4:
- [x] Exploring the MNIST dataset

### Result
After downloading the dataset and unpacking it the 6-th element was tasken for observation.
```
label for first training example:  2
---beginning of pattern for first training example---
                            
                            
                            
                            
                            
               *****        
             *******        
           **********       
         ************       
         ******* ****       
         ******  ****       
         ****    ****       
                 ****       
             ********       
          **********        
        *************       
       ****** ********      
      ****** ************   
     ****** ***** *******   
     ***********     ****   
     **********             
     *********              
     ******                 
                            
                            
                            
                            
                            
---end of pattern for first training example---
```
Here we can see that the 6-th element in dataset is a handwritten '2'.


# ![Week 5](Weekly_sessions/week5/week_5.ipynb "Go to code")
### Goals of week 5:
- [x] Learn how to create variables in tensorflow
- [x] Learn how to create constants in tensorflow
- [x] Learn how to create placeholders in tensorflow
- [x] Learn how to work with the tesorboard
- [x] Learn how to create a computational graph in tensorflow
- [x] Learn how to use `tf.summary.scalar` and `tf.summary.histogram` commands

### Variables in tensorflow
Variables are tensors trainable during a session.
In tensotrflow they are defined using following syntax: `a = tf.Variable(0, name = 'input')`

### Constants in tensorflow
Constants are tensors that are NOT trainable during a session.
In tensotrflow they are defined using following syntax: `b = tf.constant(5)`

### Placeholders
Placeholders are tensors whose values are unknown during the graph construction but passed as input during a session.
They can be defined with this syntax: `x = tf.placeholder("float32", None)`

### Computational graph
Next, we define a computational graph:
```python
a = tf.add(1,2, name="addition_1")
b = tf.multiply(a,3, name="multiplying_1")
c = tf.add(4,5, name="addition_2")
d = tf.multiply(c,6, name="multiplying_2")
e = tf.multiply(4,5)
f = tf.div(c,6, name="division")

g = tf.add(b,d)
h = tf.multiply(g,f)
```
and then we can visualize it via the tensorboard.
Tensorboards are avaliable to be ran inside the cells of google colab.
For enabling inline tensorboard run `%load_ext tensorboard`
and `%tensorboard --logdir='./logs'` with specified logging folder used in program.\
The output of a tensorboard with computational graph looks like this:
![Computational graph](images/1.png "Computational graph")

## `tf.summary.scalar` and `tf.summary.histogram` commands
To understand how these commands work, let's run a following script:
```python
tf.reset_default_graph()
s_scalar = tf.get_variable(name='s_scalar',
                           shape=[],
                           initializer=tf.truncated_normal_initializer(
                               mean=0,
                               stddev=1
                           ))

y_matrix = tf.get_variable(name='y_matrix',
                           shape=[40,50],
                           initializer=tf.truncated_normal_initializer(
                               mean=0,
                               stddev=1
                           ))

# A scalar summary for scalar tensor
first_summary = tf.summary.scalar(name='First_summary', tensor = s_scalar)

# A histogram summary for non-scalar tensor 2-D and 3-D matrix
histogram_summary = tf.summary.histogram('histogram_summary', y_matrix)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  writer = tf.summary.FileWriter('./logs', sess.graph)

  for i in range(100):
    sess.run(init)
    summary1, summary2 = sess.run([first_summary, histogram_summary])
    writer.add_summary(summary1, i)
    writer.add_summary(summary2, i)
```
Now we can visualize results in tensorboard
![tf.summary.scalar](images/3.png "tf.summary.scalar")
![tf.summary.histogram](images/7.png "tf.summary.histogram")


# ![Week 6-1](Weekly_sessions/week6/week_6.ipynb "Go to code")
### Goals of week 6-1:
- [x] Perform the linear regression using tensorflow
- [x] Show the learning process of a linear regression model
- [x] Design and train neural network for MNIST dataset digit classification
- [x] Test learning parameters on an image from the web

### Linear regression
Let's make a model for linear regression problem and train it
```python
Y = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(), name = 'weights')
B = tf.Variable(np.random.randn(), name = 'bias')
```
```python
# Create model for regression
with tf.name_scope("Model") as scope:
  prediciton = W*X + B

# Add summary to study behavior of weights and biases with epochs
weight_histogram = tf.summary.histogram("Weights", W)
bias_histogram = tf.summary.histogram("Bias", B)

# Cost function
with tf.name_scope('Cost_function') as scope:
  cost_function = tf.reduce_sum((prediction-Y)**2 / (2*sample_points))

# Record the scalar summary of the cost funciton
cost_summary = tf.summary.scalar("Cost", cost_iteration)

# Define the optimizer
with tf.name_scope("Training") as scope:
  optimizer = tf \
              .train \
              .GradientDescentOptimizer(learning_parameter) \
              .minimize(cost_iteration)

# Initialize the variables
init = tf.global_variables_initializer()

# Merge all the summaries into a single operator
merged_summary = tf.summary.merge_all()
```
Result:
![Linear regression](images/8.png "Linear regression")
As we can see, model works and we received the desired output.
Red line represents the trained model and blue dots represent data.

### Neural Network for MNIST digit classification
At first, we need to define parameters of neural network
```python
n_input = 784 # Input image of size 28 x 28
n_hidden_1 = 512 # First hidden layer
n_hidden_2 = 256 # Second hidden layer
n_hidden_3 = 128 # Third hidden layer
n_output = 10 # Output layer having (0-9) digits

learning_rate = 1e-4
epochs = 3000
batch_size = 128
```
Next step is to connect layers and create the network
```python
nn_weight = {"W1" : tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
             "W2" : tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
             "W3" : tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
             "Wout" : tf.Variable(tf.truncated_normal([n_hidden_3, n_output]))
}

nn_bias = {"B1" : tf.Variable(tf.truncated_normal([n_hidden_1])),
           "B2" : tf.Variable(tf.truncated_normal([n_hidden_2])),
           "B3" : tf.Variable(tf.truncated_normal([n_hidden_3])),
           "B4" : tf.Variable(tf.truncated_normal([n_output]))
}

nn_layer_1 = tf.add(tf.matmul(X, nn_weight["W1"]), nn_bias["B1"])
nn_layer_2 = tf.add(tf.matmul(nn_layer_1, nn_weight["W2"]), nn_bias["B2"])
nn_layer_3 = tf.add(tf.matmul(nn_layer_2, nn_weight["W3"]), nn_bias["B3"])
layer_drop = tf.nn.dropout(nn_layer_3, keep_prob)
output_layer = tf.add(tf.matmul(layer_drop, nn_weight["Wout"]), nn_bias["B4"])
```

Choosing loss, prediction, optimizer and model accuracy
```python
# Define the loss
computed_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
              logits=output_layer, labels=Y)
)

# Define the optimizer
optimizer = tf\
            .train\
            .GradientDescentOptimizer(learning_rate = learning_rate)\
            .minimize(computed_loss)

# Define prediction
prediction_out = tf.equal(tf.argmax(output_layer,1), tf.argmax(Y,1))

# Define accuracy of model
nn_accuracy = tf.reduce_mean(tf.cast(prediction_out, tf.float32))
```
After training all parameters were saved to a file, so they could be used further in a new model
```
Optimization Finished
Testing accuracy is 0.8561999797821045
```

### Testing on an image
For the test, an image of '7' was downloaded from the web.\
Next, we can use previously saved weights to create a new model that will make a prediction.\
Result:\
![Result](images/5.png "Result")

# ![Week 6-2](Weekly_sessions/week6/week_6_2.ipynb "Go to code")
### Goals of week 6-2:
- [x] Explore the Fashion MNIST dataset
- [x] Perform image denoising unsing Neural Network

### Exploring the Fashion MNIST dataset
Dataset consists of low resolution pictures of clothes and accessories.\
Let's randomly pick an item from dataset and display it.\
![original image](images/13.png "original image")

Now let's prepare a noisy dataset, that Neural Network will try to clear out\

```python
X_train_noisy = X_train + 10*np.random.normal(0,1,size=X_train.shape)
```
Noisy image looks like this:\
![noisy image](images/14.png "noisy image")

### Building a model
Let's define parameters of a Neural Network
```python
n_input = 784 # Input image is of size 28 x 28 
hidden_layer_1 = 256
hidden_layer_2 = 32
hidden_layer_3 = 32
hidden_layer_4 = 256
output_layer = 784 # Same as the n_input dimension

# Parameters
learning_rate = 0.1
epochs = 100
batch_size = 100
```
Weights and Biases
```python
Weight_NN = {"W1" : tf.Variable(tf.random_normal([n_input, hidden_layer_1])),
             "W2" : tf.Variable(tf.random_normal([hidden_layer_1, hidden_layer_2])),
             "W3" : tf.Variable(tf.random_normal([hidden_layer_2, hidden_layer_3])),
             "W4" : tf.Variable(tf.random_normal([hidden_layer_3, hidden_layer_4])),
             "W5" : tf.Variable(tf.random_normal([hidden_layer_4, output_layer]))
}

Bias_NN = {"B1" : tf.Variable(tf.random_normal([hidden_layer_1])),
           "B2" : tf.Variable(tf.random_normal([hidden_layer_2])),
           "B3" : tf.Variable(tf.random_normal([hidden_layer_3])),
           "B4" : tf.Variable(tf.random_normal([hidden_layer_4])),
           "B5" : tf.Variable(tf.random_normal([output_layer]))
}
```
Now we can connect layers
```python
Z1 = tf.add(tf.matmul(X, Weight_NN["W1"]), Bias_NN["B1"])
Z1_out = tf.nn.sigmoid(Z1)

Z2 = tf.add(tf.matmul(Z1_out, Weight_NN["W2"]), Bias_NN["B2"])
Z2_out = tf.nn.sigmoid(Z2)

Z3 = tf.add(tf.matmul(Z2_out, Weight_NN["W3"]), Bias_NN["B3"])
Z3_out = tf.nn.sigmoid(Z3)

Z4 = tf.add(tf.matmul(Z3_out, Weight_NN["W4"]), Bias_NN["B4"])
Z4_out = tf.nn.sigmoid(Z4)

Z5 = tf.add(tf.matmul(Z4_out, Weight_NN["W5"]), Bias_NN["B5"])
Z5_out = tf.nn.sigmoid(Z5)

Z1 = tf.layers.dense(X, hidden_layer_1, activation = tf.nn.sigmoid)
Z2 = tf.layers.dense(Z1, hidden_layer_2, activation = tf.nn.sigmoid)
Z3 = tf.layers.dense(Z1, hidden_layer_3, activation = tf.nn.sigmoid)
Z4 = tf.layers.dense(Z1, hidden_layer_4, activation = tf.nn.sigmoid)
NN_output = tf.layers.dense(Z4, output_layer)
```
And define metrics
```python
# The loss funciton
computed_loss = tf.reduce_mean(tf.square(NN_output-Y))

# Define the optimizer
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(computed_loss)
```
Now Neural Network can be trained.
```
Epoch 99 / 100 loss 2750.8054
```
After training we can try denoising an image using prepared Neural Network and compare the results.\
Original image:\
![Original image](images/10.png "Original image")

Noisy image:\
![Noisy image](images/11.png "Noisy image")

Denoised image:\
![Denoised image](images/12.png "Denoised image")




# ![Week 7-1](Weekly_sessions/week7/week_7.ipynb "Go to code")
### Goals of week 7-1:
- [x] Exploring the CIFAR10 dataset

### Result
Let's import the dataset using following command: `from torchvision.datasets import CIFAR10`\
Now we can take a look at example data, stored in the dataset:\
![Boat](images/9.png "Boat")\
This image has category '8'

# ![Week 7-2](Weekly_sessions/week7/week_7_2.ipynb "Go to code")
### Goals of week 7-2:
- [x] Building a model for predicting category in the CIFAR10 dataset

### Result
Let's configure a model
```python
# Model with 2 convolutional and 1 fully-connected layer
model = nn.Sequential(
    nn.Conv2d(3, 64, 5, stride=2, padding=2), # Output is 64x16x16
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, stride=2, padding=1), # Output is 64x8x8
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 10)
)
```
```python
# Retrieve layers for custom weight initialization
layers = next(model.modules())
conv_layer0 = layers[0]
conv_layer1 = layers[2]
output_layer = layers[5]

# Kaiming (He) initialization
nn.init.kaiming_normal_(conv_layer0.weight)
nn.init.constant_(conv_layer0.bias, 0.0)
nn.init.kaiming_normal_(conv_layer0.weight)
nn.init.constant_(conv_layer1.bias, 0.0)

# Xavier (Glorot) initialization
nn.init.xavier_uniform_(output_layer.weight)
nn.init.constant_(output_layer.bias, 0.0)

# Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()
```
Training:
```python
# Train the model
train_model(model, device, EPOCHS, BATCH_SIZE, trainset, testset, optimizer, loss_function, 'acc')
```

Training result:
```python
Epoch 128/128 loss: 0.1304 - acc: 0.9665 - val_loss: 10.4689 - val_acc: 0.5475

[0.96649072296865, 0.5475239616613419]
```


# ![Week 9-1](Weekly_sessions/week9/Week_9.ipynb "Go to code")
### Goals of week 9-1:
- [x] Building a neural network using Keras Sequential and Keras Dense

### Result
Let's build a model:
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=[13]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="adam", 
              metrics=["mean_absolute_error"])
```

We will train and test our model on keras `boston_housing` dataset
```python
boston_housing = keras.datasets.boston_housing
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()
```
Model's results:
```python
Prediction:  6.04861 , true_value:  7.2
Prediction:  19.704561 , true_value:  18.8
Prediction:  20.227697 , true_value:  19.0
Prediction:  33.061592 , true_value:  27.0
```
# ![Week 9-2](Weekly_sessions/week9/Week_9_2.ipynb "Go to code")
### Goals of week 9-2:
- [x] Comparing different neural network architectures
- [x] Learning about dropout in neural network architectures

### Result
5 different neural network architectures were tested:
- Configuration 5
```python
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=[13]))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))

model.compile(loss="mean_squared_error", optimizer="adam", 
              metrics=["mean_absolute_error"])
```
- Configuration 4
```python
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=[13]))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

model.compile(loss="mean_squared_error", optimizer="adam", 
              metrics=["mean_absolute_error"])
```
- Configuration 3
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=[13]))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

model.compile(loss="mean_squared_error", optimizer="adam", 
              metrics=["mean_absolute_error"])
model.summary()
```
- Configuration 2
```python
model = Sequential()
model.add(Dense(64, activation='relu',
kernel_regularizer=l2(0.1),
bias_regularizer=l2(0.1),
input_shape=[13]))
model.add(Dense(64, activation='relu',
kernel_regularizer=l2(0.1),
bias_regularizer=l2(0.1)))
model.add(Dense(1, activation='linear',
kernel_regularizer=l2(0.1),
bias_regularizer=l2(0.1)))

model.compile(loss="mean_squared_error", optimizer="adam", 
              metrics=["mean_absolute_error"])
```
- Configuration 1
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=[13]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss="mean_squared_error", optimizer="adam", 
              metrics=["mean_absolute_error"])
```

All architectures were tested and trained on a keras `boston_housing` dataset
```python
boston_housing = keras.datasets.boston_housing
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()
```

Model evealations are given in the following table
|CONFIGURATION|TOPOLOGY|REGULARIZATION|TRAINING ERROR| TEST ERROR|
|:-------------:|:--------:|:--------------:|:--------------:|:-----------:|
|Conf1|64/64/1|None|0.76|12.8|
|Conf2|64/64/1|L2=0.1|8.9|20.6|
|Conf3|64/64/1|Dropout=0.2|10.0|12.6|
|Conf4|128/128/64/1|Dropout=0.2|7.3|13.0|
|Conf5|128/128/64/1|Dropout=0.3|11.3|13.0|

# ![Week 10-1](Weekly_sessions/week9/Week_10_1.ipynb "Go to code")
### Goals of week 10-1:
- [x] Working with keras Resnet50

### Result
As a sample image for working with resnet50 was choosen this picture of a dog:
-- picutre of a dog
At first we need to preprocess the chosen image
```python
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456,0.406], std=[0.229, 0.224, 0.225])
]) # Parameters recommended by pytorch.org
input_tensor = preprocess(image)
```
Then, we load pretrained model
```python
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()
```
Make the prediction
```python
inputs = inputs.to(device)
with torch.no_grad():
  outputs = model(inputs)
```
And print 5 most probable classes
```python
_, indices = torch.sort(probabilities, descending=True)
for i in range(0,5):
  print("Image class:", indices[i].item(), ", probability = %4.3f" % probabilities[indices[i]].item())
```
As a result we get the following data:
```
Image class: 232 , probability = 0.649
Image class: 231 , probability = 0.131
Image class: 200 , probability = 0.110
Image class: 193 , probability = 0.018
Image class: 199 , probability = 0.010
```

# ![Week 10-2](Weekly_sessions/week9/Week_10_2.ipynb "Go to code")
### Goals of week 10-2:
- [x] Working with keras Sequential
- [x] Working with keras API

### Result
Let's create a simple Sequential model...
```python
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```
..and build it
```python
model.build(input_shape=(None, 3))
```
It is possible to build the same configuration incrementally
```python
inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)
```
Let's take a look at the model that was built using `model.summary()` method
```
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 my_input (InputLayer)       [(None, 3)]               0         
                                                                 
 dense_6 (Dense)             (None, 64)                256       
                                                                 
 dense_7 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 906
Trainable params: 906
Non-trainable params: 0
```
# ![Week 11](Weekly_sessions/week9/Week_11.ipynb "Go to code")
### Goals of week 11:
- [x] Working with real data
- [x] Predicting bookstore monthliy sales  

### Result
At first let's explore given dataset
-- plot
Now, let's make a naive prediction by shifitng a given data
-- plot 2
Time to build and train a RNN
```python
# Custom layer that retrieves only last time step from RNN output.
class LastTimestep(nn.Module):
  def forward(self, inputs):
    return inputs[1][0]

# Create RNN model
model = nn.Sequential(
    nn.RNN(1,128, nonlinearity='relu', batch_first=True),
    LastTimestep(),
    nn.Linear(128, 1)
)

# Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.MSELoss()

# Train model
train_model(model, device, EPOCHS, BATCH_SIZE, trainset, testset, optimizer,
            loss_function, "mae")     
```
Now we can plot the prediciton of our RNN
-- plot
