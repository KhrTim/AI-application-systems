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
![Computational graph](images/1.png)

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
