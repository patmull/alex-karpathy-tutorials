import math
import random

import numpy as np
import matplotlib.pyplot as plt

from graph_visualisation import draw_dot


def easy_function(x):
    return 4 * x ** 2 + 5 * x + 1


x_values = np.arange(-10, 10, 0.2)
y_values = easy_function(x_values)
# plt.plot(x_values, y_values)
# plt.show()

# derivative of easy function

# by derivation definition
# now we need to select particular point
h = 0.001
x = 3.0
derivative = (easy_function(x + h) - easy_function(x)) / h
print(derivative)  # We go up

x = -3.0
derivative = (easy_function(x + h) - easy_function(x)) / h
print(derivative)  # We go down

# what happen when we change the h so smaller?
h = 0.00000001
x = 3.0

derivative = (easy_function(x + h) - easy_function(x)) / h
print(derivative)  # The derivative slightly converges to more precise solution

h = 0.000000000000000001
x = 3.0

derivative = (easy_function(x + h) - easy_function(x)) / h
print(derivative)  # Why is this zero?
# this is not mathematically correct, we need to be aware that we are using computers and computers are limited
# this is floating point arithmetics error

# How to calculate derivative of a with respect to d? (da/dd in Leibniz notation)?
a = 2.0
b = -3.0
c = 10.0
d = a * b + c
print(d)

# we just need to slightly increase every of the variables
values = {'a': 2.0, 'b': -3.0, 'c': 10.0}


def derivative(function_values, with_respect_to, h=0.0001):
    function = function_values['a'] * function_values['b'] + function_values['c']
    d1 = function
    # change
    function_values[with_respect_to] += h
    function = function_values['a'] * function_values['b'] + function_values['c']
    d2 = function

    derivative = (d2 - d1) / h  # RISE/RUN

    print('d1', d1)
    print('d2', d2)
    print('slope (derivative with respect to {}): {}'.format(with_respect_to, derivative))

    """
    NOTES: At the case of 'a' we go down because the expression a*b is negative thanks to b being -3.0.
    Since we increase the a, we also increase the influence of negative value here presente due to the influence of b
    
    In case of increase b we however increased the negative value (negative value is no longer that big), thus the
    derivative/slope is positive.
    
    c is alone, so it's always positive, thus when increased the change of the whole expression is also positive 
    (it slightly eliminates the influence of the negative b) 
    
    """


derivative(values, 'a')
derivative(values, 'b')
derivative(values, 'c')


# For handling the math complexity of the NN it is better to create special data structure
class Value:

    def __init__(self, data, children=(), operator='', label=''):
        self.data = data
        # We have a graph structure, thus we need to know the connected values
        self.previous_values = set(children)  # The tuple to set conversion is there just for the greater efficiency
        self.operator_symbol = operator
        # This was added later
        self.label = label
        self.gradient = 0.0  # 0 = "no change", "no effect", will not affect anything
        self._backward = lambda: None

    def __repr__(self):
        """
        For the representation of the object.
        __str__ is rather used in the case you need some "pretty prints" about the characteristics of the object
        :return:
        """
        return f"Value(data={self.data})"

    # We are rewriting the default math operations by own representation
    # where we can also take the child nodes into an account
    def __add__(self, added_value):
        added_value = added_value if isinstance(added_value, Value) else Value(added_value)
        result = Value(self.data + added_value.data, (self, added_value), '+')

        def _backward():
            self.gradient += 1.0 * result.gradient
            added_value.gradient += 1.0 * result.gradient

        result._backward = _backward  # saving the result, not call the result (output itself is None)
        return result

    def __mul__(self, multiplier_value):
        multiplier_value = multiplier_value if isinstance(multiplier_value, Value) else Value(multiplier_value)
        # this line need to be added after torch is started
        result = Value(self.data * multiplier_value.data, (self, multiplier_value), '*')

        def _backward():
            # Chain rule
            self.gradient += multiplier_value.data * result.gradient
            multiplier_value.gradient += self.data * result.gradient

        result._backward = _backward
        return result

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        result = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.gradient += other * (self.data ** (other - 1)) * result.gradient

        result._backward = _backward

        return result

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __neg__(self):  # -self
        return self * Value(-1)

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __radd__(self, other):  # other + self
        return self + other

    def activation_function_tanh(self):

        """
        Alternative implementation:

        e = math.e
        print("x type: {}".format(type(x)))
        print("e type: {}".format(type(e)))
        f_x = (2 / (1 + e ** (-2 * x))) - 1
        threshold_out = Value(f_x, (self,), 'tanh')  # We want to start recording children

        def _backward():
            self.gradient += (1 - f_x ** 2) * threshold_out.gradient

        threshold_out._backward = _backward

        return threshold_out
        """
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.gradient += (1 - t ** 2) * out.gradient

        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
            self.gradient += out.data * out.gradient

        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.previous_values:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.gradient = 1.0
        for node in reversed(topo):
            node._backward()


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
print("d now, also with the children and the operator which produced the value:")
print(d)
# print(d.previous_values)
# print(d.operator_symbol)

# Now let's visualize
# draw_dot(d)

"""
##Forward pass of some math expressions
### Easy model of something that happens later in the NN in more complex way
"""

# More complex graph
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b
# we added the labels here for the node naming
e.label = 'e'

d = e + c;
d.label = 'd'  # Cool trick when need to write one line (but not best practise, IDE usually doesn't like this)
f = Value(-2.0, label='f')
L = d * f  # Loss function
L.label = 'L'
print(L)

# draw_dot(L)

# For the backpropagation, we need to calculate a local derivatives now (see introductory OneNote notes)
# => we need to calculate local derivatives of all the nodes used (the end node d/dL, here L is just =1)
# d/df, d/dd...,
# How are weights affecting the loss function? Derivative of L with respect to the particular node.
# THIS IS SUPER IMPORTANT FOR THE NEURAL NETWORK!!!
# dL/df, dL/dd, ...
# In NN the node represents a weight. Important note that weight is in respect to this model a whole node,
# not only the value!
# The value itself is a fixed data
# ==>

# For storing the derivative with respect to the particular value,
# we use the 'gradient' attribute

"""
## (Half-)manual backpropagation calculation
### For the sake of demonstration, late better method will be introduced
"""


def derivatives_of_L_calculation(with_respect_to):
    """
    ## Calculating the derivatives demonstration for the manual backpropagation
    ### dL/dL
    :return:
    """

    # More complex graph
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b
    # we added the labels here for the node naming
    e.label = 'e'

    d = e + c;
    d.label = 'd'  # Cool trick when need to write one line (but not best practise, IDE usually doesn't like this)
    f = Value(-2.0, label='f')
    L = d * f  # Loss function
    L1 = L.data

    h = 0.000001
    # add the h according to which variable in respect to you calculate
    # lower the 'h' is, more precise the result is (until it gets out of the floating point memory)

    a = Value(2.0, label='a')
    if with_respect_to == 'a':
        a.data += h
    b = Value(-3.0, label='b')
    if with_respect_to == 'b':
        b.data += h
    c = Value(10.0, label='c')
    if with_respect_to == 'c':
        c.data += h
    e = a * b
    e.label = 'e'
    if with_respect_to == 'e':
        e.data += h

    d = e + c
    d.label = 'd'
    if with_respect_to == 'd':
        d.data += h

    f = Value(-2.0, label='f')
    if with_respect_to == 'f':
        f.data += h

    L = d * f  # Loss function

    if with_respect_to == 'L':
        L2 = L.data + h
    else:
        L2 = L.data

    # this is the gradient now
    derivative = (L2 - L1) / h

    print(derivative)
    return derivative


# dL/dL
L.gradient = derivatives_of_L_calculation('L')
print(L.gradient)
# draw_dot(L)

# dL/dd
d.gradient = derivatives_of_L_calculation('d')
print(d.gradient)
# draw_dot(L)

# others
e.gradient = derivatives_of_L_calculation('e')
f.gradient = derivatives_of_L_calculation('f')
c.gradient = derivatives_of_L_calculation('c')
b.gradient = derivatives_of_L_calculation('b')
a.gradient = derivatives_of_L_calculation('a')

# draw_dot(L)

"""
Since we are multiplying the local derivatives and forward the partial solution, 
we have basically created the famous derivative Chain Rule AND THIS IS THE BACKPROPAGATION
"""

"""
## Using the backpropagation: single optimization step
"""


def optimization_step(node):
    learning_rate = 0.01  # Also called 'step_size' in some ML libraries
    return node.data * learning_rate


nodes = {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f}
updated_nodes = {}

for label, node in nodes.items():
    updated_node = node
    updated_node.data = optimization_step(node)
    updated_nodes[label] = updated_node

a = updated_nodes['a']
b = updated_nodes['b']
e = a * b
c = updated_nodes['c']
d = e + c
f = updated_nodes['f']
L = d * f

print("Updated L after the optimization:")
print(L.data)  # Less negative L, it's closer to zero, which is better

## Finally Neural Network
# IN => sum(w_i+x_i) + b => activation function (threshold) => OUT

# IN
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# w_i
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias
b_precalculated = 6.8813735870195432
b = Value(b_precalculated, label='b')

# NN: sum of multiplication of inputs from previous layer and weights
x1w1 = x1 * w1
x1w1.label = 'x1*w1'
x2w2 = x2 * w2
x2w2.label = 'x2*w2'
x1w1_plus_x2w2 = x1w1 + x2w2
x1w1_plus_x2w2.label = 'x1*w1+x2*w2'
y_in = x1w1_plus_x2w2 + b
y_in.label = 'y_in'

# draw_dot(y_in)

# we need to implement the activation function
# we also want to start recording children to save the previous nodes

out = y_in.activation_function_tanh()
out.label = 'out'
# draw_dot(out)

## Gradients for this simple NN
# The last node is again just 1

out.gradient = 1
# d/dx
# derivative of tanh (see Wiki or something for reference)
# NOTICE: You cannot call out.gradient here!!! But the value (data).
y_in.gradient = 1 - out.data ** 2

print(y_in.gradient)

# draw_dot(out)
# because there is plus on the graph, backprop just forwards further without changing the value
# whenever there is a '+' we can just feed the previous value (propagate) backwards without needing to compute another
# derivatives
x1w1_plus_x2w2.gradient = y_in.gradient
b.gradient = y_in.gradient
# also here is a plus

x1w1.gradient = y_in.gradient
x2w2.gradient = y_in.gradient

# Local part of chain rule for x2 and w2
x2.gradient = w2.data * x2w2.gradient
w2.gradient = x2.data * x2w2.gradient  # no change, derivative is zero, because x2 is 0.5 and x2*w2 is 0.5

# Local part of chain rule for x1 and w1
x1.gradient = w1.data * x1w1.gradient
w1.gradient = x1.data * x1w1.gradient

## Now backprop better
# using the _backward function
out._backward()
y_in._backward()
b._backward()
x1w1_plus_x2w2._backward()
x1w1._backward()
x2w2._backward()
w1._backward()
x1._backward()
w2._backward()
x2._backward()

# draw_dot(out)

# This is not correct yet. We want to solve propagation for every dependency
# until we calculate the local backpropagation.
# For this purpose we will use a topological sort.

a = Value(3.0, label='a')
b = a + a
b.backward()

# draw_dot(b)

# implementing other alternative derivatives of tanh along with the supportive functions
a = Value(2.0)
print(a.exp())

## Real (production) library

import torch

x1 = torch.Tensor([2.0]).double()  # for aligning the data type with Python
x1.requires_grad = True
x2 = torch.Tensor([0.0]).double();
x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double();
w1.requires_grad = True
w2 = torch.Tensor([1.0]).double();
w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double();
b.requires_grad = True
mlp = x1 * w1 + x2 * w2 + b

o = torch.tanh(mlp)
print(o.data.item())
o.backward()

print('---')
print("Gradients:")
print('---')
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())


class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        output = activation.activation_function_tanh()
        return output

    def parameters(self):
        return self.w + [self.b]


#  line needs to be added to __mul__ after torch is started
x = [2.0, 3.0]
neuron = Neuron(2)
neuron(x)


# Multi-layer perceptron
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params


class MultiLayerPerceptron:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


x = [2.0, 3.0, -1.0]
mlp = MultiLayerPerceptron(3, [4, 4, 1])
print(mlp(x))

X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

Y = [1.0, -1.0, -1.0, 1.0]

y_predicted = [mlp(x) for x in X]
print(y_predicted)
y_predicted_old = y_predicted.copy()

print("Losses:")
print([(y_out - y_true) ** 2 for y_true, y_out in zip(Y, y_predicted)])  # we are measuring how far we are by a square
loss = sum([(y_out - y_true) ** 2 for y_true, y_out in zip(Y, y_predicted)])

print("loss: ", loss)
print("example weight: ", mlp.layers[0].neurons[0].w[0])
print("example gradient: ", mlp.layers[0].neurons[0].w[0].gradient)

print("Evaluating backpropagation....")
loss.backward()

print("loss: ", loss)
print("example weight: ", mlp.layers[0].neurons[0].w[0])
print("example gradient: ", mlp.layers[0].neurons[0].w[0].gradient)

# draw_dot(loss)

## Optimization

print("mlp.parameters()")
print(mlp.parameters())

old_losses = set()
for i, p in enumerate(mlp.parameters()):
    # print("{}, loss: {}".format(i, p))
    old_losses.add(p.data)

new_losses = set()
for i, p in enumerate(mlp.parameters()):
    learning_rate = -0.01  # alpha, we want to minimize loss, shouldn't be too high (overshooting), nor too low (slow)
    p.data += learning_rate * p.gradient
    #  print("{}, loss: {}".format(i, p))
    new_losses.add(p.data)

print(new_losses)
print(old_losses)

improvements = [old_loss - (old_loss - new_loss) for new_loss, old_loss in zip(new_losses, old_losses)]

improvement = sum(improvements)

print("improvement:")
print(improvement)

y_predicted = [mlp(x) for x in X]

print("Losses:")
print([(y_out - y_true) ** 2 for y_true, y_out in zip(Y, y_predicted)])  # we are measuring how far we are by a square
loss = sum([(y_out - y_true) ** 2 for y_true, y_out in zip(Y, y_predicted)])
loss.backward()

print("y_prediceted old:")
print(y_predicted_old)
print("y_predicted:")
print(y_predicted)

# we are closer to the right values now, but we need to progress further

## Automatization of optimization

learning_iterations = 20
for k in range(learning_iterations):
    # FORWARD
    y_predicted = [mlp(x) for x in X]
    loss = sum((y_out - y_true)**2 for y_true, y_out in zip(Y, y_predicted))

    # BACKWARD
    loss.backward()

    # UPDATE
    for p in mlp.parameters():
        p.data += -0.05 * p.gradient

    print(k, loss.data)

# Now we are close to one. We managed to learn the NN this value.
print("y_predicted:")
print(y_predicted)
