import math
import numpy as np
import matplotlib.pyplot as plt

from graph_visualisation import draw_dot


def easy_function(x):
    return 4 * x ** 2 + 5 * x + 1


x_values = np.arange(-10, 10, 0.2)
y_values = easy_function(x_values)
plt.plot(x_values, y_values)
plt.show()

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

    def __init__(self, data, children=(), operator=''):
        self.data = data
        # We have a graph structure, thus we need to know the connected values
        self.previous_values = set(children)  # The tuple to set conversion is there just for the greater efficiency
        self.operator_symbol = operator

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
        result = Value(self.data + added_value.data, (self, added_value), '+')
        return result

    def __mul__(self, multiplier_value):
        result = Value(self.data * multiplier_value.data, (self, multiplier_value), '*')
        return result


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b + c
print("d now, also with the children and the operator which produced the value:")
print(d)
print(d.previous_values)
print(d.operator_symbol)

# Now let's visualize
draw_dot(d)


