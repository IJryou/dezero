import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input): # 여기서 input은 variable 인스턴스로 가정
        x = input.data # data를 꺼낸다
        y = self.forward(x)
        output = Variable(y) # Variable 형태로 되돌린다
        return output
    
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data + eps)
    x1 = Variable(x.data - eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y0.data - y1.data) / (2*eps)

data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)

x = np.array(1.0)
print(x.ndim)

x = np.array([1,2,3])
print(x.ndim)

x = np.array([[1,2,3],
            [4,5,6]])
print(x.ndim)

x_1 = Variable(np.array(10)) # variable의 인스턴스 생성
f = Square() # Function 선언
y = f(x_1)

print(type(y))
print(y.data)

print("------------------ Chap3 ------------------")

A = Square()
B = Exp()
C = Square()

x_2 = Variable(np.array(0.5))
a = A(x_2)
b = B(a)
c = C(b)

print(type(c))
print(c.data)

print("------------------ Chap4 ------------------")

f = Square()
x = Variable(np.array(2.0))
y = numerical_diff(f, x)
print(y)

print("------------------ Chap4 ------------------")

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
y = numerical_diff(f, x)
print(y)