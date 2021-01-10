import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')
        
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) # 주어진 데이터와 똑같은 shape의 input 생성

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 앞의 함수를 가져온다
            x, y = f.input, f.output # 함수의 입력과 출력을 가져온다
            x.grad = f.backward(y.grad) # backward 메서드 호출

            if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가한다

class Function:
    def __call__(self, input): # 여기서 input은 variable 인스턴스로 가정
        x = input.data # data를 꺼낸다
        y = self.forward(x)
        output = Variable(as_array(y)) # Variable 형태로 되돌린다
        output.set_creator(self) # 출력 변수에 창조자(creator) 설정
        self.input = input # 입력변수를 기억해둔다
        self.output = output # 출력도 저장해버리기
        return output
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        return 2 * x * gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x) # Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
    
def numerical_diff(f, x, eps=as_array(1e-4)):
    input_x0 = np.array(x.data + eps) # Variable 선언단계에서 data type 안맞으면 에러남
    input_x1 = np.array(x.data - eps)
    x0 = Variable(input_x0)
    x1 = Variable(input_x1)
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

x_1 = Variable(np.array(10.0)) # variable의 인스턴스 생성
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

print("------------------ Backward From Chap3 ------------------")

A = Square()
B = Exp()
C = Square()

x_3 = Variable(np.array(0.5))
a = A(x_3)
b = B(a)
c = C(b)

assert c.creator == C
assert c.creator.input == b
assert c.creator.input.creator == B
assert c.creator.input.creator.input == a
assert c.creator.input.creator.input.creator == A
assert c.creator.input.creator.input.creator.input == x_3

print(type(c))
print(c.data)

c.grad = np.array(1.0)
b.grad = C.backward(c.grad)
a.grad = B.backward(b.grad)
x_3.grad = A.backward(a.grad)
print(x_3.grad)

print("------------------ Chap7 ------------------")

c.grad = np.array(1.0)

C = c.creator
b = C.input
b.grad = C.backward(c.grad)

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x_3 = A.input
x_3.grad = A.backward(a.grad)

print(x_3.grad)

print("------------------ Chap7 adjust_autograd ------------------")

c.grad = np.array(1.0)
c.backward()

print(x_3.grad)

print("------------------ Chap9 method to ftn ------------------")

x = Variable(np.array(0.5))
# a = square(x)
# b = exp(a)
# c = square(b)
c = square(exp(square(x)))


c.grad = np.array(1.0)
c.backward()
print(x.grad)
print("------------------ Chap9 modified ------------------")

x = Variable(np.array(0.5))
c = square(exp(square(x)))
c.backward()
print(x.grad)