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
    def __call__(self, *inputs): # 여기서 input은 variable 인스턴스로 가정
        # x = input.data # data를 꺼낸다
        xs = [x.data for x in inputs] # 여러개의 변수에 대응할 수 있도록 변화
        ys = self.forward(*xs) # 언팩
        if not isinstance(ys, tuple): # tuple이 아닌 경우 tuple로
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys] # Variable 형태로 되돌린다

        for output in outputs:
            output.set_creator(self) # 출력 변수에 창조자(creator) 설정
        self.inputs = inputs # 입력변수를 기억해둔다
        self.outputs = outputs # 출력도 저장해버리기
        return outputs if len(outputs) > 1 else outputs[0] # 리스트의 원소가 하나라면 첫번째 원소를 반환한다.
    
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

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y # tuple

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x) # Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

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

print("------------------ Chap2.1 ------------------")

# xs = [Variable(np.array(2)), Variable(np.array(3))]
# f = Add()
# ys = f(xs)
# y = ys[0]
# print(y.data)

print("------------------ Chap2.1.2 ------------------")

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)