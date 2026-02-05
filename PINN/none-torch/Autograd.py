import math
import cupy as cp

class AutoGrad:  ## from Autograd import AutoGrad as AG 로 사용 바람.
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # 초기 노드 미분값은 0으로 초기화함 
        self._backward = lambda: None
        self._prev = set(_children) # 그래프 연결 정보
        self._op = _op # 디버깅용 연산자 표시

    def __add__(self, other):
        other = other if isinstance(other, AutoGrad) else AutoGrad(other)

        out = AutoGrad(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, AutoGrad) else AutoGrad(other)
        out = AutoGrad(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, AutoGrad) else AutoGrad(other)

        out = AutoGrad(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out._unbroadcast(out.grad, self.data.shape)
            other.grad += self.data * out._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out
    
    def sin(self):
        out = AutoGrad(math.sin(self.data), (self,), 'sin')

        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward

        return out
    
    def cos(self):
        out = AutoGrad(math.cos(self.data), (self,), 'cos')

        def _backward():
            self.grad += -math.sin(self.data) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        out = AutoGrad(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def log(self):
        out = AutoGrad(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, power):
        out = AutoGrad(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        other = other if isinstance(other, AutoGrad) else AutoGrad(other)

        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out = AutoGrad(self.data / other.data, (self, other), '/')
        out._backward = _backward

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


class Tensor: # from Autograd import Tensor as Tsr 로 사용 바람.
    def __init__(self, data, _children=(), _op='', label=''):

            if not isinstance(data, cp.ndarray):
                data = cp.array(data, dtype=cp.float32)
                
            self.data = data
            self.grad = cp.zeros_like(data, dtype=cp.float32)
            
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out._unbroadcast(out.grad, self.data.shape)
            other.grad += out._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out._unbroadcast(out.grad, self.data.shape)
            other.grad += -1.0 * out._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
            other = other if isinstance(other, Tensor) else Tensor(other)
            
            out = Tensor(self.data / other.data, (self, other), '/')

            def _backward():
                grad_self = (1 / other.data) * out.grad
                self.grad += self._unbroadcast(grad_self, self.data.shape)
                
                grad_other = (-self.data / (other.data ** 2)) * out.grad
                other.grad += self._unbroadcast(grad_other, other.data.shape)
                
            out._backward = _backward

            return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += self._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward

        return out
    
    def sin(self):
        out = Tensor(cp.sin(self.data), (self,), 'sin')

        def _backward():
            self.grad += cp.cos(self.data) * out.grad
        out._backward = _backward

        return out
    
    def cos(self):
        out = Tensor(cp.cos(self.data), (self,), 'cos')
                     
        def _backward():
            self.grad += -cp.sin(self.data) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        out = Tensor(cp.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def log(self):
        out = Tensor(cp.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(cp.matmul(self.data, other.data), (self, other), '@')

        def _backward():
            self.grad += cp.matmul(out.grad, other.data.T)
            other.grad += cp.matmul(self.data.T, out.grad)
        out._backward = _backward

        return out
    
    def tanh(self):
        out = Tensor(cp.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward

        return out
    
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), 'getitem')
        
        def _backward():
            # 전체 0 행렬을 만들고
            grad = cp.zeros_like(self.data)
            # 선택된 인덱스에만 out.grad를 더해줌
            grad[idx] += out.grad
            self.grad += grad
            
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(cp.sum(self.data), (self,), 'sum')
        
        def _backward():
            self.grad += cp.ones_like(self.data) * out.grad
            
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__truediv__(self)
    
    def _unbroadcast(self, grad, shape):
            
            if grad.shape == shape:
                return grad
                
            while grad.ndim > len(shape):
                grad = cp.sum(grad, axis=0)
            for i, dim in enumerate(shape):
                if dim == 1:
                    grad = cp.sum(grad, axis=i, keepdims=True)
            return grad
    
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Tensor(other) - self

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = cp.ones_like(self.data, dtype=cp.float32)

        for node in reversed(topo):
            node._backward()