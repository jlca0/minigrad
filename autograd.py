import numpy as np

class Value:
    """
    A class representing a scalar value with automatic differentiation capabilities.
    Attributes:
        data (float): The underlying scalar value.
        grad (float): The gradient of the value, initialized to zero.
        _backward (function): A function to compute the gradient of the value.
        _prev (set): A set of parent values in the computation graph.
        _op (str): The operation that produced this value.
        label (str): An optional label for the value.
    """
    def __init__(
            self, 
            data: float, 
            _children: tuple = (), 
            _op: str = '', 
            label: str = ''
            )-> None:
        """
        Initializes a Value object.
        Args:
            data (float): The scalar value.
            _children (tuple): A tuple of parent Value objects in the computation graph.
            _op (str): The operation that produced this value.
            label (str): An optional label for the value.
        """
        self.data = data   
        self.grad = 0.0               
        self._backward = lambda: None 
        self._prev = set(_children)  
        self._op = _op      
        self.label = label          

    def __add__(
            self, 
            other: 'Value'
            )-> 'Value':
        """
        Adds two Value objects.
        Args:
            other (Value): Another Value object or a scalar to add.
        Returns:
            out (Value): A new Value object representing the sum.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(
            self, 
            other: 'Value'
            )-> 'Value':
        """
        Multiplies two Value objects.
        Args:
            other (Value): Another Value object or a scalar to multiply.
        Returns:
            out (Value): A new Value object representing the product.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def reLU(self)-> 'Value':
        """
        Applies the ReLU activation function.
        Returns:
            out (Value): A new Value object with ReLU applied.
        """
        out = Value(max(0, self.data), (self,), _op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        Computes the gradients of the value with respect to its inputs using backpropagation.
        """ 
        topo = []
        visited = set()

        # Build the topological order of the computation graph.
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

class Tensor:
    """
    A class representing a tensor with automatic differentiation capabilities.
    Attributes:
        data (np.ndarray): The underlying data of the tensor.
        grad (np.ndarray): The gradient of the tensor, initialized to zeros.
        _backward (function): A function to compute the gradient of the tensor.
        _prev (set): A set of parent tensors in the computation graph.
        _op (str): The operation that produced this tensor.
        label (str): An optional label for the tensor.
    """
    def __init__(
            self, 
            data: np.ndarray, 
            _children: tuple =(), 
            _op: str = '', 
            label: str = ''
            )-> None:
        """
        Initializes a Tensor object.
        """
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data, dtype=float)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    @property
    def T(self):
        return self.transpose()

    def __add__(
            self, 
            other: 'Tensor')-> 'Tensor':
        """
        Adds two tensors element-wise.
        Args:
            other (Tensor): Another tensor or a scalar to add.
        Returns:
            out (Tensor): A new tensor representing the element-wise sum.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), _op='+')

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)
            
        out._backward = _backward
        return out

    def __mul__(
            self, 
            other: 'Tensor')-> 'Tensor':
        """
        Multiplies two tensors element-wise.
        Args:
            other (Tensor): Another tensor or a scalar to multiply.
        Returns:
            out (Tensor): A new tensor representing the element-wise product.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), _op='*')

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += self._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(
            self, 
            other: 'Tensor')-> 'Tensor':
        """
        Performs matrix multiplication between two tensors.
        Args:
            other (Tensor): Another tensor to multiply with.
        Returns:
            out (Tensor): A new tensor representing the matrix product.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), _op='@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def __neg__(self)->'Tensor':
        """
        Negates a Tensor (multiply by -1.)
        Returns:
            out (Tensor): A new tensor with opposite data.
        """
        out = Tensor(-self.data, (self, ), _op='-')

        def _backward():
            self.grad += self._unbroadcast(-out.grad, self.data.shape)
        out._backward = _backward

        return out

    def transpose(self)-> 'Tensor':
        """
        Transposes a tensor.
        Returns:
            out (Tensor): A new tensor with transposed data.
        """
        out = Tensor(self.data.T, (self, ), _op='T')

        def _backward():
            self.grad += self._unbroadcast(out.grad.T, self.data.shape)
        out._backward =_backward

        return out

    def reLU(self)-> 'Tensor':
        """
        Applies the ReLU activation function element-wise.
        Returns:
            out (Tensor): A new tensor with ReLU applied.
        """
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op = 'reLU')

        def _backward():
            mask = self.data > 0
            self.grad += mask * out.grad

        out._backward = _backward
        return out

    def sigmoid(self)-> 'Tensor':
        """
        Applies the sigmoid activation function element-wise.
        Returns:
            out (Tensor): A new tensor with sigmoid applied.
        """
        x = np.clip(self.data, -500.0, 500.0)
        out_data = 1.0 / (1.0 + np.exp(-x))
        out = Tensor(out_data, _children=(self,), _op = 'sigmoid')

        def _backward():
            local_grad = out.data * (1.0 - out.data)
            self.grad += local_grad * out.grad

        out._backward = _backward
        return out

    def log(self)-> 'Tensor':
        """
        Applies the natural logarithm activation function element-wise.
        Returns:
            out (Tensor): A new tensor with natural logarithm applied.
        """
        out = Tensor(np.log(self.data), _children=(self,), _op='log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def softmax(self)-> 'Tensor':
        """
        Applies the softmax activation function along the last axis.
        Returns:
            out (Tensor): A new tensor with softmax applied.
        """
        exp_data = np.exp(self.data - np.max(self.data, axis=0, keepdims=True))
        out_data = exp_data / np.sum(exp_data, axis=0, keepdims=True)
        out = Tensor(out_data, _children=(self,), _op='softmax')

        def _backward():
            sum_grad = np.sum(out.grad * out.data, axis=0, keepdims=True)
            self.grad += out.data * (out.grad - sum_grad)

        out._backward = _backward

        return out

    def backward(self):
        """
        Computes the gradients of the tensor with respect to its inputs using backpropagation.
        """
        topo = []
        visited = set()

        # Build the topological order of the computation graph.
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def _unbroadcast(
            self, 
            grad: np.ndarray, 
            target_shape: tuple
            )-> np.ndarray:
        """
        Adjusts the gradient shape to match the target shape by summing over broadcasted dimensions.
        Args:
            grad (np.ndarray): The gradient to be adjusted.
            target_shape (tuple): The desired shape of the gradient.
        Returns:
            np.ndarray: The adjusted gradient with the target shape.
        """
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
                
        return grad
