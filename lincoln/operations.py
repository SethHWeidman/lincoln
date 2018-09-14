import torch
from torch import Tensor

from .exc import assert_same_shape

__all__ = ["WeightMultiply", "BiasAdd", "Sigmoid", "LogSigmoid", "Softmax", "LogSoftmax", "ReLU"]

class Operation(object):

    def __init__(self):
        pass


    def forward(self, input: Tensor):
        self.input = input

        self.output = self._output()

        return self.output


    def backward(self, output_grad: Tensor) -> Tensor:

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input, self.input_grad)
        return self.input_grad

    def _output(self) -> Tensor:
        raise NotImplementedError()

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()


class ParamOperation(Operation):

    def __init__(self, param: Tensor) -> Tensor:
        super().__init__()
        self.param = param

    def backward(self, output_grad: Tensor) -> Tensor:

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input, self.input_grad)
        return self.input_grad

    def _param_grad(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()


class WeightMultiply(ParamOperation):

    def __init__(self, W: Tensor):
        super().__init__(W)


    def _output(self) -> Tensor:
        return torch.mm(self.input, self.param)


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return torch.mm(output_grad, self.param.transpose(0, 1))


    def _param_grad(self, output_grad: Tensor) -> Tensor:
        return torch.mm(self.input.transpose(0, 1), output_grad)


class BiasAdd(ParamOperation):

    def __init__(self,
                 B: Tensor):
        super().__init__(B)


    def _output(self) -> Tensor:
        return torch.add(self.input, self.param)


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return torch.ones_like(self.input) * output_grad


    def _param_grad(self, output_grad: Tensor) -> Tensor:
        param_grad = torch.ones_like(self.param) * output_grad
        return torch.sum(param_grad, dim=0).reshape(1, param_grad.shape[1])


class Sigmoid(Operation):
    '''
    Sigmoid activation function
    '''
    def __init__(self) -> None:
        super().__init__()


    def _output(self) -> Tensor:
        return 1.0/(1.0+torch.exp(-1.0 * self.input))


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        # Lines specific to this class
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad

    def __repr__(self):
        return "Sigmoid"


class ReLU(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> Tensor:
        self.output = torch.clamp(self.input, 0, 1e5)
        return self.output

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        relu_backward = (self.output > 0).type(self.output.dtype)
        return relu_backward * output_grad

    def __repr__(self):
        return "ReLU"


class Flatten(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> Tensor:
        return self.input.view(self.input.shape[0], -1)

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return output_grad.view(*self.last_input.shape)

    def __repr__(self):
        return "Flatten"


class LogSigmoid(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> Tensor:
        self.output = self.input - torch.log(torch.exp(self.input) + 1)
        return self.output

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return (1 - torch.exp(self.output))*output_grad

    def __repr__(self):
        return "LogSigmoid"
         

class Softmax(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> Tensor:
        self.output = torch.exp(self.input) / torch.sum(torch.exp(self.input), dim=1).view(n, 1)
        return self.output

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        ps = self.output
        N, M = ps.shape[0], ps.shape[1]
        batch_jacobian = torch.zeros((N, M, M))
        
        for ii, p in enumerate(ps):
            batch_jacobian[ii,:,:] = torch.diag(p) - torch.ger(p, p)
        
        backward_grad = torch.bmm(output_grad.view(N, 1, -1), batch_jacobian)
        backward_grad.squeeze_()

        return backward_grad
    
    def __repr__(self):
        return "Softmax"

class LogSoftmax(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> Tensor:
        self.output = self.input - torch.log(torch.exp(self.input).sum(dim=1).view(-1, 1))
        return self.output

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        ps = torch.exp(self.output)
        N, M = ps.shape[0], ps.shape[1]
        batch_jacobian = torch.zeros((N, M, M))
         
        # Create an identity matrix
        ones = torch.diagflat(torch.ones(M))
        
        for ii, p in enumerate(ps):
            # Repeat the p values across columns to get p_k
            p_k = p.repeat((M, 1))
            batch_jacobian[ii,:,:] = ones - p_k
        
        backward_grad = torch.bmm(output_grad.view(N, 1, -1), batch_jacobian)
        backward_grad.squeeze_()

        return backward_grad

    def __repr__(self):
        return "LogSoftmax"