import torch
from torch import Tensor

from .utils import assert_same_shape, assert_dim
from torch import nn

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

        self._compute_grads(output_grad)

        assert_same_shape(self.input, self.input_grad)
        return self.input_grad


    def _compute_grads(self, output_grad: Tensor) -> Tensor:

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


    def _compute_grads(self, output_grad: Tensor) -> Tensor:
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)


    def _param_grad(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()


class Flatten(Operation):
    def __init__(self):
        super().__init__()


    def _output(self) -> Tensor:
        return self.input.view(self.input.shape[0], -1)


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return output_grad.view(*self.input.shape)


class Softmax(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> Tensor:
        self.output = torch.exp(self.input) / torch.sum(torch.exp(self.input), dim=1).view(-1, 1)
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


class ReLU(Operation):
    def __init__(self):
        super().__init__()

<<<<<<< HEAD
    def backward(self, output_grad: Tensor) -> Tensor:

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input, self.input_grad)
        return self.input_grad
=======
    def _output(self) -> Tensor:
        self.output = torch.clamp(self.input, 0, 1e8)
        return self.output

    def _input_grad(self, output_grad: Tensor) -> Tensor:
        relu_backward = (self.output > 0).type(self.output.dtype)
        return relu_backward * output_grad

    def __repr__(self):
        return "ReLU"
>>>>>>> test_conv


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

class Conv2D_Op(ParamOperation):

    def __init__(self, param: Tensor):
        assert_dim(param, 4)
        super().__init__(param)
        self.param_size = self.param.shape[2]
        self.param_pad = self.param_size // 2
        self.in_channels = self.param.shape[1]
        self.out_channels = self.param.shape[0]


    def _pad_1d_obs(self, obs: Tensor) -> Tensor:
        assert_dim(obs, 1)
        z = torch.Tensor([0])
        z = z.repeat(self.param_pad)
        return torch.cat([z, obs, z])


    def _pad_1d_batch(self, inp: Tensor) -> Tensor:
        assert_dim(inp, 2)
        outs = [self._pad_1d_obs(obs) for obs in inp]
        return torch.stack(outs)


    def _pad_2d_obs(self, inp: Tensor):
        assert_dim(inp, 2)
        inp_pad = self._pad_1d_batch(inp)
        other = torch.zeros(self.param_pad, inp.shape[0] + self.param_pad * 2)
        # import pdb; pdb.set_trace()
        return torch.cat([other, inp_pad, other])


    def _pad_2d_batch(self, inp: Tensor):
        assert_dim(inp, 3)
        outs = [self._pad_2d_obs(obs) for obs in inp]
        return torch.stack(outs)


    def _pad_2d_channel(self, input_obs: Tensor):

        assert_dim(input_obs, 3)
        return torch.stack([self._pad_2d_obs(channel) for channel in input_obs])


    def _pad_conv_input(self):
        return torch.stack([self._pad_2d_channel(obs)
                            for obs in self.input])


    def _compute_output_obs(self, obs: Tensor):

        assert_dim(obs, 3)
        obs_pad = self._pad_2d_channel(obs)

        out = torch.zeros((self.out_channels,) + obs.shape[1:])
        for c_out in range(self.out_channels):
            for c_in in range(self.in_channels):
                for o_w in range(self.input.shape[2]):
                    for o_h in range(self.input.shape[3]):
                        for p_w in range(self.param_size):
                            for p_h in range(self.param_size):
                                out[c_out][o_w][o_h] += \
                                self.param[c_out][c_in][p_w][p_h] * obs_pad[c_in][o_w+p_w][o_h+p_h]
        return out


    def _output(self):

        outs = [self._compute_output_obs(obs) for obs in self.input]
        return torch.stack(outs)


    def _compute_grads_obs(self, input_obs: Tensor,
                           output_grad_obs: Tensor) -> Tensor:

        assert_dim(input_obs, 3)
        assert_dim(output_grad_obs, 3)

        output_obs_pad = self._pad_2d_channel(output_grad_obs)
        input_grad = torch.zeros_like(input_obs)

        for c_in in range(self.in_channels):
            for c_out in range(self.out_channels):
                for i_w in range(input_obs.shape[1]):
                    for i_h in range(input_obs.shape[2]):
                        for p_w in range(self.param_size):
                            for p_h in range(self.param_size):
                                input_grad[c_in][i_w][i_h] += \
                                output_obs_pad[c_out][i_w+self.param_size-p_w-1][i_h+self.param_size-p_h-1] \
                                * self.param[c_out][c_in][p_w][p_h]

        return input_grad


    def _input_grad(self, output_grad: Tensor) -> Tensor:

        grads = [self._compute_grads_obs(self.input[i], output_grad[i]) for i in range(self.input.shape[0])]

        return torch.stack(grads)


    def _param_grad(self, output_grad: Tensor) -> Tensor:

        inp_pad = self._pad_conv_input()
        param_grad = torch.zeros_like(self.param)

        for i in range(self.input.shape[0]):
            for c_in in range(self.in_channels):
                for c_out in range(self.out_channels):
                    for o_w in range(output_grad.shape[2]):
                        for o_h in range(output_grad.shape[3]):
                            for p_w in range(self.param_size):
                                for p_h in range(self.param_size):
                                    param_grad[c_out][c_in][p_w][p_h] += \
                                    inp_pad[i][c_in][o_w+p_w][o_h+p_h] \
                                    * output_grad[i][c_out][o_w][o_h]
        return param_grad


from .conv_c import (_pad_1d_obs_cy,
                    _pad_1d_batch_cy,
                    _pad_2d_obs_cy,
                    _pad_2d_channel_cy,
                    _pad_conv_input_cy,
                    _compute_output_obs_cy,
                    _output_cy,
                    _compute_grads_obs_cy,
                    _input_grad_cy,
                    _param_grad_cy)


class Conv2D_Op_cy(ParamOperation):


    def __init__(self, param: Tensor):
        assert_dim(param, 4)
        super().__init__(param)
        self.param_size = self.param.shape[2]
        self.param_pad = self.param_size // 2
        self.in_channels = self.param.shape[1]
        self.out_channels = self.param.shape[0]


    def _pad_1d_obs(self, obs: Tensor) -> Tensor:
        assert_dim(obs, 1)
        obs_np = obs.numpy()
        return Tensor(_pad_1d_obs_cy(obs_np, self.param_pad))


    def _pad_1d_batch(self, inp: Tensor) -> Tensor:
        assert_dim(inp, 2)
        inp_np = inp.numpy()
        return Tensor(_pad_1d_batch_cy(inp_np, self.param_pad))


    def _pad_2d_obs(self, inp: Tensor):
        assert_dim(inp, 2)
        inp_np = inp.numpy()
        return Tensor(_pad_2d_obs_cy(inp_np, self.param_pad))


    def _pad_2d_batch(self, inp: Tensor):
        assert_dim(inp, 3)
        inp_np = inp.numpy()
        return Tensor(_pad_2d_batch_cy(inp_np, self.param_pad))


    def _select_channel(self, inp: Tensor, i: int):
        assert_dim(inp, 3)
        inp_np = inp.numpy()
        return Tensor(_select_channel_cy(inp_np, i))


    def _pad_2d_channel(self, input_obs: Tensor):
        assert_dim(input_obs, 3)
        input_obs_np = input_obs.numpy()
        return Tensor(_pad_2d_channel_cy(input_obs_np,
                                         self.param_pad))


    def _pad_conv_input(self):
        return Tensor(_pad_conv_input_cy(self.input.numpy(),
                                         self.param_pad))


    def _compute_output_obs(self, obs: Tensor):
        assert_dim(obs, 3)
        obs_np = obs.numpy()
        return Tensor(_compute_output_obs_cy(obs_np,
                                      self.param.numpy()))


    def _output(self):
        # import pdb; pdb.set_trace()
        return Tensor(_output_cy(self.input.numpy(),
                         self.param.numpy()))



    def _compute_grads_obs(self, input_obs: Tensor,
                           output_grad_obs: Tensor) -> Tensor:
        assert_dim(input_obs, 3)
        assert_dim(output_grad_obs, 3)
        input_obs_np = input_obs.numpy()
        output_grad_obs_np = output_grad_obs.numpy()
        return Tensor(_compute_grads_obs_cy(input_obs_np,
                                            output_grad_obs_np,
                                            self.param.numpy()))


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        assert_dim(output_grad, 4)
        output_grad_np = output_grad.numpy()
        return Tensor(_input_grad_cy(self.input.numpy(),
                                     output_grad_np,
                                     self.param.numpy()))


    def _param_grad(self, output_grad: Tensor) -> Tensor:
        assert_dim(output_grad, 4)
        output_grad_np = output_grad.numpy()
        return Tensor(_param_grad_cy(self.input.numpy(),
                       output_grad_np,
                       self.param.numpy()))


class PyTorchOperation(ParamOperation):

    def __init__(self, param: Tensor) -> Tensor:
        super().__init__(param)
        self.op = nn.Linear(param.shape[0],
                            param.shape[0])


    def _output(self) -> Tensor:

        self.input_with_grad = self.input.detach()
        self.input_with_grad.requires_grad = True

        return self.op(self.input_with_grad)


    def _input_grad(self, output_grad: Tensor) -> Tensor:

        self.output.backward(gradient=output_grad)

        return self.input_with_grad.grad

    def _param_grad(self, output_grad: Tensor) -> Tensor:

        return self.op.weight.grad


class Conv2D_Op_Pyt(PyTorchOperation):


    def __init__(self, param: Tensor):
        assert_dim(param, 4)
        super().__init__(param)
        self.param_size = self.param.shape[2]
        self.param_pad = self.param_size // 2
        self.in_channels = self.param.shape[1]
        self.out_channels = self.param.shape[0]
        self.op = nn.Conv2d(self.in_channels,
                            self.out_channels,
                            self.param_size,
                            padding=self.param_pad,
                            bias=False)
