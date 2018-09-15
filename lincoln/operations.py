import torch
from torch import Tensor

from .utils import assert_same_shape, assert_dim

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


    def _output(self) -> Tensor:
        raise NotImplementedError()


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()


class Flatten(Operation):
    def __init__(self):
        super().__init__()


    def _output(self) -> Tensor:
        return self.input.view(self.input.shape[0], -1)


    def _input_grad(self, output_grad: Tensor) -> Tensor:
        return output_grad.view(*self.input.shape)



class ParamOperation(Operation):

    def __init__(self, param: Tensor) -> Tensor:
        super().__init__()
        self.param = param


    def _compute_grads(self, output_grad: Tensor) -> Tensor:
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)


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


class Conv2D_Op(ParamOperation):


    def __init__(self, param: Tensor):
        assert_dim(param, 4)
        super().__init__(param)
        self.param_size = self.param.shape[0]
        self.param_pad = self.param_size // 2
        self.in_channels = self.param.shape[2]
        self.out_channels = self.param.shape[3]


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
        return torch.cat([other, inp_pad, other])


    def _pad_2d_batch(self, inp: Tensor):
        assert_dim(inp, 3)
        outs = [self._pad_2d_obs(obs) for obs in inp]
        return torch.stack(outs)


    def _select_channel(self, inp: Tensor, i: int):
        assert_dim(inp, 3)
        return torch.index_select(inp, dim=2, index=torch.LongTensor([i])).squeeze(2)


    def _pad_2d_channel(self, input_obs: Tensor):
        '''
        "inp" is a 3 dimensional tensor:
        * image width
        * image height
        * channels
        '''
        assert_dim(input_obs, 3)
        num_channels = input_obs.shape[2]
        return torch.stack([self._pad_2d_obs(self._select_channel(input_obs, i))
                            for i in range(num_channels)], dim=2)


    def _pad_conv_input(self):
        return torch.stack([self._pad_2d_channel(obs)
                            for obs in self.input])


    def _compute_output_obs(self, obs: Tensor):

        assert_dim(obs, 3)
        obs_pad = self._pad_2d_channel(obs)

        out = torch.zeros(obs.shape[:2] + (self.out_channels,))
        for c_out in range(self.out_channels):
            for c_in in range(self.in_channels):
                for o_w in range(out.shape[0]):
                    for o_h in range(out.shape[1]):
                        for p_w in range(self.param_size):
                            for p_h in range(self.param_size):
                                out[o_w][o_h][c_out] += \
                                self.param[p_w][p_h][c_in][c_out] * obs_pad[o_w+p_w][o_h+p_h][c_in]
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
                for i_w in range(input_obs.shape[0]):
                    for i_h in range(input_obs.shape[1]):
                        for p_w in range(self.param_size):
                            for p_h in range(self.param_size):
                                input_grad[i_w][i_h][c_in] += \
                                output_obs_pad[i_w+self.param_size-p_w-1][i_h+self.param_size-p_h-1][c_out] \
                                * self.param[p_w][p_h][c_in][c_out]

        return input_grad


    def _input_grad(self, output_grad: Tensor) -> Tensor:

        grads = [self._compute_grads_obs(self.input[i], output_grad[i]) for i in range(output_grad.shape[0])]

        return torch.stack(grads)


    def _param_grad(self, output_grad: Tensor) -> Tensor:

        inp_pad = self._pad_conv_input()

        param_grad = torch.zeros_like(self.param)

        for i in range(self.input.shape[0]):
            for c_in in range(self.in_channels):
                for c_out in range(self.out_channels):
                    for o_w in range(output_grad.shape[1]):
                        for o_h in range(output_grad.shape[2]):
                            for p_w in range(self.param_size):
                                for p_h in range(self.param_size):
                                    param_grad[p_w][p_h][c_in][c_out] += \
                                    inp_pad[i][o_w+p_w][o_h+p_h][c_in] \
                                    * output_grad[i][o_w][o_h][c_out]
        return param_grad
