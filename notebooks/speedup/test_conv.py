# Add Lincoln to system path
import sys
sys.path.append("/Users/seth/development/lincoln/")

import time

import numpy as np
from numba import jit

from torch import Tensor
import torch

from lincoln.operations import Operation, ParamOperation, WeightMultiply
from lincoln.layers import Layer, Dense
from lincoln.activations import Activation, LinearAct

class Conv2D(ParamOperation):

    def __init__(self,
                 param: Tensor):
        super().__init__(param)
        self.param_size = param.shape[0]
        self.param_pad = self.param_size // 2

    def _pad_1d_obs(self, obs: Tensor) -> Tensor:
        z = torch.Tensor([0])
        z = z.repeat(self.param_pad)
        return torch.cat([z, obs, z])

    def _pad_1d(self, inp: Tensor) -> Tensor:
        outs = [self._pad_1d_obs(obs) for obs in inp]
        return torch.stack(outs)

    def _pad_2d_obs(self,
                    inp: Tensor):

        inp_pad = self._pad_1d(inp)
        other = torch.zeros(self.param_pad, inp.shape[0] + self.param_pad * 2)
        return torch.cat([other, inp_pad, other])

    def _pad_2d(self, inp: Tensor):

        outs = [_pad_2d_obs(obs, self.param_pad) for obs in inp]
        return torch.stack(outs)

    def _compute_output_obs(self,
                            obs: Tensor):
        '''
        Obs is a 2d square Tensor, so is param
        '''
        obs_pad = self._pad_2d_obs(obs)

        out = torch.zeros(obs.shape)

        for o_w in range(out.shape[0]):
            for o_h in range(out.shape[1]):
                for p_w in range(self.param_size):
                    for p_h in range(self.param_size):
                        out[o_w][o_h] += self.param[p_w][p_h] * obs_pad[o_w+p_w][o_h+p_h]
        return out

    def _compute_output(self):

        outs = [self._compute_output_obs(obs) for obs in self.input_]
        return torch.stack(outs)

    def _compute_grads_obs(self,
                           input_obs: Tensor,
                           output_grad_obs: Tensor) -> Tensor:

        output_obs_pad = self._pad_2d_obs(output_grad_obs)
        input_grad = torch.zeros_like(input_obs)

        for i_w in range(input_obs.shape[0]):
            for i_h in range(input_obs.shape[1]):
                for p_w in range(param_size):
                    for p_h in range(param_size):
                        input_grad[i_w][i_h] += output_obs_pad[i_w+self.param_size-p_w-1][i_h+self.param_size-p_h-1] \
                        * self.param[p_w][p_h]

        return input_grad

    def _compute_grads(self, output_grad: Tensor) -> Tensor:

        grads = [_compute_grads_obs(self.input_[i], output_grad[i], self.param) for i in range(output_grad.shape[0])]

        return torch.stack(grads)


    def _param_grad(inp: Tensor,
                    output_grad: Tensor,
                    param: Tensor) -> Tensor:

        param_size = param.shape[0]
        inp_pad = _pad_2d(inp, param_size // 2)

        param_grad = torch.zeros_like(param)
        img_shape = output_grad.shape[1:]

        for i in range(inp.shape[0]):
            for o_w in range(img_shape[0]):
                for o_h in range(img_shape[1]):
                    for p_w in range(param_size):
                        for p_h in range(param_size):
                            param_grad[p_w][p_h] += inp_pad[i][o_w+p_w][o_h+p_h] \
                            * output_grad[i][o_w][o_h]
        return param_grad

from conv_c import _pad_1d_obs_conv

class Conv2D_c(ParamOperation):

    def __init__(self,
                 param: Tensor):
        super().__init__(param)
        self.param_size = param.shape[0]
        self.param_pad = self.param_size // 2

    def _pad_1d_obs(self, obs: Tensor) -> Tensor:
        obs_np = obs.numpy()
        # import pdb; pdb.set_trace()
        a = Tensor(_pad_1d_obs_conv(obs_np, self.param_pad))
        # import pdb; pdb.set_trace()
        return a

    def _pad_1d(self, inp: Tensor) -> Tensor:
        outs = [self._pad_1d_obs(obs) for obs in inp]
        return torch.stack(outs)

    def _pad_2d_obs(self,
                    inp: Tensor):

        inp_pad = self._pad_1d(inp)
        other = torch.zeros(self.param_pad, inp.shape[0] + self.param_pad * 2)
        return torch.cat([other, inp_pad, other])

    def _pad_2d(self, inp: Tensor):

        outs = [_pad_2d_obs(obs, self.param_pad) for obs in inp]
        return torch.stack(outs)

    def _compute_output_obs(self,
                            obs: Tensor):
        '''
        Obs is a 2d square Tensor, so is param
        '''
        obs_pad = self._pad_2d_obs(obs)

        out = torch.zeros(obs.shape)

        for o_w in range(out.shape[0]):
            for o_h in range(out.shape[1]):
                for p_w in range(self.param_size):
                    for p_h in range(self.param_size):
                        out[o_w][o_h] += self.param[p_w][p_h] * obs_pad[o_w+p_w][o_h+p_h]
        return out

    def _compute_output(self):

        outs = [self._compute_output_obs(obs) for obs in self.input_]
        return torch.stack(outs)

    def _compute_grads_obs(self,
                           input_obs: Tensor,
                           output_grad_obs: Tensor) -> Tensor:

        output_obs_pad = self._pad_2d_obs(output_grad_obs)
        input_grad = torch.zeros_like(input_obs)

        for i_w in range(input_obs.shape[0]):
            for i_h in range(input_obs.shape[1]):
                for p_w in range(param_size):
                    for p_h in range(param_size):
                        input_grad[i_w][i_h] += output_obs_pad[i_w+self.param_size-p_w-1][i_h+self.param_size-p_h-1] \
                        * self.param[p_w][p_h]

        return input_grad

    def _compute_grads(self, output_grad: Tensor) -> Tensor:

        grads = [_compute_grads_obs(self.input_[i], output_grad[i], self.param) for i in range(output_grad.shape[0])]

        return torch.stack(grads)


    def _param_grad(inp: Tensor,
                    output_grad: Tensor,
                    param: Tensor) -> Tensor:

        param_size = param.shape[0]
        inp_pad = _pad_2d(inp, param_size // 2)

        param_grad = torch.zeros_like(param)
        img_shape = output_grad.shape[1:]

        for i in range(inp.shape[0]):
            for o_w in range(img_shape[0]):
                for o_h in range(img_shape[1]):
                    for p_w in range(param_size):
                        for p_h in range(param_size):
                            param_grad[p_w][p_h] += inp_pad[i][o_w+p_w][o_h+p_h] \
                            * output_grad[i][o_w][o_h]
        return param_grad


def main(fil: Tensor,
         imgs: Tensor):

    a = Conv2D(fil)
    start = time.time()
    a.forward(imgs)
    end = time.time()
    print("Forward took", round(end - start, 4), "seconds")

    b = Conv2D_c(fil)
    start = time.time()
    b.forward(imgs)
    end = time.time()
    print("Forward for Cython class took\n",
          round(end - start, 4), "seconds")


if __name__=="__main__":
    fil = Tensor(torch.empty(3, 3).uniform_(-1, 1))
    mnist_imgs = torch.load("data/img_batch")
    main(fil, mnist_imgs)
