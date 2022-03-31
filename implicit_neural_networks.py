import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
pytorch에서 모델의 파라미터 개수를 셀 때 자주 사용하는 방식인가보다.

여기서도 나오고
https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
여기서도 언급하는 방식이고.
https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model


numel() : 입력 텐서의 총 원소 수를 반환한다.
이에 따라서 조금 더 해석해보자면, model.parameters()는 텐서들로 이뤄져 있는데,
이 각 텐서들의 원소 수를 모두 더하면 총 parameter 수를 구할 수 있다.
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


'''
논문에서 나온 positionalEncoding term과 비교해서 볼 것.
논문에서는 스칼라 값을 넘기거나 (1 x 3) 또는 (3 x 1) 벡터를 넘겼는데, 여기서는 텐서다.
어떻게 이용되는지 사용처를 봐야 할 것.

einsum에서 ij, k -> ijk는 Pseudocode로 표현하자면 이렇다.
for each i:
    for each j:
        for each k:
            output[i][j][k] = in_tensor[i][j] * b[k]
'''
def positionalEncoding_vec(in_tensor, b):

    # shape (batch, in_tensor.size(1), freqNum)
    proj = torch.einsum('ij, k -> ijk', in_tensor, b)
    # shape (batch, 2*in_tensor.size(1), freqNum)
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)

    return output


class IMLP(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=256,
            use_positional=True,
            positional_dim=10,
            skip_layers=[4, 6],
            num_layers=8,
            verbose=True, use_tanh=True, apply_softmax=False):
        super(IMLP, self).__init__()
        self.verbose = verbose
        self.use_tanh = use_tanh
        self.apply_softmax = apply_softmax
        if apply_softmax:
            self.softmax=nn.Softmax()
        if use_positional:
            encoding_dimensions = 2* input_dim * positional_dim
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(positional_dim)], requires_grad=False)
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim

            if i == num_layers -1:
                # last layer
                self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
            else:
                self.hidden.append(nn.Linear(input_dims, hidden_dim, bias=True))

        self.skip_layers = skip_layers
        self.num_layers = num_layers

        self.positional_dim = positional_dim
        self.use_positional = use_positional

        if self.verbose:
            print(f'Model has {count_parameters(self)} params')

    def forward(self, x):
        if self.use_positional:
            if self.b.device != x.device:
                self.b = self.b.to(x.device)
            pos = positionalEncoding_vec(x, self.b)
            x = pos

        # Tensor.detach() : 그래프에서 뗴어놓은 새로운 텐서를 반환한다.
        # 결과물은 gradient를 쓰지 않는다.
        # 이런 방식으로 복사해두면 x에 어떠한 연산을 진행하더라도 input에 미치는 영향은 없다.
        # 출처 : https://seducinghyeok.tistory.com/10
        input = x.detach.clone()
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), 1)
            x = layer(x)
        if self.use_tanh:
            x = torch.tanh(x)

        if self.apply_softmax:
            x = self.softmax(x)
        return x