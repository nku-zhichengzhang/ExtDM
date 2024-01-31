from einops import rearrange
from einops_exts import rearrange_many
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc =  nn.Linear(in_features=32*28*28, out_features=10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
class SimpleAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(SimpleAttentionBlock, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        print(values.shape)
        values = self.values(values).reshape(N,  self.heads, value_len, self.head_dim)
        keys = self.keys(keys).reshape(N, self.heads, key_len, self.head_dim)
        queries = self.queries(queries).reshape(N,  self.heads, query_len, self.head_dim)


        energy = torch.matmul(queries, keys.transpose(-2, -1))        

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy, dim=3)
        out = torch.matmul(attention, values).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)

input_shape = (3, 28, 28)
input_shape_with_bs = (1, 3, 28, 28)
x = torch.rand(input_shape).cuda()
x_with_bs = torch.rand(input_shape_with_bs).cuda()
conv_model = SimpleConv().cuda()

def count_parameters(model):
    res = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"count_training_parameters: {res}")
    res = sum(p.numel() for p in model.parameters())
    print(f"count_all_parameters:      {res}")

# torchprofile 0.0.4
# 4202240 MACs 准确
# from torchprofile import profile_macs
# macs = profile_macs(conv_model,(x_with_bs,))
# print(macs)

# # ptflops 0.7.1.2
# # 4277514 MACs 把加法算进去了，很少用
# # 255978 Paras 准确
# from ptflops import get_model_complexity_info
# macs, params = get_model_complexity_info(
#     conv_model, input_shape, as_strings=False,
#     print_per_layer_stat=True, verbose=True
# )
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# # thop 0.1.1
# # 4202240 MACs 准确
# # 255978 Paras 准确
# from thop import profile
# macs, params = profile(conv_model, inputs=(x_with_bs, ))
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# # 准确
# # count_training_parameters: 255978
# # count_all_parameters:      255978
# count_parameters(conv_model)

# 有三个工具计算参数、三个工具计算MACs

##############################################################
    
# model = SimpleAttentionBlock(embed_size=256, heads=8).cuda()
# # Generate some sample data (batch of 5 sequences, each of length 10, embedding size 256)
# values = torch.randn(1, 10, 256).cuda()
# keys = torch.randn(1, 10, 256).cuda()
# queries = torch.randn(1, 10, 256).cuda()
# mask = None

# # torchprofile 0.0.4
# # 4202240 MACs 准确
# from torchprofile import profile_macs
# macs = profile_macs(model, (values, keys, queries, mask))
# print(macs)

# # ptflops 0.7.1.2
# # 4277514 MACs 把加法算进去了，很少用
# # 255978 Paras 准确
# from ptflops import get_model_complexity_info
# def aaa(res):
#     return dict(
#         values=values, 
#         keys=keys, 
#         queries=queries, 
#         mask=mask
#     )
# macs, params = get_model_complexity_info(
#     model, (0,0), as_strings=False,
#     print_per_layer_stat=True, verbose=False,
#     input_constructor=aaa
# )
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# # thop 0.1.1
# # 4202240 MACs 准确
# # 255978 Paras 准确
# from thop import profile
# macs, params = profile(model, inputs=(values, keys, queries, mask))
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
################################################################
def exists(x):
    return x is not None

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        return x
        # return self.act(x)
    
block1 = Block(3, 16, groups=8).cuda()
# Generate some sample data (batch of 5 sequences, each of length 10, embedding size 256)
x = torch.randn(1,3,10,64,64).cuda()

# torchprofile 0.0.4
# 18350080 MACs 准确
from torchprofile import profile_macs
macs = profile_macs(block1, (x,))
print(macs)

# ptflops 0.7.1.2
# 20316160 MACs 把加法算进去了，很少用
# 480 Paras 准确
from ptflops import get_model_complexity_info
def aaa(res):
    return dict(
        x=x
    )
macs, params = get_model_complexity_info(
    block1, (0,0), as_strings=False,
    print_per_layer_stat=True, verbose=False,
    input_constructor=aaa
)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# # thop 0.1.1
# 1769472 MACs 准确
# 480 Paras 准确
from thop import profile
macs, params = profile(block1, inputs=(x,))
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))