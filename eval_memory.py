from transformers import GPT2Config, GPT2Model
import torch
from torch import nn

import copy
import tqdm
import argparse

from light_attention.models.gpt2 import LightGPT2Attention
from light_attention.profile import estimate_layer_memory, mem_usage


print('Pytorch version: {}'.format(torch.__version__))
assert torch.cuda.is_available(), 'Cuda is not available. The modules will work, but the memory estimation will not be correct.'


parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_layer', type=int, required=True, help='number of transformer blocks')
parser.add_argument('--n_head', type=int, required=True, help='number of attention heads')
parser.add_argument('--n_embd', type=int, required=True, help='embeddings dimention')
parser.add_argument('--n_positions', type=int, required=True, help='max context length')
parser.add_argument('--batch_size', type=int, required=True, help='batch size')
parser.add_argument('--light_softmax', action='store_true', help='flag that indicates whether to use light_softmax or pytorch softmax')
parser.add_argument('--drop_matmul', action='store_true', help='flag that indicates whether to use drop_matmul or unfused implementation of dropout and matmul')
parser.add_argument('--mixed_precision', action='store_true', help='flag that indicates whether to use mixed precision')
parser.add_argument('--seed', type=int, required=True, help='random seed')
parser.add_argument('--save_graph', action='store_true', help='flag that indicates whether to use save the backward computation graph (try not to save graphs for networks with more than a couple of transformer blocks since resolution is not enough to adequately depict it)')
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
configuration = GPT2Config(n_head=args.n_head, n_layer=args.n_layer, n_positions=args.n_positions, n_embd=args.n_embd, use_lightsoftmax=args.light_softmax, use_dropmatmul=args.drop_matmul)
model = GPT2Model(configuration)
if args.light_softmax or args.drop_matmul:
    for i in range(len(model.h)):
        model.h[i].attn = LightGPT2Attention(configuration)

b = args.batch_size
seq = configuration.n_positions
x = torch.randint(0, configuration.vocab_size, size=(b,seq), device='cuda')
if args.save_graph: 
    fout=f'lsoftmax={args.light_softmax}_dropmat={args.drop_matmul}_mixed={args.mixed_precision}'
else:
    fout = None
estimate_layer_memory(copy.deepcopy(model), x, device='cuda', input_shape=None, 
                      mixed_precision=args.mixed_precision, fout=fout)
# estimate_layer_memory(copy.deepcopy(model), x, device='cuda', input_shape=None, mixed_precision=args.mixed_precision)
