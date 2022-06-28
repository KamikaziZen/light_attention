from transformers import GPT2Config
import torch
from torch import nn
import copy
import tqdm
import argparse

from light_attention.models.gpt2 import LightGPT2LMHeadModel, LightGPT2Model
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
parser.add_argument('--seed', type=int, required=True, help='random seed')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
configuration = GPT2Config(n_head=args.n_head, n_layer=args.n_layer, n_positions=args.n_positions, n_embd=args.n_embd)
configuration.use_lightsoftmax = args.light_softmax
configuration.use_dropmatmul = args.drop_matmul
model = LightGPT2Model(configuration)
b = args.batch_size
seq = configuration.n_positions
x = torch.randint(0, configuration.vocab_size, size=(b,seq), device='cuda')
estimate_layer_memory(copy.deepcopy(model), x, device='cuda', input_shape=None)
torch.cuda.empty_cache()
