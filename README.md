# <img align="left" alt="Icon" width="50px" src="./img/lightweight.png"> Light Attention

This project reduces memory requirements of attention mechanism. This is achieved by two adjustments: Softmax function that doesn't save inputs for backward computation and a merged Dropout + Matmul operation.

<img src="./img/LightAttention.png">

## Light Softmax
Attention mechanism uses softmax on the output of Q*K.T operation. Thus, both input and output of a Softmax layer allocate O(seq_length^2) memory. 
Vanilla Softmax implementation in PyTorch saves both input and output for a backward operation. 
Implementation in this repository computes gradients using only layer outputs. This allowes reduction in memory requirements by O(seq_length^2).

<img src="./img/softmax.svg">

To use LightSoftmax outside of attention block, import it with:
```
from light_attention.attention import light_softmax

light_softmax(x)
```

To use LightSoftmax in Attention block you can import either LightAttention module or whole LightGPT2 model:
```
from transformers import GPT2Config
from light_attention.attention import LightAttention, LightGPT2LMHeadModel, LightGPT2Model

config = GPT2Config(use_lightsoftmax=True)
attn = LightAttention(config)
model = LightGPT2Model(config)
```

## Merged DropMatmul
When using dropout before multiplying Softmax output (S) by Values tensor (V) PyTorch saves both input to Dropout and input to Matmul operation (both require O(seq_length^2) memory). This repository offers a merged Dropout + Marmul layer which computes gradients using only S and a Dropout mask. 

<img src="./img/dropmatmul.svg">


To use DropMatmul outside of attention block, import it with:
```
from light_attention.attention import drop_matmul

drop_matmul(x)
```

To use Dropmatmul in Attention block you can import either LightAttention module or whole LightGPT2 model:
```
from transformers import GPT2Config
from light_attention.attention import LightAttention, LightGPT2LMHeadModel, LightGPT2Model

config = GPT2Config(use_dropmatmul=True)
attn = LightAttention(config)
model = LightGPT2Model(config)
```

You can use both DropMatmul and LightAttention in the same block:
```
config = GPT2Config(use_lightsoftmax=True, use_dropmatmul=True)
```

## Benchmarks
(tested on torch==1.10. torch==1.11 negates the effect of lightsoftmax. probably because of functorch implementations. dropmatmul still gives about 20% memory reduction.)

Experiment was conducted on a single GPU NVIDIA A100 80Gb. Memory stats for a training loop of classic GPT2 model configurations(gpt2-small/-medium/-large/-xl) and batch_size==4:
| Model  | Max Memory Allocated, MB | Max Memory Reserved, MB |
|  :---:  |  :---:  |  :---:  |
| Vanilla gpt2-small | 12119.3 | 12654.0 |
| Light gpt2-small | 7799.3 | 7902.0 |
| Vanilla gpt2-medium | 32351.126 | 33558.0 |
| Light gpt2-medium | 20831.126 | 20870.0 |
| Vanilla gpt2-large | 61104.6338 | 63376.0 |
| Light gpt2-large | 39504.6338 | 39616.0 |
| Vanilla gpt2-xl | - | - |
| Light gpt2-xl | 67049.7734 | 67132.0 |

```-``` means that there was not enough memory to perform a single forward-backward iteration with this configuration.

### Attributions
Code in this repository is a modified version of gpt2 model from [huggingface transformers](https://github.com/huggingface/transformers).
Icon is taken from [Flaticon](https://www.flaticon.com/free-icons/lightweight).
