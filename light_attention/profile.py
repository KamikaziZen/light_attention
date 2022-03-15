import copy
import gc
from collections import namedtuple
from distutils.version import LooseVersion
from graphviz import Digraph
import torch
from torch.autograd import Variable
import warnings


Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"


def get_fn_name(fn, show_attrs, max_attr_chars):
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = dict()
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX):]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width)+ 's'
    truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params


def make_dot(var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50, verbose=False):
    """ Produces Graphviz representation of PyTorch autograd graph.
    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:
     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.
     - Pink: Tensors saved by custom autograd functions.
    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
        show_attrs: whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved: whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
    """
    RES = {}
    if LooseVersion(torch.__version__) < LooseVersion("1.9") and \
            (show_attrs or show_saved):
        warnings.warn(
            "make_dot: showing grad_fn attributes and saved variables"
            " requires PyTorch version >= 1.9. (This does NOT apply to"
            " saved tensors saved by custom autograd functions.)")

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return '%s\n %s' % (name, size_to_str(var.size()))

    def add_nodes(fn):
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        if show_saved:
            # for custom autograd functions
            if hasattr(fn, 'saved_tensors'):
                for t in fn.saved_tensors:
                    seen.add(t)
                    dot.edge(str(t.data_ptr()), str(id(fn)), dir="none")
                    # priority over vanilla saved tensors
                    dot.node(str(t.data_ptr()), get_var_name(t, f'custom save {t.dtype}'), fillcolor='pink')
                    if t.data_ptr() in RES:
                        if verbose:
                            print(f"Replacing {RES[t.data_ptr()]} with \
                            {(get_var_name(t, f'custom_save'), t.numel() * t.element_size())}")
                    RES[t.data_ptr()] = (get_var_name(t, 'custom save'), t.numel() * t.element_size(), fn)

            # for vanilla functions
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(fn, attr)
                seen.add(val)
                attr = attr[len(SAVED_PREFIX):]
                if torch.is_tensor(val):
                    if verbose:
                        print('Edge from Fn to saved attribute', fn)
                        print()
                    dot.edge(str(id(fn)), str(val.data_ptr()), dir="none")
                    if val.data_ptr() in RES:
                        if verbose:
                            print(f'Replacing {RES[val.data_ptr()]} with \
                            {(get_var_name(val, attr), val.numel() * val.element_size())}')
                    else:
                        dot.node(str(val.data_ptr()), get_var_name(val, attr), fillcolor='orange')
                    RES[val.data_ptr()] = (get_var_name(val, attr), val.numel() * val.element_size(), fn)

                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
                            name = attr + '[%s]' % str(i)
                            if verbose:
                                print('Edge from Fn to saved attribute', fn, attr)
                                print()
                            dot.edge(str(id(fn)), str(t.data_ptr()), dir="none")
                            if t.data_ptr() in RES:
                                if verbose:
                                    print(f'Replacing {RES[t.data_ptr()]} with \
                                    {(get_var_name(t, name), t.numel() * t.element_size())}')
                            else:
                                dot.node(str(t.data_ptr()), get_var_name(t, name), fillcolor='orange')
                            RES[t.data_ptr()] = (get_var_name(t, name), t.numel() * t.element_size(), fn)

        if hasattr(fn, 'variable'):
            # if grad_accumulator, add the node for `.variable`
            var = fn.variable
            seen.add(var)
            if verbose:
                print('Edge from Fn to Variable', fn, get_var_name(var))
            dot.edge(str(var.data_ptr()), str(id(fn)))
            if var.data_ptr() in RES:
                if verbose:
                    print(f'Removing {RES[var.data_ptr()]} because of Variable')
            else:
                dot.node(str(var.data_ptr()), get_var_name(var), fillcolor='lightblue')
            RES[var.data_ptr()] = (None, None, fn)

        # add the node for this grad_fn
        dot.node(str(id(fn)), get_fn_name(fn, show_attrs, max_attr_chars))

        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(fn)))
                    add_nodes(u[0])

    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if (var.grad_fn):
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.edge(str(id(var._base)), str(id(var)), style="dotted")

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)

    return dot, RES


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def mem_usage():
    gc.collect()

    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()

    ma = torch.cuda.memory_allocated() / (1024 * 1024)
    max_ma = torch.cuda.max_memory_allocated() / (1024 * 1024)
    ca = torch.cuda.memory_reserved() / (1024 * 1024)
    max_ca = torch.cuda.max_memory_reserved() / (1024 * 1024)

    print(
        f"MA {round(ma, 4)} MB \
        Max_MA {round(max_ma, 4)} MB \
        CA {round(ca, 4)} MB \
        Max_CA {round(max_ca, 4)} MB "
    )
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()

    return (ma, max_ma, ca, max_ca)


def estimate_layer_memory(m, x=None, device='cuda', input_shape=None, fout=None, verbose=False):
    print('\nBefore placing the model on GPU')
    mem_stats_0 = mem_usage()

    m.to(device)
    print('\nAfter placing the model on GPU:')
    mem_stats_1 = mem_usage()
    print(f'\nParams (empirical) {round(mem_stats_1[0] - mem_stats_0[0], 4)} MB')

    param_bytes = 0
    for pname, p in m.named_parameters():
        # print(pname, p.shape)
        param_bytes += p.numel() * p.element_size()
    param_bytes = param_bytes * 1 / (1024 * 1024)
    print(f'\nParams (analytical, torch) {round(param_bytes, 4)} MB')

    if x is None:
        x = torch.randn(input_shape, device=device)
    print('\nAfter input batch generation, before forward pass:')
    mem_stats_2 = mem_usage()
    y = m(x)

    if isinstance(y, tuple):
        y = y[0]
        
    if hasattr(y, 'last_hidden_state'):
        y = y.last_hidden_state
        
    y = y.cos().mean()
    
    dot, RES = make_dot(y, params=dict(m.named_parameters()), show_attrs=True, show_saved=True, verbose=verbose)
    if fout is not None:
        dot.render(fout, view=False)
        print(f'\nGraph has been saved in {fout}.pdf.')

    print('\nAfter backward:')
    mem_stats_3 = mem_usage()

    print()
    act_bytes = 0
    for k, (name, el, fn) in RES.items():
        if el is None:
            continue
        if ("weight" in name) or ("bias" in name):
            continue
        act_bytes += el
        if verbose:
            print(k, fn, name, el / (1024 * 1024), )
            print()
    act_bytes = act_bytes * 1 / (1024 * 1024)
    print(f'\nActivations (analytical, torchviz) {round(act_bytes, 4)} MB')
    print(f'Activations (empirical) {round(mem_stats_3[0] - mem_stats_1[0], 4)} MB')

    x = m = None
    del x
    del m
    torch.cuda.empty_cache()