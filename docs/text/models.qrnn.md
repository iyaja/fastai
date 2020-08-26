# QRNN
> Quasi-recurrent neural networks introduced in <a href='https://arxiv.org/abs/1611.01576'>Bradbury et al.</a>


## ForgetMult

```python
__file__ = Path.cwd().parent/'fastai'/'text'/'models'/'qrnn.py'
```


<h4 id="load_cpp" class="doc_header"><code>load_cpp</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/qrnn.py#L16" class="source_link" style="float:right">[source]</a></h4>

> <code>load_cpp</code>(**`name`**, **`files`**, **`path`**)





<h4 id="dispatch_cuda" class="doc_header"><code>dispatch_cuda</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/qrnn.py#L41" class="source_link" style="float:right">[source]</a></h4>

> <code>dispatch_cuda</code>(**`cuda_class`**, **`cpu_func`**, **`x`**)

Depending on `x.device` uses `cpu_func` or `cuda_class.apply`


The ForgetMult gate is the quasi-recurrent part of the network, computing the following from `x` and `f`.
``` python
h[i+1] = x[i] * f[i] + h[i] + (1-f[i])
```
The initial value for `h[0]` is either a tensor of zeros or the previous hidden state.


<h4 id="forget_mult_CPU" class="doc_header"><code>forget_mult_CPU</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/qrnn.py#L46" class="source_link" style="float:right">[source]</a></h4>

> <code>forget_mult_CPU</code>(**`x`**, **`f`**, **`first_h`**=*`None`*, **`batch_first`**=*`True`*, **`backward`**=*`False`*)

ForgetMult gate applied to `x` and `f` on the CPU.


`first_h` is the tensor used for the value of `h[0]` (defaults to a tensor of zeros). If `batch_first=True`, `x` and `f` are expected to be of shape `batch_size x seq_length x n_hid`, otherwise they are expected to be of shape `seq_length x batch_size x n_hid`. If `backwards=True`, the elements in `x` and `f` on the sequence dimension are read in reverse.

```python
def manual_forget_mult(x, f, h=None, batch_first=True, backward=False):
    if batch_first: x,f = x.transpose(0,1),f.transpose(0,1)
    out = torch.zeros_like(x)
    prev = h if h is not None else torch.zeros_like(out[0])
    idx_range = range(x.shape[0]-1,-1,-1) if backward else range(x.shape[0])
    for i in idx_range:
        out[i] = f[i] * x[i] + (1-f[i]) * prev
        prev = out[i]
    if batch_first: out = out.transpose(0,1)
    return out

x,f = torch.randn(5,3,20).chunk(2, dim=2)
for (bf, bw) in [(True,True), (False,True), (True,False), (False,False)]:
    th_out = manual_forget_mult(x, f, batch_first=bf, backward=bw)
    out = forget_mult_CPU(x, f, batch_first=bf, backward=bw)
    test_close(th_out,out)
    h = torch.randn((5 if bf else 3), 10)
    th_out = manual_forget_mult(x, f, h=h, batch_first=bf, backward=bw)
    out = forget_mult_CPU(x, f, first_h=h, batch_first=bf, backward=bw)
    test_close(th_out,out)
```

```python
x = torch.randn(3,4,5)
x.size() + torch.Size([0,1,0])
```




    torch.Size([3, 4, 5, 0, 1, 0])




<h3 id="ForgetMultGPU" class="doc_header"><code>class</code> <code>ForgetMultGPU</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/qrnn.py#L61" class="source_link" style="float:right">[source]</a></h3>

> <code>ForgetMultGPU</code>() :: `Function`

Wraper around the CUDA kernels for the ForgetMult gate.


## QRNN


<h3 id="QRNNLayer" class="doc_header"><code>class</code> <code>QRNNLayer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/qrnn.py#L86" class="source_link" style="float:right">[source]</a></h3>

> <code>QRNNLayer</code>(**`input_size`**, **`hidden_size`**=*`None`*, **`save_prev_x`**=*`False`*, **`zoneout`**=*`0`*, **`window`**=*`1`*, **`output_gate`**=*`True`*, **`batch_first`**=*`True`*, **`backward`**=*`False`*) :: [`Module`](/torch_core.html#Module)

Apply a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.


```python
qrnn_fwd = QRNNLayer(10, 20, save_prev_x=True, zoneout=0, window=2, output_gate=True)
qrnn_bwd = QRNNLayer(10, 20, save_prev_x=True, zoneout=0, window=2, output_gate=True, backward=True)
qrnn_bwd.load_state_dict(qrnn_fwd.state_dict())
x_fwd = torch.randn(7,5,10)
x_bwd = x_fwd.clone().flip(1)
y_fwd,h_fwd = qrnn_fwd(x_fwd)
y_bwd,h_bwd = qrnn_bwd(x_bwd)
test_close(y_fwd, y_bwd.flip(1), eps=1e-4)
test_close(h_fwd, h_bwd, eps=1e-4)
y_fwd,h_fwd = qrnn_fwd(x_fwd, h_fwd)
y_bwd,h_bwd = qrnn_bwd(x_bwd, h_bwd)
test_close(y_fwd, y_bwd.flip(1), eps=1e-4)
test_close(h_fwd, h_bwd, eps=1e-4)
```


<h3 id="QRNN" class="doc_header"><code>class</code> <code>QRNN</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/qrnn.py#L137" class="source_link" style="float:right">[source]</a></h3>

> <code>QRNN</code>(**`input_size`**, **`hidden_size`**, **`n_layers`**=*`1`*, **`batch_first`**=*`True`*, **`dropout`**=*`0`*, **`bidirectional`**=*`False`*, **`save_prev_x`**=*`False`*, **`zoneout`**=*`0`*, **`window`**=*`None`*, **`output_gate`**=*`True`*) :: [`Module`](/torch_core.html#Module)

Apply a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.


```python
qrnn = QRNN(10, 20, 2, bidirectional=True, batch_first=True, window=2, output_gate=False)
x = torch.randn(7,5,10)
y,h = qrnn(x)
test_eq(y.size(), [7, 5, 40])
test_eq(h.size(), [4, 7, 20])
#Without an out gate, the last timestamp in the forward output is the second to last hidden
#and the first timestamp of the backward output is the last hidden
test_close(y[:,-1,:20], h[2])
test_close(y[:,0,20:], h[3])
```
