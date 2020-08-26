# Layers
> Custom fastai layers and basic functions to grab them.


## Basic manipulations and resize


<h4 id="module" class="doc_header"><code>module</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L21" class="source_link" style="float:right">[source]</a></h4>

> <code>module</code>(**\*`flds`**, **\*\*`defaults`**)

Decorator to create an `nn.Module` using `f` as `forward` method



<h3 id="Identity" class="doc_header"><code>class</code> <code>Identity</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>Identity</code>() :: [`Module`](/torch_core.html#Module)

Do nothing at all


```python
test_eq(Identity()(1), 1)
```


<h3 id="Lambda" class="doc_header"><code>class</code> <code>Lambda</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>Lambda</code>(**`func`**) :: [`Module`](/torch_core.html#Module)

An easy way to create a pytorch layer for a simple `func`


```python
def _add2(x): return x+2
tst = Lambda(_add2)
x = torch.randn(10,20)
test_eq(tst(x), x+2)
tst2 = pickle.loads(pickle.dumps(tst))
test_eq(tst2(x), x+2)
tst
```




    Lambda(func=_add2)




<h3 id="PartialLambda" class="doc_header"><code>class</code> <code>PartialLambda</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L57" class="source_link" style="float:right">[source]</a></h3>

> <code>PartialLambda</code>(**`func`**) :: [`Lambda`](/layers.html#Lambda)

Layer that applies `partial(func, **kwargs)`


```python
def test_func(a,b=2): return a+b
tst = PartialLambda(test_func, b=5)
test_eq(tst(x), x+5)
```


<h3 id="Flatten" class="doc_header"><code>class</code> <code>Flatten</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>Flatten</code>(**`full`**=*`False`*) :: [`Module`](/torch_core.html#Module)

Flatten `x` to a single dimension, e.g. at end of a model. `full` for rank-1 tensor


```python
tst = Flatten()
x = torch.randn(10,5,4)
test_eq(tst(x).shape, [10,20])
tst = Flatten(full=True)
test_eq(tst(x).shape, [200])
```


<h3 id="View" class="doc_header"><code>class</code> <code>View</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L73" class="source_link" style="float:right">[source]</a></h3>

> <code>View</code>(**\*`size`**) :: [`Module`](/torch_core.html#Module)

Reshape `x` to `size`


```python
tst = View(10,5,4)
test_eq(tst(x).shape, [10,5,4])
```


<h3 id="ResizeBatch" class="doc_header"><code>class</code> <code>ResizeBatch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L79" class="source_link" style="float:right">[source]</a></h3>

> <code>ResizeBatch</code>(**\*`size`**) :: [`Module`](/torch_core.html#Module)

Reshape `x` to `size`, keeping batch dim the same size


```python
tst = ResizeBatch(5,4)
test_eq(tst(x).shape, [10,5,4])
```


<h3 id="Debugger" class="doc_header"><code>class</code> <code>Debugger</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>Debugger</code>() :: [`Module`](/torch_core.html#Module)

A module to debug inside a model.



<h4 id="sigmoid_range" class="doc_header"><code>sigmoid_range</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L92" class="source_link" style="float:right">[source]</a></h4>

> <code>sigmoid_range</code>(**`x`**, **`low`**, **`high`**)

Sigmoid function with range `(low, high)`


```python
test = tensor([-10.,0.,10.])
assert torch.allclose(sigmoid_range(test, -1,  2), tensor([-1.,0.5, 2.]), atol=1e-4, rtol=1e-4)
assert torch.allclose(sigmoid_range(test, -5, -1), tensor([-5.,-3.,-1.]), atol=1e-4, rtol=1e-4)
assert torch.allclose(sigmoid_range(test,  2,  4), tensor([2.,  3., 4.]), atol=1e-4, rtol=1e-4)
```


<h3 id="SigmoidRange" class="doc_header"><code>class</code> <code>SigmoidRange</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>SigmoidRange</code>(**`low`**, **`high`**) :: [`Module`](/torch_core.html#Module)

Sigmoid module with range `(low, high)`


```python
tst = SigmoidRange(-1, 2)
assert torch.allclose(tst(test), tensor([-1.,0.5, 2.]), atol=1e-4, rtol=1e-4)
```

## Pooling layers


<h3 id="AdaptiveConcatPool1d" class="doc_header"><code>class</code> <code>AdaptiveConcatPool1d</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L103" class="source_link" style="float:right">[source]</a></h3>

> <code>AdaptiveConcatPool1d</code>(**`size`**=*`None`*) :: [`Module`](/torch_core.html#Module)

Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`



<h3 id="AdaptiveConcatPool2d" class="doc_header"><code>class</code> <code>AdaptiveConcatPool2d</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L112" class="source_link" style="float:right">[source]</a></h3>

> <code>AdaptiveConcatPool2d</code>(**`size`**=*`None`*) :: [`Module`](/torch_core.html#Module)

Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`


If the input is `bs x nf x h x h`, the output will be `bs x 2*nf x 1 x 1` if no size is passed or `bs x 2*nf x size x size`

```python
tst = AdaptiveConcatPool2d()
x = torch.randn(10,5,4,4)
test_eq(tst(x).shape, [10,10,1,1])
max1 = torch.max(x,    dim=2, keepdim=True)[0]
maxp = torch.max(max1, dim=3, keepdim=True)[0]
test_eq(tst(x)[:,:5], maxp)
test_eq(tst(x)[:,5:], x.mean(dim=[2,3], keepdim=True))
tst = AdaptiveConcatPool2d(2)
test_eq(tst(x).shape, [10,10,2,2])
```


<h3 id="PoolType" class="doc_header"><code>class</code> <code>PoolType</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L121" class="source_link" style="float:right">[source]</a></h3>

> <code>PoolType</code>()





<h4 id="adaptive_pool" class="doc_header"><code>adaptive_pool</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L124" class="source_link" style="float:right">[source]</a></h4>

> <code>adaptive_pool</code>(**`pool_type`**)





<h3 id="PoolFlatten" class="doc_header"><code>class</code> <code>PoolFlatten</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L128" class="source_link" style="float:right">[source]</a></h3>

> <code>PoolFlatten</code>(**`pool_type`**=*`'Avg'`*) :: `Sequential`

Combine `nn.AdaptiveAvgPool2d` and [`Flatten`](/layers.html#Flatten).


```python
tst = PoolFlatten()
test_eq(tst(x).shape, [10,5])
test_eq(tst(x), x.mean(dim=[2,3]))
```

## BatchNorm layers


<h4 id="BatchNorm" class="doc_header"><code>BatchNorm</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L146" class="source_link" style="float:right">[source]</a></h4>

> <code>BatchNorm</code>(**`nf`**, **`ndim`**=*`2`*, **`norm_type`**=*`<NormType.Batch: 1>`*, **`eps`**=*`1e-05`*, **`momentum`**=*`0.1`*, **`affine`**=*`True`*, **`track_running_stats`**=*`True`*)

BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`.



<h4 id="InstanceNorm" class="doc_header"><code>InstanceNorm</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L152" class="source_link" style="float:right">[source]</a></h4>

> <code>InstanceNorm</code>(**`nf`**, **`ndim`**=*`2`*, **`norm_type`**=*`<NormType.Instance: 5>`*, **`affine`**=*`True`*, **`eps`**:`float`=*`1e-05`*, **`momentum`**:`float`=*`0.1`*, **`track_running_stats`**:`bool`=*`False`*)

InstanceNorm layer with `nf` features and `ndim` initialized depending on `norm_type`.


`kwargs` are passed to `nn.BatchNorm` and can be `eps`, `momentum`, `affine` and `track_running_stats`.

```python
tst = BatchNorm(15)
assert isinstance(tst, nn.BatchNorm2d)
test_eq(tst.weight, torch.ones(15))
tst = BatchNorm(15, norm_type=NormType.BatchZero)
test_eq(tst.weight, torch.zeros(15))
tst = BatchNorm(15, ndim=1)
assert isinstance(tst, nn.BatchNorm1d)
tst = BatchNorm(15, ndim=3)
assert isinstance(tst, nn.BatchNorm3d)
```

```python
tst = InstanceNorm(15)
assert isinstance(tst, nn.InstanceNorm2d)
test_eq(tst.weight, torch.ones(15))
tst = InstanceNorm(15, norm_type=NormType.InstanceZero)
test_eq(tst.weight, torch.zeros(15))
tst = InstanceNorm(15, ndim=1)
assert isinstance(tst, nn.InstanceNorm1d)
tst = InstanceNorm(15, ndim=3)
assert isinstance(tst, nn.InstanceNorm3d)
```

If `affine` is false the weight should be `None`

```python
test_eq(BatchNorm(15, affine=False).weight, None)
test_eq(InstanceNorm(15, affine=False).weight, None)
```


<h3 id="BatchNorm1dFlat" class="doc_header"><code>class</code> <code>BatchNorm1dFlat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L158" class="source_link" style="float:right">[source]</a></h3>

> <code>BatchNorm1dFlat</code>(**`num_features`**, **`eps`**=*`1e-05`*, **`momentum`**=*`0.1`*, **`affine`**=*`True`*, **`track_running_stats`**=*`True`*) :: `BatchNorm1d`

`nn.BatchNorm1d`, but first flattens leading dimensions


```python
tst = BatchNorm1dFlat(15)
x = torch.randn(32, 64, 15)
y = tst(x)
mean = x.mean(dim=[0,1])
test_close(tst.running_mean, 0*0.9 + mean*0.1)
var = (x-mean).pow(2).mean(dim=[0,1])
test_close(tst.running_var, 1*0.9 + var*0.1, eps=1e-4)
test_close(y, (x-mean)/torch.sqrt(var+1e-5) * tst.weight + tst.bias, eps=1e-4)
```


<h3 id="LinBnDrop" class="doc_header"><code>class</code> <code>LinBnDrop</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L167" class="source_link" style="float:right">[source]</a></h3>

> <code>LinBnDrop</code>(**`n_in`**, **`n_out`**, **`bn`**=*`True`*, **`p`**=*`0.0`*, **`act`**=*`None`*, **`lin_first`**=*`False`*) :: `Sequential`

Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers


The [`BatchNorm`](/layers.html#BatchNorm) layer is skipped if `bn=False`, as is the dropout if `p=0.`. Optionally, you can add an activation for after the linear layer with `act`.

```python
tst = LinBnDrop(10, 20)
mods = list(tst.children())
test_eq(len(mods), 2)
assert isinstance(mods[0], nn.BatchNorm1d)
assert isinstance(mods[1], nn.Linear)

tst = LinBnDrop(10, 20, p=0.1)
mods = list(tst.children())
test_eq(len(mods), 3)
assert isinstance(mods[0], nn.BatchNorm1d)
assert isinstance(mods[1], nn.Dropout)
assert isinstance(mods[2], nn.Linear)

tst = LinBnDrop(10, 20, act=nn.ReLU(), lin_first=True)
mods = list(tst.children())
test_eq(len(mods), 3)
assert isinstance(mods[0], nn.Linear)
assert isinstance(mods[1], nn.ReLU)
assert isinstance(mods[2], nn.BatchNorm1d)

tst = LinBnDrop(10, 20, bn=False)
mods = list(tst.children())
test_eq(len(mods), 1)
assert isinstance(mods[0], nn.Linear)
```

## Inits


<h4 id="sigmoid" class="doc_header"><code>sigmoid</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L178" class="source_link" style="float:right">[source]</a></h4>

> <code>sigmoid</code>(**`input`**, **`eps`**=*`1e-07`*)

Same as `torch.sigmoid`, plus clamping to `(eps,1-eps)



<h4 id="sigmoid_" class="doc_header"><code>sigmoid_</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L183" class="source_link" style="float:right">[source]</a></h4>

> <code>sigmoid_</code>(**`input`**, **`eps`**=*`1e-07`*)

Same as `torch.sigmoid_`, plus clamping to `(eps,1-eps)



<h4 id="vleaky_relu" class="doc_header"><code>vleaky_relu</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L191" class="source_link" style="float:right">[source]</a></h4>

> <code>vleaky_relu</code>(**`input`**, **`inplace`**=*`True`*)

`F.leaky_relu` with 0.3 slope



<h4 id="init_default" class="doc_header"><code>init_default</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L683" class="source_link" style="float:right">[source]</a></h4>

> <code>init_default</code>(**`m`**, **`func`**=*`kaiming_normal_`*)

Initialize `m` weights with `func` and set `bias` to 0.



<h4 id="init_linear" class="doc_header"><code>init_linear</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L212" class="source_link" style="float:right">[source]</a></h4>

> <code>init_linear</code>(**`m`**, **`act_func`**=*`None`*, **`init`**=*`'auto'`*, **`bias_std`**=*`0.01`*)




## Convolutions


<h3 id="ConvLayer" class="doc_header"><code>class</code> <code>ConvLayer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L232" class="source_link" style="float:right">[source]</a></h3>

> <code>ConvLayer</code>(**`ni`**, **`nf`**, **`ks`**=*`3`*, **`stride`**=*`1`*, **`padding`**=*`None`*, **`bias`**=*`None`*, **`ndim`**=*`2`*, **`norm_type`**=*`<NormType.Batch: 1>`*, **`bn_1st`**=*`True`*, **`act_cls`**=*`ReLU`*, **`transpose`**=*`False`*, **`init`**=*`'auto'`*, **`xtra`**=*`None`*, **`bias_std`**=*`0.01`*, **`dilation`**:`Union`\[`int`, `Tuple`\[`int`, `int`\]\]=*`1`*, **`groups`**:`int`=*`1`*, **`padding_mode`**:`str`=*`'zeros'`*) :: `Sequential`

Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers.


The convolution uses `ks` (kernel size) `stride`, `padding` and `bias`. `padding` will default to the appropriate value (`(ks-1)//2` if it's not a transposed conv) and `bias` will default to `True` the `norm_type` is `Spectral` or `Weight`, `False` if it's `Batch` or `BatchZero`. Note that if you don't want any normalization, you should pass `norm_type=None`.

This defines a conv layer with `ndim` (1,2 or 3) that will be a ConvTranspose if `transpose=True`. `act_cls` is the class of the activation function to use (instantiated inside). Pass `act=None` if you don't want an activation function. If you quickly want to change your default activation, you can change the value of [`defaults.activation`](/layers.html#defaults.activation).

`init` is used to initialize the weights (the bias are initialized to 0) and `xtra` is an optional layer to add at the end.

```python
tst = ConvLayer(16, 32)
mods = list(tst.children())
test_eq(len(mods), 3)
test_eq(mods[1].weight, torch.ones(32))
test_eq(mods[0].padding, (1,1))
```

```python
x = torch.randn(64, 16, 8, 8)#.cuda()
```

```python
test_eq(tst(x).shape, [64,32,8,8])
```

```python
tst = ConvLayer(16, 32, stride=2)
test_eq(tst(x).shape, [64,32,4,4])
```

```python
tst = ConvLayer(16, 32, padding=0)
test_eq(tst(x).shape, [64,32,6,6])
```

```python
assert mods[0].bias is None
#But can be overridden with `bias=True`
tst = ConvLayer(16, 32, bias=True)
assert first(tst.children()).bias is not None
#For no norm, or spectral/weight, bias is True by default
for t in [None, NormType.Spectral, NormType.Weight]:
    tst = ConvLayer(16, 32, norm_type=t)
    assert first(tst.children()).bias is not None
```

```python
tst = ConvLayer(16, 32, ndim=3)
assert isinstance(list(tst.children())[0], nn.Conv3d)
tst = ConvLayer(16, 32, ndim=1, transpose=True)
assert isinstance(list(tst.children())[0], nn.ConvTranspose1d)
```

```python
tst = ConvLayer(16, 32, ndim=3, act_cls=None)
mods = list(tst.children())
test_eq(len(mods), 2)
tst = ConvLayer(16, 32, ndim=3, act_cls=partial(nn.LeakyReLU, negative_slope=0.1))
mods = list(tst.children())
test_eq(len(mods), 3)
assert isinstance(mods[2], nn.LeakyReLU)
```

```python
# def linear(in_features, out_features, bias=True, act_cls=None, init='auto'):
#     "Linear layer followed by optional activation, with optional auto-init"
#     res = nn.Linear(in_features, out_features, bias=bias)
#     if act_cls: act_cls = act_cls()
#     init_linear(res, act_cls, init=init)
#     if act_cls: res = nn.Sequential(res, act_cls)
#     return res
```

```python
# @delegates(ConvLayer)
# def conv1d(ni, nf, ks, stride=1, ndim=1, norm_type=None, **kwargs):
#     "Convolutional layer followed by optional activation, with optional auto-init"
#     return ConvLayer(ni, nf, ks, stride=stride, ndim=ndim, norm_type=norm_type, **kwargs)
```

```python
# @delegates(ConvLayer)
# def conv2d(ni, nf, ks, stride=1, ndim=2, norm_type=None, **kwargs):
#     "Convolutional layer followed by optional activation, with optional auto-init"
#     return ConvLayer(ni, nf, ks, stride=stride, ndim=ndim, norm_type=norm_type, **kwargs)
```

```python
# @delegates(ConvLayer)
# def conv3d(ni, nf, ks, stride=1, ndim=3, norm_type=None, **kwargs):
#     "Convolutional layer followed by optional activation, with optional auto-init"
#     return ConvLayer(ni, nf, ks, stride=stride, ndim=ndim, norm_type=norm_type, **kwargs)
```


<h4 id="AdaptiveAvgPool" class="doc_header"><code>AdaptiveAvgPool</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L258" class="source_link" style="float:right">[source]</a></h4>

> <code>AdaptiveAvgPool</code>(**`sz`**=*`1`*, **`ndim`**=*`2`*)

nn.AdaptiveAvgPool layer for `ndim`



<h4 id="MaxPool" class="doc_header"><code>MaxPool</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L264" class="source_link" style="float:right">[source]</a></h4>

> <code>MaxPool</code>(**`ks`**=*`2`*, **`stride`**=*`None`*, **`padding`**=*`0`*, **`ndim`**=*`2`*, **`ceil_mode`**=*`False`*)

nn.MaxPool layer for `ndim`



<h4 id="AvgPool" class="doc_header"><code>AvgPool</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L270" class="source_link" style="float:right">[source]</a></h4>

> <code>AvgPool</code>(**`ks`**=*`2`*, **`stride`**=*`None`*, **`padding`**=*`0`*, **`ndim`**=*`2`*, **`ceil_mode`**=*`False`*)

nn.AvgPool layer for `ndim`


## fastai loss functions

The following class if the base class to warp a loss function it provides several added functionality:
- it flattens the tensors before trying to take the losses since it's more convenient (with a potential tranpose to put `axis` at the end)
- it has a potential `activation` method that tells the library if there is an activation fused in the loss (useful for inference and methods such as [`Learner.get_preds`](/learner.html#Learner.get_preds) or [`Learner.predict`](/learner.html#Learner.predict))
- it has a potential <code>decodes</code> method that is used on predictions in inference (for instance, an argmax in classification)

```python
F.binary_cross_entropy_with_logits(torch.randn(4,5), torch.randint(0, 2, (4,5)).float(), reduction='none')
```




    tensor([[0.4444, 1.1849, 1.1411, 2.2376, 0.4800],
            [3.0970, 0.2376, 0.2159, 2.0667, 0.5246],
            [0.7885, 0.7743, 0.5355, 0.6340, 1.5417],
            [0.5340, 0.4066, 0.9115, 0.5817, 0.2920]])



```python
funcs_kwargs
```




    <function fastcore.foundation.funcs_kwargs(cls)>




<h3 id="BaseLoss" class="doc_header"><code>class</code> <code>BaseLoss</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L277" class="source_link" style="float:right">[source]</a></h3>

> <code>BaseLoss</code>(**`loss_cls`**, **\*`args`**, **`axis`**=*`-1`*, **`flatten`**=*`True`*, **`floatify`**=*`False`*, **`is_2d`**=*`True`*, **\*\*`kwargs`**)

Same as `loss_cls`, but flattens input and target.


The `args` and `kwargs` will be passed to `loss_cls` during the initialization to instantiate a loss function. `axis` is put at the end for losses like softmax that are often performed on the last axis. If `floatify=True` the targs will be converted to float (useful for losses that only accept float targets like `BCEWithLogitsLoss`) and `is_2d` determines if we flatten while keeping the first dimension (batch size) or completely flatten the input. We want the first for losses like Cross Entropy, and the second for pretty much anything else.


<h3 id="CrossEntropyLossFlat" class="doc_header"><code>class</code> <code>CrossEntropyLossFlat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L302" class="source_link" style="float:right">[source]</a></h3>

> <code>CrossEntropyLossFlat</code>(**\*`args`**, **`axis`**=*`-1`*, **`weight`**=*`None`*, **`ignore_index`**=*`-100`*, **`reduction`**=*`'mean'`*, **`flatten`**=*`True`*, **`floatify`**=*`False`*, **`is_2d`**=*`True`*) :: [`BaseLoss`](/layers.html#BaseLoss)

Same as `nn.CrossEntropyLoss`, but flattens input and target.


```python
tst = CrossEntropyLossFlat()
output = torch.randn(32, 5, 10)
target = torch.randint(0, 10, (32,5))
#nn.CrossEntropy would fail with those two tensors, but not our flattened version.
_ = tst(output, target)
test_fail(lambda x: nn.CrossEntropyLoss()(output,target))

#Associated activation is softmax
test_eq(tst.activation(output), F.softmax(output, dim=-1))
#This loss function has a decodes which is argmax
test_eq(tst.decodes(output), output.argmax(dim=-1))
```

```python
tst = CrossEntropyLossFlat(axis=1)
output = torch.randn(32, 5, 128, 128)
target = torch.randint(0, 5, (32, 128, 128))
_ = tst(output, target)

test_eq(tst.activation(output), F.softmax(output, dim=1))
test_eq(tst.decodes(output), output.argmax(dim=1))
```


<h3 id="BCEWithLogitsLossFlat" class="doc_header"><code>class</code> <code>BCEWithLogitsLossFlat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L313" class="source_link" style="float:right">[source]</a></h3>

> <code>BCEWithLogitsLossFlat</code>(**\*`args`**, **`axis`**=*`-1`*, **`floatify`**=*`True`*, **`thresh`**=*`0.5`*, **`weight`**=*`None`*, **`reduction`**=*`'mean'`*, **`pos_weight`**=*`None`*, **`flatten`**=*`True`*, **`is_2d`**=*`True`*) :: [`BaseLoss`](/layers.html#BaseLoss)

Same as `nn.CrossEntropyLoss`, but flattens input and target.


```python
tst = BCEWithLogitsLossFlat()
output = torch.randn(32, 5, 10)
target = torch.randn(32, 5, 10)
#nn.BCEWithLogitsLoss would fail with those two tensors, but not our flattened version.
_ = tst(output, target)
test_fail(lambda x: nn.BCEWithLogitsLoss()(output,target))
output = torch.randn(32, 5)
target = torch.randint(0,2,(32, 5))
#nn.BCEWithLogitsLoss would fail with int targets but not our flattened version.
_ = tst(output, target)
test_fail(lambda x: nn.BCEWithLogitsLoss()(output,target))

#Associated activation is sigmoid
test_eq(tst.activation(output), torch.sigmoid(output))
```


<h4 id="BCELossFlat" class="doc_header"><code>BCELossFlat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L324" class="source_link" style="float:right">[source]</a></h4>

> <code>BCELossFlat</code>(**\*`args`**, **`axis`**=*`-1`*, **`floatify`**=*`True`*, **`weight`**=*`None`*, **`reduction`**=*`'mean'`*)

Same as `nn.BCELoss`, but flattens input and target.


```python
tst = BCELossFlat()
output = torch.sigmoid(torch.randn(32, 5, 10))
target = torch.randint(0,2,(32, 5, 10))
_ = tst(output, target)
test_fail(lambda x: nn.BCELoss()(output,target))
```


<h4 id="MSELossFlat" class="doc_header"><code>MSELossFlat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L331" class="source_link" style="float:right">[source]</a></h4>

> <code>MSELossFlat</code>(**\*`args`**, **`axis`**=*`-1`*, **`floatify`**=*`True`*, **`reduction`**=*`'mean'`*)

Same as `nn.MSELoss`, but flattens input and target.


```python
tst = MSELossFlat()
output = torch.sigmoid(torch.randn(32, 5, 10))
target = torch.randint(0,2,(32, 5, 10))
_ = tst(output, target)
test_fail(lambda x: nn.MSELoss()(output,target))
```


<h4 id="L1LossFlat" class="doc_header"><code>L1LossFlat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L338" class="source_link" style="float:right">[source]</a></h4>

> <code>L1LossFlat</code>(**\*`args`**, **`axis`**=*`-1`*, **`floatify`**=*`True`*, **`reduction`**=*`'mean'`*)

Same as `nn.L1Loss`, but flattens input and target.



<h3 id="LabelSmoothingCrossEntropy" class="doc_header"><code>class</code> <code>LabelSmoothingCrossEntropy</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L346" class="source_link" style="float:right">[source]</a></h3>

> <code>LabelSmoothingCrossEntropy</code>(**`eps`**:`float`=*`0.1`*, **`reduction`**=*`'mean'`*) :: [`Module`](/torch_core.html#Module)

Same as `nn.Module`, but no need for subclasses to call `super().__init__`


On top of the formula we define:
- a `reduction` attribute, that will be used when we call [`Learner.get_preds`](/learner.html#Learner.get_preds)
- an `activation` function that represents the activation fused in the loss (since we use cross entropy behind the scenes). It will be applied to the output of the model when calling [`Learner.get_preds`](/learner.html#Learner.get_preds) or [`Learner.predict`](/learner.html#Learner.predict)
- a <code>decodes</code> function that converts the output of the model to a format similar to the target (here indices). This is used in [`Learner.predict`](/learner.html#Learner.predict) and [`Learner.show_results`](/learner.html#Learner.show_results) to decode the predictions 


<h3 id="LabelSmoothingCrossEntropyFlat" class="doc_header"><code>class</code> <code>LabelSmoothingCrossEntropyFlat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L365" class="source_link" style="float:right">[source]</a></h3>

> <code>LabelSmoothingCrossEntropyFlat</code>(**\*`args`**, **`axis`**=*`-1`*, **`eps`**=*`0.1`*, **`reduction`**=*`'mean'`*, **`flatten`**=*`True`*, **`floatify`**=*`False`*, **`is_2d`**=*`True`*) :: [`BaseLoss`](/layers.html#BaseLoss)

Same as [`LabelSmoothingCrossEntropy`](/layers.html#LabelSmoothingCrossEntropy), but flattens input and target.


## Embeddings


<h4 id="trunc_normal_" class="doc_header"><code>trunc_normal_</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L374" class="source_link" style="float:right">[source]</a></h4>

> <code>trunc_normal_</code>(**`x`**, **`mean`**=*`0.0`*, **`std`**=*`1.0`*)

Truncated normal initialization (approximation)



<h3 id="Embedding" class="doc_header"><code>class</code> <code>Embedding</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L380" class="source_link" style="float:right">[source]</a></h3>

> <code>Embedding</code>(**`ni`**, **`nf`**) :: [`Embedding`](/layers.html#Embedding)

Embedding layer with truncated normal initialization


Truncated normal initialization bounds the distribution to avoid large value. For a given standard deviation `std`, the bounds are roughly `-std`, `std`.

```python
tst = Embedding(10, 30)
assert tst.weight.min() > -0.02
assert tst.weight.max() < 0.02
test_close(tst.weight.mean(), 0, 1e-2)
test_close(tst.weight.std(), 0.01, 0.1)
```

## Self attention


<h3 id="SelfAttention" class="doc_header"><code>class</code> <code>SelfAttention</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L387" class="source_link" style="float:right">[source]</a></h3>

> <code>SelfAttention</code>(**`n_channels`**) :: [`Module`](/torch_core.html#Module)

Self attention layer for `n_channels`.


Self-attention layer as introduced in [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318).

Initially, no change is done to the input. This is controlled by a trainable parameter named `gamma` as we return `x + gamma * out`.

```python
tst = SelfAttention(16)
x = torch.randn(32, 16, 8, 8)
test_eq(tst(x),x)
```

Then during training `gamma` will probably change since it's a trainable parameter. Let's see what's happening when it gets a nonzero value.

```python
tst.gamma.data.fill_(1.)
y = tst(x)
test_eq(y.shape, [32,16,8,8])
```

The attention mechanism requires three matrix multiplications (here represented by 1x1 convs). The multiplications are done on the channel level (the second dimension in our tensor) and we flatten the feature map (which is 8x8 here). As in the paper, we note `f`, `g` and `h` the results of those multiplications.

```python
q,k,v = tst.query[0].weight.data,tst.key[0].weight.data,tst.value[0].weight.data
test_eq([q.shape, k.shape, v.shape], [[2, 16, 1], [2, 16, 1], [16, 16, 1]])
f,g,h = map(lambda m: x.view(32, 16, 64).transpose(1,2) @ m.squeeze().t(), [q,k,v])
test_eq([f.shape, g.shape, h.shape], [[32,64,2], [32,64,2], [32,64,16]])
```

The key part of the attention layer is to compute attention weights for each of our location in the feature map (here 8x8 = 64). Those are positive numbers that sum to 1 and tell the model to pay attention to this or that part of the picture. We make the product of `f` and the transpose of `g` (to get something of size bs by 64 by 64) then apply a softmax on the first dimension (to get the positive numbers that sum up to 1). The result can then be multiplied with `h` transposed to get an output of size bs by channels by 64, which we can then be viewed as an output the same size as the original input. 

The final result is then `x + gamma * out` as we saw before.

```python
beta = F.softmax(torch.bmm(f, g.transpose(1,2)), dim=1)
test_eq(beta.shape, [32, 64, 64])
out = torch.bmm(h.transpose(1,2), beta)
test_eq(out.shape, [32, 16, 64])
test_close(y, x + out.view(32, 16, 8, 8), eps=1e-4)
```


<h3 id="PooledSelfAttention2d" class="doc_header"><code>class</code> <code>PooledSelfAttention2d</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L406" class="source_link" style="float:right">[source]</a></h3>

> <code>PooledSelfAttention2d</code>(**`n_channels`**) :: [`Module`](/torch_core.html#Module)

Pooled self attention layer for 2d.


Self-attention layer used in the [Big GAN paper](https://arxiv.org/abs/1809.11096).

It uses the same attention as in [`SelfAttention`](/layers.html#SelfAttention) but adds a max pooling of stride 2 before computing the matrices `g` and `h`: the attention is ported on one of the 2x2 max-pooled window, not the whole feature map. There is also a final matrix product added at the end to the output, before retuning `gamma * out + x`.


<h3 id="SimpleSelfAttention" class="doc_header"><code>class</code> <code>SimpleSelfAttention</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L435" class="source_link" style="float:right">[source]</a></h3>

> <code>SimpleSelfAttention</code>(**`n_in`**:`int`, **`ks`**=*`1`*, **`sym`**=*`False`*) :: [`Module`](/torch_core.html#Module)

Same as `nn.Module`, but no need for subclasses to call `super().__init__`


## PixelShuffle

PixelShuffle introduced in [this article](https://arxiv.org/pdf/1609.05158.pdf) to avoid checkerboard artifacts when upsampling images. If we want an output with `ch_out` filters, we use a convolution with `ch_out * (r**2)` filters, where `r` is the upsampling factor. Then we reorganize those filters like in the picture below:

{% include image.html alt="Pixelshuffle" style="width: 100%; height: auto;" file="/images/pixelshuffle.png" %}


<h4 id="icnr_init" class="doc_header"><code>icnr_init</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L457" class="source_link" style="float:right">[source]</a></h4>

> <code>icnr_init</code>(**`x`**, **`scale`**=*`2`*, **`init`**=*`kaiming_normal_`*)

ICNR init of `x`, with `scale` and `init` function


ICNR init was introduced in [this article](https://arxiv.org/abs/1707.02937). It suggests to initialize the convolution that will be used in PixelShuffle so that each of the `r**2` channels get the same weight (so that in the picture above, the 9 colors in a 3 by 3 window are initially the same).
{% include note.html content='This is done on the first dimension because PyTorch stores the weights of a convolutional layer in this format: `ch_out x ch_in x ks x ks`. ' %}

```python
tst = torch.randn(16*4, 32, 1, 1)
tst = icnr_init(tst)
for i in range(0,16*4,4):
    test_eq(tst[i],tst[i+1])
    test_eq(tst[i],tst[i+2])
    test_eq(tst[i],tst[i+3])
```


<h3 id="PixelShuffle_ICNR" class="doc_header"><code>class</code> <code>PixelShuffle_ICNR</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L467" class="source_link" style="float:right">[source]</a></h3>

> <code>PixelShuffle_ICNR</code>(**`ni`**, **`nf`**=*`None`*, **`scale`**=*`2`*, **`blur`**=*`False`*, **`norm_type`**=*`<NormType.Weight: 3>`*, **`act_cls`**=*`ReLU`*) :: `Sequential`

Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`.


The convolutional layer is initialized with [`icnr_init`](/layers.html#icnr_init) and passed `act_cls` and `norm_type` (the default of weight normalization seemed to be what's best for super-resolution problems, in our experiments). 

The `blur` option comes from [Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts](https://arxiv.org/abs/1806.02658) where the authors add a little bit of blur to completely get rid of checkerboard artifacts.

```python
psfl = PixelShuffle_ICNR(16, norm_type=None) #Deactivate weight norm as it changes the weight
x = torch.randn(64, 16, 8, 8)
y = psfl(x)
test_eq(y.shape, [64, 16, 16, 16])
#ICNR init makes every 2x2 window (stride 2) have the same elements
for i in range(0,16,2):
    for j in range(0,16,2):
        test_eq(y[:,:,i,j],y[:,:,i+1,j])
        test_eq(y[:,:,i,j],y[:,:,i  ,j+1])
        test_eq(y[:,:,i,j],y[:,:,i+1,j+1])
```

## Sequential extensions


<h4 id="sequential" class="doc_header"><code>sequential</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L479" class="source_link" style="float:right">[source]</a></h4>

> <code>sequential</code>(**\*`args`**)

Create an `nn.Sequential`, wrapping items with [`Lambda`](/layers.html#Lambda) if needed



<h3 id="SequentialEx" class="doc_header"><code>class</code> <code>SequentialEx</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L488" class="source_link" style="float:right">[source]</a></h3>

> <code>SequentialEx</code>(**\*`layers`**) :: [`Module`](/torch_core.html#Module)

Like `nn.Sequential`, but with ModuleList semantics, and can access module input


This is useful to write layers that require to remember the input (like a resnet block) in a sequential way.


<h3 id="MergeLayer" class="doc_header"><code>class</code> <code>MergeLayer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L508" class="source_link" style="float:right">[source]</a></h3>

> <code>MergeLayer</code>(**`dense`**:`bool`=*`False`*) :: [`Module`](/torch_core.html#Module)

Merge a shortcut with the result of the module by adding them or concatenating them if `dense=True`.


```python
res_block = SequentialEx(ConvLayer(16, 16), ConvLayer(16,16))
res_block.append(MergeLayer()) # just to test append - normally it would be in init params
x = torch.randn(32, 16, 8, 8)
y = res_block(x)
test_eq(y.shape, [32, 16, 8, 8])
test_eq(y, x + res_block[1](res_block[0](x)))
```

## Concat

Equivalent to keras.layers.Concatenate, it will concat the outputs of a ModuleList over a given dimension (default the filter dimension)


<h3 id="Cat" class="doc_header"><code>class</code> <code>Cat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L514" class="source_link" style="float:right">[source]</a></h3>

> <code>Cat</code>(**`layers`**, **`dim`**=*`1`*) :: `ModuleList`

Concatenate layers outputs over a given dim


```python
layers = [ConvLayer(2,4), ConvLayer(2,4), ConvLayer(2,4)] 
x = torch.rand(1,2,8,8) 
cat = Cat(layers) 
test_eq(cat(x).shape, [1,12,8,8]) 
test_eq(cat(x), torch.cat([l(x) for l in layers], dim=1))
```

## Ready-to-go models


<h3 id="SimpleCNN" class="doc_header"><code>class</code> <code>SimpleCNN</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L522" class="source_link" style="float:right">[source]</a></h3>

> <code>SimpleCNN</code>(**`filters`**, **`kernel_szs`**=*`None`*, **`strides`**=*`None`*, **`bn`**=*`True`*) :: `Sequential`

Create a simple CNN with `filters`.


The model is a succession of convolutional layers from `(filters[0],filters[1])` to `(filters[n-2],filters[n-1])` (if `n` is the length of the `filters` list) followed by a [`PoolFlatten`](/layers.html#PoolFlatten). `kernel_szs` and `strides` defaults to a list of 3s and a list of 2s. If `bn=True` the convolutional layers are successions of conv-relu-batchnorm, otherwise conv-relu.

```python
tst = SimpleCNN([8,16,32])
mods = list(tst.children())
test_eq(len(mods), 3)
test_eq([[m[0].in_channels, m[0].out_channels] for m in mods[:2]], [[8,16], [16,32]])
```

Test kernel sizes

```python
tst = SimpleCNN([8,16,32], kernel_szs=[1,3])
mods = list(tst.children())
test_eq([m[0].kernel_size for m in mods[:2]], [(1,1), (3,3)])
```

Test strides

```python
tst = SimpleCNN([8,16,32], strides=[1,2])
mods = list(tst.children())
test_eq([m[0].stride for m in mods[:2]], [(1,1),(2,2)])
```


<h3 id="ProdLayer" class="doc_header"><code>class</code> <code>ProdLayer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L534" class="source_link" style="float:right">[source]</a></h3>

> <code>ProdLayer</code>() :: [`Module`](/torch_core.html#Module)

Merge a shortcut with the result of the module by multiplying them.



<h4 id="SEModule" class="doc_header"><code>SEModule</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L542" class="source_link" style="float:right">[source]</a></h4>

> <code>SEModule</code>(**`ch`**, **`reduction`**, **`act_cls`**=*`ReLU`*)





<h3 id="ResBlock" class="doc_header"><code>class</code> <code>ResBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L550" class="source_link" style="float:right">[source]</a></h3>

> <code>ResBlock</code>(**`expansion`**, **`ni`**, **`nf`**, **`stride`**=*`1`*, **`groups`**=*`1`*, **`reduction`**=*`None`*, **`nh1`**=*`None`*, **`nh2`**=*`None`*, **`dw`**=*`False`*, **`g2`**=*`1`*, **`sa`**=*`False`*, **`sym`**=*`False`*, **`norm_type`**=*`<NormType.Batch: 1>`*, **`act_cls`**=*`ReLU`*, **`ndim`**=*`2`*, **`ks`**=*`3`*, **`pool`**=*`AvgPool`*, **`pool_first`**=*`True`*, **`padding`**=*`None`*, **`bias`**=*`None`*, **`bn_1st`**=*`True`*, **`transpose`**=*`False`*, **`init`**=*`'auto'`*, **`xtra`**=*`None`*, **`bias_std`**=*`0.01`*, **`dilation`**:`Union`\[`int`, `Tuple`\[`int`, `int`\]\]=*`1`*, **`padding_mode`**:`str`=*`'zeros'`*) :: [`Module`](/torch_core.html#Module)

Resnet block from `ni` to `nh` with `stride`


This is a resnet block (normal or bottleneck depending on `expansion`, 1 for the normal block and 4 for the traditional bottleneck) that implements the tweaks from [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187). In particular, the last batchnorm layer (if that is the selected `norm_type`) is initialized with a weight (or gamma) of zero to facilitate the flow from the beginning to the end of the network. It also implements optional [Squeeze and Excitation](https://arxiv.org/abs/1709.01507) and grouped convs for [ResNeXT](https://arxiv.org/abs/1611.05431) and similar models (use `dw=True` for depthwise convs).

The `kwargs` are passed to [`ConvLayer`](/layers.html#ConvLayer) along with `norm_type`.


<h4 id="SEBlock" class="doc_header"><code>SEBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L581" class="source_link" style="float:right">[source]</a></h4>

> <code>SEBlock</code>(**`expansion`**, **`ni`**, **`nf`**, **`groups`**=*`1`*, **`reduction`**=*`16`*, **`stride`**=*`1`*, **\*\*`kwargs`**)





<h4 id="SEResNeXtBlock" class="doc_header"><code>SEResNeXtBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L585" class="source_link" style="float:right">[source]</a></h4>

> <code>SEResNeXtBlock</code>(**`expansion`**, **`ni`**, **`nf`**, **`groups`**=*`32`*, **`reduction`**=*`16`*, **`stride`**=*`1`*, **`base_width`**=*`4`*, **\*\*`kwargs`**)





<h4 id="SeparableBlock" class="doc_header"><code>SeparableBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L590" class="source_link" style="float:right">[source]</a></h4>

> <code>SeparableBlock</code>(**`expansion`**, **`ni`**, **`nf`**, **`reduction`**=*`16`*, **`stride`**=*`1`*, **`base_width`**=*`4`*, **\*\*`kwargs`**)




## Swish and Mish


<h4 id="swish" class="doc_header"><code>swish</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L616" class="source_link" style="float:right">[source]</a></h4>

> <code>swish</code>(**`x`**, **`inplace`**=*`False`*)





<h3 id="Swish" class="doc_header"><code>class</code> <code>Swish</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L619" class="source_link" style="float:right">[source]</a></h3>

> <code>Swish</code>() :: [`Module`](/torch_core.html#Module)

Same as `nn.Module`, but no need for subclasses to call `super().__init__`



<h3 id="MishJitAutoFn" class="doc_header"><code>class</code> <code>MishJitAutoFn</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L632" class="source_link" style="float:right">[source]</a></h3>

> <code>MishJitAutoFn</code>() :: `Function`

Records operation history and defines formulas for differentiating ops.

See the Note on extending the autograd engine for more details on how to use
this class: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd

Every operation performed on :class:`Tensor` s creates a new function
object, that performs the computation, and records that it happened.
The history is retained in the form of a DAG of functions, with edges
denoting data dependencies (``input <- output``). Then, when backward is
called, the graph is processed in the topological ordering, by calling
:func:`backward` methods of each :class:`Function` object, and passing
returned gradients on to next :class:`Function` s.

Normally, the only way users interact with functions is by creating
subclasses and defining new operations. This is a recommended way of
extending torch.autograd.

Examples::

    >>> class Exp(Function):
    >>>
    >>>     @staticmethod
    >>>     def forward(ctx, i):
    >>>         result = i.exp()
    >>>         ctx.save_for_backward(result)
    >>>         return result
    >>>
    >>>     @staticmethod
    >>>     def backward(ctx, grad_output):
    >>>         result, = ctx.saved_tensors
    >>>         return grad_output * result
    >>>
    >>> #Use it by calling the apply method:
    >>> output = Exp.apply(input)



<h4 id="mish" class="doc_header"><code>mish</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L644" class="source_link" style="float:right">[source]</a></h4>

> <code>mish</code>(**`x`**)





<h3 id="Mish" class="doc_header"><code>class</code> <code>Mish</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L647" class="source_link" style="float:right">[source]</a></h3>

> <code>Mish</code>() :: [`Module`](/torch_core.html#Module)

Same as `nn.Module`, but no need for subclasses to call `super().__init__`


## Helper functions for submodules

It's easy to get the list of all parameters of a given model. For when you want all submodules (like linear/conv layers) without forgetting lone parameters, the following class wraps those in fake modules.


<h3 id="ParameterModule" class="doc_header"><code>class</code> <code>ParameterModule</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L654" class="source_link" style="float:right">[source]</a></h3>

> <code>ParameterModule</code>(**`p`**) :: [`Module`](/torch_core.html#Module)

Register a lone parameter `p` in a module.



<h4 id="children_and_parameters" class="doc_header"><code>children_and_parameters</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L660" class="source_link" style="float:right">[source]</a></h4>

> <code>children_and_parameters</code>(**`m`**)

Return the children of `m` and its direct parameters not registered in modules.


```python
class TstModule(Module):
    def __init__(self): self.a,self.lin = nn.Parameter(torch.randn(1)),nn.Linear(5,10)

tst = TstModule()
children = children_and_parameters(tst)
test_eq(len(children), 2)
test_eq(children[0], tst.lin)
assert isinstance(children[1], ParameterModule)
test_eq(children[1].val, tst.a)
```

```python
class A(Module): pass
assert not A().has_children
assert TstModule().has_children
```


<h4 id="flatten_model" class="doc_header"><code>flatten_model</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L677" class="source_link" style="float:right">[source]</a></h4>

> <code>flatten_model</code>(**`m`**)

Return the list of all submodules and parameters of `m`


```python
tst = nn.Sequential(TstModule(), TstModule())
children = flatten_model(tst)
test_eq(len(children), 4)
assert isinstance(children[1], ParameterModule)
assert isinstance(children[3], ParameterModule)
```


<h3 id="NoneReduce" class="doc_header"><code>class</code> <code>NoneReduce</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L682" class="source_link" style="float:right">[source]</a></h3>

> <code>NoneReduce</code>(**`loss_func`**)

A context manager to evaluate `loss_func` with none reduce.


```python
x,y = torch.randn(5),torch.randn(5)
loss_fn = nn.MSELoss()
with NoneReduce(loss_fn) as loss_func:
    loss = loss_func(x,y)
test_eq(loss.shape, [5])
test_eq(loss_fn.reduction, 'mean')

loss_fn = F.mse_loss
with NoneReduce(loss_fn) as loss_func:
    loss = loss_func(x,y)
test_eq(loss.shape, [5])
test_eq(loss_fn, F.mse_loss)
```


<h4 id="in_channels" class="doc_header"><code>in_channels</code><a href="https://github.com/fastai/fastai/tree/master/fastai/layers.py#L697" class="source_link" style="float:right">[source]</a></h4>

> <code>in_channels</code>(**`m`**)

Return the shape of the first weight layer in `m`.


```python
test_eq(in_channels(nn.Sequential(nn.Conv2d(5,4,3), nn.Conv2d(4,3,3))), 5)
test_eq(in_channels(nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(4,3,3))), 4)
test_eq(in_channels(nn.Sequential(BatchNorm(4), nn.Conv2d(4,3,3))), 4)
test_eq(in_channels(nn.Sequential(InstanceNorm(4), nn.Conv2d(4,3,3))), 4)
test_eq(in_channels(nn.Sequential(InstanceNorm(4, affine=False), nn.Conv2d(4,3,3))), 4)
test_fail(lambda : in_channels(nn.Sequential(nn.AvgPool2d(4))))
```
