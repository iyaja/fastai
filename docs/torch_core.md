# Torch Core
> Basic pytorch functions used in the fastai library


```python
from PIL import Image
```

## Arrays and show


<h4 id="subplots" class="doc_header"><code>subplots</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L31" class="source_link" style="float:right">[source]</a></h4>

> <code>subplots</code>(**`nrows`**=*`1`*, **`ncols`**=*`1`*, **`figsize`**=*`None`*, **`imsize`**=*`3`*, **`add_vert`**=*`0`*, **`sharex`**=*`False`*, **`sharey`**=*`False`*, **`squeeze`**=*`True`*, **`subplot_kw`**=*`None`*, **`gridspec_kw`**=*`None`*, **\*\*`kwargs`**)





<h4 id="show_image" class="doc_header"><code>show_image</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L44" class="source_link" style="float:right">[source]</a></h4>

> <code>show_image</code>(**`im`**, **`ax`**=*`None`*, **`figsize`**=*`None`*, **`title`**=*`None`*, **`ctx`**=*`None`*, **`cmap`**=*`None`*, **`norm`**=*`None`*, **`aspect`**=*`None`*, **`interpolation`**=*`None`*, **`alpha`**=*`None`*, **`vmin`**=*`None`*, **`vmax`**=*`None`*, **`origin`**=*`None`*, **`extent`**=*`None`*, **`filternorm`**=*`True`*, **`filterrad`**=*`4.0`*, **`resample`**=*`None`*, **`url`**=*`None`*, **`data`**=*`None`*, **\*\*`kwargs`**)

Show a PIL or PyTorch image on `ax`.


[`show_image`](/torch_core.html#show_image) can show PIL images...

```python
im = Image.open(TEST_IMAGE_BW)
ax = show_image(im, cmap="Greys")
```


![png](output_12_0.png)


...and color images with standard `CHW` dim order...

```python
im2 = np.array(Image.open(TEST_IMAGE))
ax = show_image(im2, figsize=(2,2))
```


![png](output_14_0.png)


...and color images with `HWC` dim order...

```python
im3 = torch.as_tensor(im2).permute(2,0,1)
ax = show_image(im3, figsize=(2,2))
```


![png](output_16_0.png)



<h4 id="show_titled_image" class="doc_header"><code>show_titled_image</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L64" class="source_link" style="float:right">[source]</a></h4>

> <code>show_titled_image</code>(**`o`**, **`ax`**=*`None`*, **`figsize`**=*`None`*, **`title`**=*`None`*, **`ctx`**=*`None`*, **`cmap`**=*`None`*, **`norm`**=*`None`*, **`aspect`**=*`None`*, **`interpolation`**=*`None`*, **`alpha`**=*`None`*, **`vmin`**=*`None`*, **`vmax`**=*`None`*, **`origin`**=*`None`*, **`extent`**=*`None`*, **`filternorm`**=*`True`*, **`filterrad`**=*`4.0`*, **`resample`**=*`None`*, **`url`**=*`None`*, **`data`**=*`None`*, **\*\*`kwargs`**)

Call [`show_image`](/torch_core.html#show_image) destructuring `o` to `(img,title)`


```python
show_titled_image((im3,'A puppy'), figsize=(2,2))
```


![png](output_19_0.png)



<h4 id="show_images" class="doc_header"><code>show_images</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L70" class="source_link" style="float:right">[source]</a></h4>

> <code>show_images</code>(**`ims`**, **`nrows`**=*`1`*, **`ncols`**=*`None`*, **`titles`**=*`None`*, **`figsize`**=*`None`*, **`imsize`**=*`3`*, **`add_vert`**=*`0`*, **`sharex`**=*`False`*, **`sharey`**=*`False`*, **`squeeze`**=*`True`*, **`subplot_kw`**=*`None`*, **`gridspec_kw`**=*`None`*)

Show all images `ims` as subplots with `rows` using `titles`


```python
show_images((im,im3), titles=('number','puppy'), imsize=2)
```


![png](output_22_0.png)


[`ArrayImage`](/torch_core.html#ArrayImage), [`ArrayImageBW`](/torch_core.html#ArrayImageBW) and [`ArrayMask`](/torch_core.html#ArrayMask) are subclasses of `ndarray` that know how to show themselves.


<h2 id="ArrayBase" class="doc_header"><code>class</code> <code>ArrayBase</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L79" class="source_link" style="float:right">[source]</a></h2>

> <code>ArrayBase</code>() :: `ndarray`

An `ndarray` that can modify casting behavior



<h2 id="ArrayImageBase" class="doc_header"><code>class</code> <code>ArrayImageBase</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L85" class="source_link" style="float:right">[source]</a></h2>

> <code>ArrayImageBase</code>() :: [`ArrayBase`](/torch_core.html#ArrayBase)

Base class for arrays representing images



<h2 id="ArrayImage" class="doc_header"><code>class</code> <code>ArrayImage</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L92" class="source_link" style="float:right">[source]</a></h2>

> <code>ArrayImage</code>() :: [`ArrayImageBase`](/torch_core.html#ArrayImageBase)

An array representing an image



<h2 id="ArrayImageBW" class="doc_header"><code>class</code> <code>ArrayImageBW</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L97" class="source_link" style="float:right">[source]</a></h2>

> <code>ArrayImageBW</code>() :: [`ArrayImage`](/torch_core.html#ArrayImage)

An array representing an image



<h2 id="ArrayMask" class="doc_header"><code>class</code> <code>ArrayMask</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L102" class="source_link" style="float:right">[source]</a></h2>

> <code>ArrayMask</code>() :: [`ArrayImageBase`](/torch_core.html#ArrayImageBase)

An array representing an image mask


```python
im = Image.open(TEST_IMAGE)
```

```python
im_t = cast(im, ArrayImage)
test_eq(type(im_t), ArrayImage)
```

```python
ax = im_t.show(figsize=(2,2))
```


![png](output_36_0.png)


```python
test_fig_exists(ax)
```

## Basics


<h4 id="Tensor.__array_eq__" class="doc_header"><code>Tensor.__array_eq__</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L107" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.__array_eq__</code>(**`b`**)





<h4 id="tensor" class="doc_header"><code>tensor</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L117" class="source_link" style="float:right">[source]</a></h4>

> <code>tensor</code>(**`x`**, **\*`rest`**, **`dtype`**=*`None`*, **`device`**=*`None`*, **`requires_grad`**=*`False`*, **`pin_memory`**=*`False`*)

Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly.


```python
test_eq(tensor(torch.tensor([1,2,3])), torch.tensor([1,2,3]))
test_eq(tensor(array([1,2,3])), torch.tensor([1,2,3]))
test_eq(tensor(1,2,3), torch.tensor([1,2,3]))
test_eq_type(tensor(1.0), torch.tensor(1.0))
```


<h4 id="set_seed" class="doc_header"><code>set_seed</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L133" class="source_link" style="float:right">[source]</a></h4>

> <code>set_seed</code>(**`s`**, **`reproducible`**=*`False`*)

Set random seed for `random`, `torch`, and `numpy` (where available)


```python
set_seed(2*33)
a1 = np.random.random()
a2 = torch.rand(())
a3 = random.random()
set_seed(2*33)
b1 = np.random.random()
b2 = torch.rand(())
b3 = random.random()
test_eq(a1,b1)
test_eq(a2,b2)
test_eq(a3,b3)
```


<h4 id="unsqueeze" class="doc_header"><code>unsqueeze</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L145" class="source_link" style="float:right">[source]</a></h4>

> <code>unsqueeze</code>(**`x`**, **`dim`**=*`-1`*, **`n`**=*`1`*)

Same as `torch.unsqueeze` but can add `n` dims


```python
t = tensor([1])
t2 = unsqueeze(t, n=2)
test_eq(t2,t[:,None,None])
```


<h4 id="unsqueeze_" class="doc_header"><code>unsqueeze_</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L151" class="source_link" style="float:right">[source]</a></h4>

> <code>unsqueeze_</code>(**`x`**, **`dim`**=*`-1`*, **`n`**=*`1`*)

Same as `torch.unsqueeze_` but can add `n` dims


```python
t = tensor([1])
unsqueeze_(t, n=2)
test_eq(t, tensor([1]).view(1,1,1))
```


<h4 id="apply" class="doc_header"><code>apply</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L161" class="source_link" style="float:right">[source]</a></h4>

> <code>apply</code>(**`func`**, **`x`**, **\*`args`**, **\*\*`kwargs`**)

Apply `func` recursively to `x`, passing on args



<h4 id="maybe_gather" class="doc_header"><code>maybe_gather</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L169" class="source_link" style="float:right">[source]</a></h4>

> <code>maybe_gather</code>(**`x`**, **`axis`**=*`0`*)

Gather copies of `x` on `axis` (if training is distributed)



<h4 id="to_detach" class="doc_header"><code>to_detach</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L178" class="source_link" style="float:right">[source]</a></h4>

> <code>to_detach</code>(**`b`**, **`cpu`**=*`True`*, **`gather`**=*`True`*)

Recursively detach lists of tensors in `b `; put them on the CPU if `cpu=True`.


`gather` only applies during distributed training and the result tensor will be the one gathered across processes if `gather=True` (as a result, the batch size will be multiplied by the number of processes).


<h4 id="to_half" class="doc_header"><code>to_half</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L188" class="source_link" style="float:right">[source]</a></h4>

> <code>to_half</code>(**`b`**)

Recursively map lists of tensors in `b ` to FP16.



<h4 id="to_float" class="doc_header"><code>to_float</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L193" class="source_link" style="float:right">[source]</a></h4>

> <code>to_float</code>(**`b`**)

Recursively map lists of int tensors in `b ` to float.



<h4 id="default_device" class="doc_header"><code>default_device</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L202" class="source_link" style="float:right">[source]</a></h4>

> <code>default_device</code>(**`use_cuda`**=*`-1`*)

Return or set default device; `use_cuda`: None - CUDA if available; True - error if not availabe; False - CPU


```python
_td = torch.device(torch.cuda.current_device())
test_eq(default_device(None), _td)
test_eq(default_device(True), _td)
test_eq(default_device(False), torch.device('cpu'))
default_device(None);
```


<h4 id="to_device" class="doc_header"><code>to_device</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L210" class="source_link" style="float:right">[source]</a></h4>

> <code>to_device</code>(**`b`**, **`device`**=*`None`*)

Recursively put `b` on `device`.


```python
t = to_device((3,(tensor(3),tensor(2))))
t1,(t2,t3) = t
```

```python
test_eq_type(t,(3,(tensor(3).cuda(),tensor(2).cuda())))
test_eq(t2.type(), "torch.cuda.LongTensor")
test_eq(t3.type(), "torch.cuda.LongTensor")
```


<h4 id="to_cpu" class="doc_header"><code>to_cpu</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L218" class="source_link" style="float:right">[source]</a></h4>

> <code>to_cpu</code>(**`b`**)

Recursively map lists of tensors in `b ` to the cpu.


```python
t3 = to_cpu(t3)
test_eq(t3.type(), "torch.LongTensor")
test_eq(t3, 2)
```


<h4 id="to_np" class="doc_header"><code>to_np</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L223" class="source_link" style="float:right">[source]</a></h4>

> <code>to_np</code>(**`x`**)

Convert a tensor to a numpy array.


```python
t3 = to_np(t3)
test_eq(type(t3), np.ndarray)
test_eq(t3, 2)
```


<h4 id="to_concat" class="doc_header"><code>to_concat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L228" class="source_link" style="float:right">[source]</a></h4>

> <code>to_concat</code>(**`xs`**, **`dim`**=*`0`*)

Concat the element in `xs` (recursively if they are tuples/lists of tensors)


```python
test_eq(to_concat([tensor([1,2]), tensor([3,4])]), tensor([1,2,3,4]))
test_eq(to_concat([tensor([[1,2]]), tensor([[3,4]])], dim=1), tensor([[1,2,3,4]]))
test_eq_type(to_concat([(tensor([1,2]), tensor([3,4])), (tensor([3,4]), tensor([5,6]))]), (tensor([1,2,3,4]), tensor([3,4,5,6])))
test_eq_type(to_concat([[tensor([1,2]), tensor([3,4])], [tensor([3,4]), tensor([5,6])]]), [tensor([1,2,3,4]), tensor([3,4,5,6])])
test_eq_type(to_concat([(tensor([1,2]),), (tensor([3,4]),)]), (tensor([1,2,3,4]),))

test_eq(to_concat([tensor([[1,2]]), tensor([[3,4], [5,6]])], dim=1), [tensor([1]),tensor([3, 5]),tensor([4, 6])])
```

```python
test_eq(type(to_concat([dict(foo=tensor([1,2]), bar=tensor(3,4))])), dict)
```

## Tensor subtypes


<h4 id="Tensor.set_meta" class="doc_header"><code>Tensor.set_meta</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L240" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.set_meta</code>(**`x`**, **`copy_meta`**=*`False`*)

Set all metadata in `__dict__`



<h4 id="Tensor.get_meta" class="doc_header"><code>Tensor.get_meta</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L251" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.get_meta</code>(**`n`**, **`d`**=*`None`*)

Set `n` from `self._meta` if it exists and returns default `d` otherwise



<h4 id="Tensor.as_subclass" class="doc_header"><code>Tensor.as_subclass</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L261" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.as_subclass</code>(**`typ`**)

Cast to `typ` and include `__dict__` and meta


[`Tensor.set_meta`](/torch_core.html#Tensor.set_meta) and [`Tensor.as_subclass`](/torch_core.html#Tensor.as_subclass) work together to maintain `_meta` after casting.

```python
class _T(Tensor): pass
t = tensor(1.).requires_grad_()
t._meta = {'img_size': 1}
t2 = t.as_subclass(_T)
test_eq(t._meta, t2._meta)
test_eq(t2.get_meta('img_size'), 1)
assert(t2.requires_grad_)
```


<h2 id="TensorBase" class="doc_header"><code>class</code> <code>TensorBase</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L267" class="source_link" style="float:right">[source]</a></h2>

> <code>TensorBase</code>(**`x`**, **\*\*`kwargs`**) :: `Tensor`





<h2 id="TensorCategory" class="doc_header"><code>class</code> <code>TensorCategory</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L315" class="source_link" style="float:right">[source]</a></h2>

> <code>TensorCategory</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorBase`](/torch_core.html#TensorBase)





<h2 id="TensorMultiCategory" class="doc_header"><code>class</code> <code>TensorMultiCategory</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L318" class="source_link" style="float:right">[source]</a></h2>

> <code>TensorMultiCategory</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorCategory`](/torch_core.html#TensorCategory)




```python
class _T(TensorBase): pass
```

```python
t = _T(range(5))
test_eq(t[0], 0)
test_eq_type(t.gi(0), _T(0))
test_eq_type(t.gi(slice(2)), _T([0,1]))
test_eq_type(t+1, _T(range(1,6)))
test_eq(repr(t), '_T([0, 1, 2, 3, 4])')

test_eq(type(pickle.loads(pickle.dumps(t))), _T)
```

```python
t = tensor([1,2,3])
m = TensorBase([False,True,True])
test_eq(t[m], tensor([2,3]))
t = tensor([[1,2,3],[1,2,3]])
m = cast(tensor([[False,True,True],
                 [False,True,True]]), TensorBase)
test_eq(t[m], tensor([2,3,2,3]))
```

```python
t = tensor([[1,2,3],[1,2,3]])
t._meta = {'img_size': 1}
t2 = cast(t, TensorBase)
test_eq(t2._meta, t._meta)
x = retain_type(tensor([4,5,6]), t2)
test_eq(x._meta, t._meta)
t3 = TensorBase([[1,2,3],[1,2,3]], img_size=1)
test_eq(t3._meta, t._meta)
t4 = t2+1
t4._meta['img_size'] = 2
test_eq(t2._meta, {'img_size': 1})
test_eq(t4._meta, {'img_size': 2})
```


<h2 id="TensorImageBase" class="doc_header"><code>class</code> <code>TensorImageBase</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L321" class="source_link" style="float:right">[source]</a></h2>

> <code>TensorImageBase</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorBase`](/torch_core.html#TensorBase)





<h2 id="TensorImage" class="doc_header"><code>class</code> <code>TensorImage</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L327" class="source_link" style="float:right">[source]</a></h2>

> <code>TensorImage</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorImageBase`](/torch_core.html#TensorImageBase)





<h2 id="TensorImageBW" class="doc_header"><code>class</code> <code>TensorImageBW</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L330" class="source_link" style="float:right">[source]</a></h2>

> <code>TensorImageBW</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorImage`](/torch_core.html#TensorImage)





<h2 id="TensorMask" class="doc_header"><code>class</code> <code>TensorMask</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L333" class="source_link" style="float:right">[source]</a></h2>

> <code>TensorMask</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorImageBase`](/torch_core.html#TensorImageBase)




```python
im = Image.open(TEST_IMAGE)
im_t = cast(array(im), TensorImage)
test_eq(type(im_t), TensorImage)
```

```python
im_t2 = cast(tensor(1), TensorMask)
test_eq(type(im_t2), TensorMask)
test_eq(im_t2, tensor(1))
```

```python
ax = im_t.show(figsize=(2,2))
```


![png](output_115_0.png)


```python
test_fig_exists(ax)
```

```python
test_eq_type(to_concat([TensorImage([1,2]), TensorImage([3,4])]), TensorImage([1,2,3,4]))
```


<h2 id="TitledTensorScalar" class="doc_header"><code>class</code> <code>TitledTensorScalar</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L342" class="source_link" style="float:right">[source]</a></h2>

> <code>TitledTensorScalar</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorBase`](/torch_core.html#TensorBase)

A tensor containing a scalar that has a `show` method



<h4 id="L.tensored" class="doc_header"><code>L.tensored</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L347" class="source_link" style="float:right">[source]</a></h4>

> <code>L.tensored</code>()

`mapped(tensor)`


There are shortcuts for `torch.stack` and `torch.cat` if your [`L`](https://fastcore.fast.ai/foundation#L) contains tensors or something convertible. You can manually convert with `tensored`.

```python
t = L(([1,2],[3,4]))
test_eq(t.tensored(), [tensor(1,2),tensor(3,4)])
```


<h4 id="L.stack" class="doc_header"><code>L.stack</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L351" class="source_link" style="float:right">[source]</a></h4>

> <code>L.stack</code>(**`dim`**=*`0`*)

Same as `torch.stack`


```python
test_eq(t.stack(), tensor([[1,2],[3,4]]))
```


<h4 id="L.cat" class="doc_header"><code>L.cat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L355" class="source_link" style="float:right">[source]</a></h4>

> <code>L.cat</code>(**`dim`**=*`0`*)

Same as `torch.cat`


```python
test_eq(t.cat(), tensor([1,2,3,4]))
```

## Chunks


<h4 id="concat" class="doc_header"><code>concat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L361" class="source_link" style="float:right">[source]</a></h4>

> <code>concat</code>(**\*`ls`**)

Concatenate tensors, arrays, lists, or tuples


```python
a,b,c = [1],[1,2],[1,1,2]
test_eq(concat(a,b), c)
test_eq_type(concat(tuple (a),tuple (b)), tuple (c))
test_eq_type(concat(array (a),array (b)), array (c))
test_eq_type(concat(tensor(a),tensor(b)), tensor(c))
test_eq_type(concat(TensorBase(a),TensorBase(b)), TensorBase(c))
test_eq_type(concat([1,1],1), [1,1,1])
test_eq_type(concat(1,1,1), L(1,1,1))
test_eq_type(concat(L(1,2),1), L(1,2,1))
```


<h2 id="Chunks" class="doc_header"><code>class</code> <code>Chunks</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L374" class="source_link" style="float:right">[source]</a></h2>

> <code>Chunks</code>(**`chunks`**, **`lens`**=*`None`*)

Slice and int indexing into a list of lists


```python
docs = L(list(string.ascii_lowercase[a:b]) for a,b in ((0,3),(3,7),(7,8),(8,16),(16,24),(24,26)))

b = Chunks(docs)
test_eq([b[ o] for o in range(0,5)], ['a','b','c','d','e'])
test_eq([b[-o] for o in range(1,6)], ['z','y','x','w','v'])
test_eq(b[6:13], 'g,h,i,j,k,l,m'.split(','))
test_eq(b[20:77], 'u,v,w,x,y,z'.split(','))
test_eq(b[:5], 'a,b,c,d,e'.split(','))
test_eq(b[:2], 'a,b'.split(','))
```

```python
t = torch.arange(26)
docs = L(t[a:b] for a,b in ((0,3),(3,7),(7,8),(8,16),(16,24),(24,26)))
b = Chunks(docs)
test_eq([b[ o] for o in range(0,5)], range(0,5))
test_eq([b[-o] for o in range(1,6)], [25,24,23,22,21])
test_eq(b[6:13], torch.arange(6,13))
test_eq(b[20:77], torch.arange(20,26))
test_eq(b[:5], torch.arange(5))
test_eq(b[:2], torch.arange(2))
```

```python
docs = L(TensorBase(t[a:b]) for a,b in ((0,3),(3,7),(7,8),(8,16),(16,24),(24,26)))
b = Chunks(docs)
test_eq_type(b[:2], TensorBase(range(2)))
test_eq_type(b[:5], TensorBase(range(5)))
test_eq_type(b[9:13], TensorBase(range(9,13)))
```

## Simple types


<h4 id="show_title" class="doc_header"><code>show_title</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L402" class="source_link" style="float:right">[source]</a></h4>

> <code>show_title</code>(**`o`**, **`ax`**=*`None`*, **`ctx`**=*`None`*, **`label`**=*`None`*, **`color`**=*`'black'`*, **\*\*`kwargs`**)

Set title of `ax` to `o`, or print `o` if `ax` is `None`


```python
test_stdout(lambda: show_title("title"), "title")
# ensure that col names are unique when showing to a pandas series
assert show_title("title", ctx=pd.Series(dict(a=1)), label='a').equals(pd.Series(dict(a=1,a_='title')))
```


<h2 id="ShowTitle" class="doc_header"><code>class</code> <code>ShowTitle</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L416" class="source_link" style="float:right">[source]</a></h2>

> <code>ShowTitle</code>()

Base class that adds a simple `show`



<h3 id="TitledInt" class="doc_header"><code>class</code> <code>TitledInt</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L423" class="source_link" style="float:right">[source]</a></h3>

> <code>TitledInt</code>() :: [`Int`](https://fastcore.fast.ai/utils#Int)

An `int` with `show`



<h3 id="TitledStr" class="doc_header"><code>class</code> <code>TitledStr</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L435" class="source_link" style="float:right">[source]</a></h3>

> <code>TitledStr</code>() :: [`Str`](https://fastcore.fast.ai/utils#Str)

An `str` with `show`



<h3 id="TitledFloat" class="doc_header"><code>class</code> <code>TitledFloat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L429" class="source_link" style="float:right">[source]</a></h3>

> <code>TitledFloat</code>(**`x`**=*`0`*) :: [`Float`](https://fastcore.fast.ai/utils#Float)

A `float` with `show`


```python
test_stdout(lambda: TitledStr('s').show(), 's')
test_stdout(lambda: TitledInt(1).show(), '1')
```


<h3 id="TitledTuple" class="doc_header"><code>class</code> <code>TitledTuple</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L441" class="source_link" style="float:right">[source]</a></h3>

> <code>TitledTuple</code>(**`x`**=*`None`*, **\*`rest`**) :: [`fastuple`](https://fastcore.fast.ai/utils#fastuple)

A [`fastuple`](https://fastcore.fast.ai/utils#fastuple) with `show`



<h4 id="TitledStr.truncate" class="doc_header"><code>TitledStr.truncate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L451" class="source_link" style="float:right">[source]</a></h4>

> <code>TitledStr.truncate</code>(**`n`**)

Truncate self to `n`


## Other functions


<h4 id="DataFrame.__init__" class="doc_header"><code>DataFrame.__init__</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L461" class="source_link" style="float:right">[source]</a></h4>

> <code>DataFrame.__init__</code>(**`data`**=*`None`*, **`index`**=*`None`*, **`columns`**=*`None`*, **`dtype`**=*`None`*, **`copy`**=*`False`*)





<h4 id="get_empty_df" class="doc_header"><code>get_empty_df</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L467" class="source_link" style="float:right">[source]</a></h4>

> <code>get_empty_df</code>(**`n`**)

Return `n` empty rows of a dataframe



<h4 id="display_df" class="doc_header"><code>display_df</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L473" class="source_link" style="float:right">[source]</a></h4>

> <code>display_df</code>(**`df`**)

Display `df` in a notebook or defaults to print



<h4 id="get_first" class="doc_header"><code>get_first</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L480" class="source_link" style="float:right">[source]</a></h4>

> <code>get_first</code>(**`c`**)

Get the first element of c, even if c is a dataframe



<h4 id="one_param" class="doc_header"><code>one_param</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L485" class="source_link" style="float:right">[source]</a></h4>

> <code>one_param</code>(**`m`**)

First parameter in `m`



<h4 id="item_find" class="doc_header"><code>item_find</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L490" class="source_link" style="float:right">[source]</a></h4>

> <code>item_find</code>(**`x`**, **`idx`**=*`0`*)

Recursively takes the `idx`-th element of `x`



<h4 id="find_device" class="doc_header"><code>find_device</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L499" class="source_link" style="float:right">[source]</a></h4>

> <code>find_device</code>(**`b`**)

Recursively search the device of `b`.


```python
t2 = to_device(tensor(0))
dev = default_device()
test_eq(find_device(t2), dev)
test_eq(find_device([t2,t2]), dev)
test_eq(find_device({'a':t2,'b':t2}), dev)
test_eq(find_device({'a':[[t2],[t2]],'b':t2}), dev)
```


<h4 id="find_bs" class="doc_header"><code>find_bs</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L504" class="source_link" style="float:right">[source]</a></h4>

> <code>find_bs</code>(**`b`**)

Recursively search the batch size of `b`.


```python
x = torch.randn(4,5)
test_eq(find_bs(x), 4)
test_eq(find_bs([x, x]), 4)
test_eq(find_bs({'a':x,'b':x}), 4)
test_eq(find_bs({'a':[[x],[x]],'b':x}), 4)
```


<h4 id="np_func" class="doc_header"><code>np_func</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L509" class="source_link" style="float:right">[source]</a></h4>

> <code>np_func</code>(**`f`**)

Convert a function taking and returning numpy arrays to one taking and returning tensors


This decorator is particularly useful for using numpy functions as fastai metrics, for instance:

```python
from sklearn.metrics import f1_score
```

```python
@np_func
def f1(inp,targ): return f1_score(targ, inp)

a1,a2 = array([0,1,1]),array([1,0,1])
t = f1(tensor(a1),tensor(a2))
test_eq(f1_score(a1,a2), t)
assert isinstance(t,Tensor)
```


<h3 id="Module" class="doc_header"><code>class</code> <code>Module</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L518" class="source_link" style="float:right">[source]</a></h3>

> <code>Module</code>() :: [`Module`](/torch_core.html#Module)

Same as `nn.Module`, but no need for subclasses to call `super().__init__`


```python
class _T(Module):
    def __init__(self): self.f = nn.Linear(1,1)
    def forward(self,x): return self.f(x)

t = _T()
t(tensor([1.]))
```




    tensor([-1.0893], grad_fn=<AddBackward0>)




<h4 id="get_model" class="doc_header"><code>get_model</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L526" class="source_link" style="float:right">[source]</a></h4>

> <code>get_model</code>(**`model`**)

Return the model maybe wrapped inside `model`.



<h4 id="one_hot" class="doc_header"><code>one_hot</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L531" class="source_link" style="float:right">[source]</a></h4>

> <code>one_hot</code>(**`x`**, **`c`**)

One-hot encode `x` with `c` classes.


```python
test_eq(one_hot([1,4], 5), tensor(0,1,0,0,1).byte())
test_eq(one_hot(torch.tensor([]), 5), tensor(0,0,0,0,0).byte())
test_eq(one_hot(2, 5), tensor(0,0,1,0,0).byte())
```


<h4 id="one_hot_decode" class="doc_header"><code>one_hot_decode</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L539" class="source_link" style="float:right">[source]</a></h4>

> <code>one_hot_decode</code>(**`x`**, **`vocab`**=*`None`*)




```python
test_eq(one_hot_decode(tensor(0,1,0,0,1)), [1,4])
test_eq(one_hot_decode(tensor(0,0,0,0,0)), [   ])
test_eq(one_hot_decode(tensor(0,0,1,0,0)), [2  ])
```


<h4 id="params" class="doc_header"><code>params</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L543" class="source_link" style="float:right">[source]</a></h4>

> <code>params</code>(**`m`**)

Return all parameters of `m`



<h4 id="trainable_params" class="doc_header"><code>trainable_params</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L548" class="source_link" style="float:right">[source]</a></h4>

> <code>trainable_params</code>(**`m`**)

Return all trainable parameters of `m`


```python
m = nn.Linear(4,5)
test_eq(trainable_params(m), [m.weight, m.bias])
m.weight.requires_grad_(False)
test_eq(trainable_params(m), [m.bias])
```


<h4 id="norm_bias_params" class="doc_header"><code>norm_bias_params</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L556" class="source_link" style="float:right">[source]</a></h4>

> <code>norm_bias_params</code>(**`m`**, **`with_bias`**=*`True`*)

Return all bias and BatchNorm parameters


```python
for norm_func in [nn.BatchNorm1d, partial(nn.InstanceNorm1d, affine=True)]:
    model = nn.Sequential(nn.Linear(10,20), norm_func(20), nn.Conv1d(3,4, 3))
    test_eq(norm_bias_params(model), [model[0].bias, model[1].weight, model[1].bias, model[2].bias])
    model = nn.ModuleList([nn.Linear(10,20, bias=False), nn.Sequential(norm_func(20), nn.Conv1d(3,4,3))])
    test_eq(norm_bias_params(model), [model[1][0].weight, model[1][0].bias, model[1][1].bias])
    model = nn.ModuleList([nn.Linear(10,20), nn.Sequential(norm_func(20), nn.Conv1d(3,4,3))])
    test_eq(norm_bias_params(model, with_bias=False), [model[1][0].weight, model[1][0].bias])
```


<h4 id="batch_to_samples" class="doc_header"><code>batch_to_samples</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L564" class="source_link" style="float:right">[source]</a></h4>

> <code>batch_to_samples</code>(**`b`**, **`max_n`**=*`10`*)

'Transposes' a batch to (at most `max_n`) samples


```python
t = tensor([1,2,3])
test_eq(batch_to_samples([t,t+1], max_n=2), ([1,2],[2,3]))
test_eq(batch_to_samples(tensor([1,2,3]), 10), [1, 2, 3])
test_eq(batch_to_samples([tensor([1,2,3]), tensor([4,5,6])], 10), [(1, 4), (2, 5), (3, 6)])
test_eq(batch_to_samples([tensor([1,2,3]), tensor([4,5,6])], 2), [(1, 4), (2, 5)])
test_eq(batch_to_samples([tensor([1,2,3]), [tensor([4,5,6]),tensor([7,8,9])]], 10), 
        [(1, (4, 7)), (2, (5, 8)), (3, (6, 9))])
test_eq(batch_to_samples([tensor([1,2,3]), [tensor([4,5,6]),tensor([7,8,9])]], 2), [(1, (4, 7)), (2, (5, 8))])

t = fastuple(tensor([1,2,3]),TensorBase([2,3,4]))
test_eq_type(batch_to_samples(t)[0][1], TensorBase(2))
test_eq(batch_to_samples(t).map(type), [fastuple]*3)
```


<h4 id="Tensor.interp_1d" class="doc_header"><code>Tensor.interp_1d</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L572" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.interp_1d</code>(**`x`**:`Tensor`, **`xp`**, **`fp`**)

Same as `np.interp`


```python
brks = tensor(0,1,2,4,8,64).float()
ys = tensor(range_of(brks)).float()
ys /= ys[-1].item()
pts = tensor(0.2,0.5,0.8,3,5,63)

preds = pts.interp_1d(brks, ys)
test_close(preds.numpy(), np.interp(pts.numpy(), brks.numpy(), ys.numpy()))

plt.scatter(brks,ys)
plt.scatter(pts,preds)
plt.legend(['breaks','preds']);
```


![png](output_201_0.png)



<h4 id="Tensor.pca" class="doc_header"><code>Tensor.pca</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L582" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.pca</code>(**`x`**:`Tensor`, **`k`**=*`2`*)

Compute PCA of `x` with `k` dimensions.



<h4 id="logit" class="doc_header"><code>logit</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L590" class="source_link" style="float:right">[source]</a></h4>

> <code>logit</code>(**`x`**)

Logit of `x`, clamped to avoid inf.



<h4 id="num_distrib" class="doc_header"><code>num_distrib</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L596" class="source_link" style="float:right">[source]</a></h4>

> <code>num_distrib</code>()

Return the number of processes in distributed training (if applicable).



<h4 id="rank_distrib" class="doc_header"><code>rank_distrib</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L601" class="source_link" style="float:right">[source]</a></h4>

> <code>rank_distrib</code>()

Return the distributed rank of this process (if applicable).



<h4 id="distrib_barrier" class="doc_header"><code>distrib_barrier</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L606" class="source_link" style="float:right">[source]</a></h4>

> <code>distrib_barrier</code>()

Place a synchronization barrier in distributed training so that ALL sub-processes in the pytorch process group must arrive here before proceeding.



<h4 id="Path.save_array" class="doc_header"><code>Path.save_array</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L619" class="source_link" style="float:right">[source]</a></h4>

> <code>Path.save_array</code>(**`p`**:`Path`, **`o`**, **`complib`**=*`'lz4'`*, **`lvl`**=*`3`*)

Save numpy array to a compressed `pytables` file, using compression level `lvl`


Compression lib can be any of: blosclz, lz4, lz4hc, snappy, zlib or zstd.


<h4 id="Path.load_array" class="doc_header"><code>Path.load_array</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L626" class="source_link" style="float:right">[source]</a></h4>

> <code>Path.load_array</code>(**`p`**:`Path`)

Save numpy array to a `pytables` file


```python
inspect.getdoc(load_array)
```




    'Save numpy array to a `pytables` file'



```python
str(inspect.signature(load_array))
```




    '(p: pathlib.Path)'




<h4 id="base_doc" class="doc_header"><code>base_doc</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L632" class="source_link" style="float:right">[source]</a></h4>

> <code>base_doc</code>(**`elt`**)

Print a base documentation of `elt`



<h4 id="doc" class="doc_header"><code>doc</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L639" class="source_link" style="float:right">[source]</a></h4>

> <code>doc</code>(**`elt`**)

Try to use doc form nbdev and fall back to [`base_doc`](/torch_core.html#base_doc)



<h4 id="nested_reorder" class="doc_header"><code>nested_reorder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L647" class="source_link" style="float:right">[source]</a></h4>

> <code>nested_reorder</code>(**`t`**, **`idxs`**)

Reorder all tensors in `t` using `idxs`


```python
x = tensor([0,1,2,3,4,5])
idxs = tensor([2,5,1,0,3,4])
test_eq_type(nested_reorder(([x], x), idxs), ([idxs], idxs))

y = L(0,1,2,3,4,5)
z = L(i.item() for i in idxs)
test_eq_type(nested_reorder((y, x), idxs), (z,idxs))
```

## Image helpers


<h4 id="make_cross_image" class="doc_header"><code>make_cross_image</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L655" class="source_link" style="float:right">[source]</a></h4>

> <code>make_cross_image</code>(**`bw`**=*`True`*)

Create a tensor containing a cross image, either `bw` (True) or color


```python
plt.imshow(make_cross_image(), cmap="Greys");
```


![png](output_231_0.png)


```python
plt.imshow(make_cross_image(False).permute(1,2,0));
```


![png](output_232_0.png)



<h4 id="show_image_batch" class="doc_header"><code>show_image_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L668" class="source_link" style="float:right">[source]</a></h4>

> <code>show_image_batch</code>(**`b`**, **`show`**=*`show_titled_image`*, **`items`**=*`9`*, **`cols`**=*`3`*, **`figsize`**=*`None`*, **\*\*`kwargs`**)

Display batch `b` in a grid of size `items` with `cols` width


```python
show_image_batch(([Image.open(TEST_IMAGE_BW),Image.open(TEST_IMAGE)],['bw','color']), items=2)
```


![png](output_235_0.png)


## Model init


<h4 id="requires_grad" class="doc_header"><code>requires_grad</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L677" class="source_link" style="float:right">[source]</a></h4>

> <code>requires_grad</code>(**`m`**)

Check if the first parameter of `m` requires grad or not


```python
tst = nn.Linear(4,5)
assert requires_grad(tst)
for p in tst.parameters(): p.requires_grad_(False)
assert not requires_grad(tst)
```


<h4 id="init_default" class="doc_header"><code>init_default</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L683" class="source_link" style="float:right">[source]</a></h4>

> <code>init_default</code>(**`m`**, **`func`**=*`kaiming_normal_`*)

Initialize `m` weights with `func` and set `bias` to 0.


```python
tst = nn.Linear(4,5)
tst.weight.data.uniform_(-1,1)
tst.bias.data.uniform_(-1,1)
tst = init_default(tst, func = lambda x: x.data.fill_(1.))
test_eq(tst.weight, torch.ones(5,4))
test_eq(tst.bias, torch.zeros(5))
```


<h4 id="cond_init" class="doc_header"><code>cond_init</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L691" class="source_link" style="float:right">[source]</a></h4>

> <code>cond_init</code>(**`m`**, **`func`**)

Apply [`init_default`](/layers.html#init_default) to `m` unless it's a batchnorm module


```python
tst = nn.Linear(4,5)
tst.weight.data.uniform_(-1,1)
tst.bias.data.uniform_(-1,1)
cond_init(tst, func = lambda x: x.data.fill_(1.))
test_eq(tst.weight, torch.ones(5,4))
test_eq(tst.bias, torch.zeros(5))

tst = nn.BatchNorm2d(5)
init = [tst.weight.clone(), tst.bias.clone()]
cond_init(tst, func = lambda x: x.data.fill_(1.))
test_eq(tst.weight, init[0])
test_eq(tst.bias, init[1])
```


<h4 id="apply_leaf" class="doc_header"><code>apply_leaf</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L696" class="source_link" style="float:right">[source]</a></h4>

> <code>apply_leaf</code>(**`m`**, **`f`**)

Apply `f` to children of `m`.


```python
tst = nn.Sequential(nn.Linear(4,5), nn.Sequential(nn.Linear(4,5), nn.Linear(4,5)))
apply_leaf(tst, partial(init_default, func=lambda x: x.data.fill_(1.)))
for l in [tst[0], *tst[1]]: test_eq(l.weight, torch.ones(5,4))
for l in [tst[0], *tst[1]]: test_eq(l.bias,   torch.zeros(5))
```


<h4 id="apply_init" class="doc_header"><code>apply_init</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L703" class="source_link" style="float:right">[source]</a></h4>

> <code>apply_init</code>(**`m`**, **`func`**=*`kaiming_normal_`*)

Initialize all non-batchnorm layers of `m` with `func`.


```python
tst = nn.Sequential(nn.Linear(4,5), nn.Sequential(nn.Linear(4,5), nn.BatchNorm1d(5)))
init = [tst[1][1].weight.clone(), tst[1][1].bias.clone()]
apply_init(tst, func=lambda x: x.data.fill_(1.))
for l in [tst[0], tst[1][0]]: test_eq(l.weight, torch.ones(5,4))
for l in [tst[0], tst[1][0]]: test_eq(l.bias,   torch.zeros(5))
test_eq(tst[1][1].weight, init[0])
test_eq(tst[1][1].bias,   init[1])
```

## autograd jit functions


<h4 id="script_use_ctx" class="doc_header"><code>script_use_ctx</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L708" class="source_link" style="float:right">[source]</a></h4>

> <code>script_use_ctx</code>(**`f`**)

Decorator: create jit script and pass everything in `ctx.saved_variables to `f`, after `*args`



<h4 id="script_save_ctx" class="doc_header"><code>script_save_ctx</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L715" class="source_link" style="float:right">[source]</a></h4>

> <code>script_save_ctx</code>(**`static`**, **\*`argidx`**)

Decorator: create jit script and save args with indices `argidx` using `ctx.save_for_backward`



<h4 id="script_fwd" class="doc_header"><code>script_fwd</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L730" class="source_link" style="float:right">[source]</a></h4>

> <code>script_fwd</code>(**\*`argidx`**)

Decorator: create static jit script and save args with indices `argidx` using `ctx.save_for_backward`



<h4 id="script_bwd" class="doc_header"><code>script_bwd</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L735" class="source_link" style="float:right">[source]</a></h4>

> <code>script_bwd</code>(**`f`**)

Decorator: create static jit script and pass everything in `ctx.saved_variables to `f`, after `*args`



<h4 id="grad_module" class="doc_header"><code>grad_module</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L740" class="source_link" style="float:right">[source]</a></h4>

> <code>grad_module</code>()

Decorator: convert `cls` into an autograd function

