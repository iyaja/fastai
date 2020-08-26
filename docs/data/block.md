# Data block
> High level API to quickly get your data in a `DataLoaders`



<h2 id="TransformBlock" class="doc_header"><code>class</code> <code>TransformBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/block.py#L13" class="source_link" style="float:right">[source]</a></h2>

> <code>TransformBlock</code>(**`type_tfms`**=*`None`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`dl_type`**=*`None`*, **`dls_kwargs`**=*`None`*)

A basic wrapper that links defaults transforms for the data block API



<h4 id="CategoryBlock" class="doc_header"><code>CategoryBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/block.py#L22" class="source_link" style="float:right">[source]</a></h4>

> <code>CategoryBlock</code>(**`vocab`**=*`None`*, **`sort`**=*`True`*, **`add_na`**=*`False`*)

[`TransformBlock`](/data.block.html#TransformBlock) for single-label categorical targets



<h4 id="MultiCategoryBlock" class="doc_header"><code>MultiCategoryBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/block.py#L27" class="source_link" style="float:right">[source]</a></h4>

> <code>MultiCategoryBlock</code>(**`encoded`**=*`False`*, **`vocab`**=*`None`*, **`add_na`**=*`False`*)

[`TransformBlock`](/data.block.html#TransformBlock) for multi-label categorical targets



<h4 id="RegressionBlock" class="doc_header"><code>RegressionBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/block.py#L33" class="source_link" style="float:right">[source]</a></h4>

> <code>RegressionBlock</code>(**`n_out`**=*`None`*)

[`TransformBlock`](/data.block.html#TransformBlock) for float targets


## General API

```python
from fastai.vision.core import *
from fastai.vision.data import *
```


<h2 id="DataBlock" class="doc_header"><code>class</code> <code>DataBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/block.py#L58" class="source_link" style="float:right">[source]</a></h2>

> <code>DataBlock</code>(**`blocks`**=*`None`*, **`dl_type`**=*`None`*, **`getters`**=*`None`*, **`n_inp`**=*`None`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`get_items`**=*`None`*, **`splitter`**=*`None`*, **`get_y`**=*`None`*, **`get_x`**=*`None`*)

Generic container to quickly build [`Datasets`](/data.core.html#Datasets) and [`DataLoaders`](/data.core.html#DataLoaders)


To build a [`DataBlock`](/data.block.html#DataBlock) you need to give the library four things: the types of your input/labels, and at least two functions: `get_items` and `splitter`. You may also need to include `get_x` and `get_y` or a more generic list of `getters` that are applied to the results of `get_items`.

Once those are provided, you automatically get a [`Datasets`](/data.core.html#Datasets) or a [`DataLoaders`](/data.core.html#DataLoaders):


<h4 id="DataBlock.datasets" class="doc_header"><code>DataBlock.datasets</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/block.py#L105" class="source_link" style="float:right">[source]</a></h4>

> <code>DataBlock.datasets</code>(**`source`**, **`verbose`**=*`False`*)

Create a [`Datasets`](/data.core.html#Datasets) object from `source`



<h4 id="DataBlock.dataloaders" class="doc_header"><code>DataBlock.dataloaders</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/block.py#L112" class="source_link" style="float:right">[source]</a></h4>

> <code>DataBlock.dataloaders</code>(**`source`**, **`path`**=*`'.'`*, **`verbose`**=*`False`*, **`bs`**=*`64`*, **`shuffle`**=*`False`*, **`num_workers`**=*`None`*, **`do_setup`**=*`True`*, **`pin_memory`**=*`False`*, **`timeout`**=*`0`*, **`batch_size`**=*`None`*, **`drop_last`**=*`False`*, **`indexed`**=*`None`*, **`n`**=*`None`*, **`device`**=*`None`*, **`wif`**=*`None`*, **`before_iter`**=*`None`*, **`after_item`**=*`None`*, **`before_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_iter`**=*`None`*, **`create_batches`**=*`None`*, **`create_item`**=*`None`*, **`create_batch`**=*`None`*, **`retain`**=*`None`*, **`get_idxs`**=*`None`*, **`sample`**=*`None`*, **`shuffle_fn`**=*`None`*, **`do_batch`**=*`None`*)

Create a [`DataLoaders`](/data.core.html#DataLoaders) object from `source`


You can create a [`DataBlock`](/data.block.html#DataBlock) by passing functions:

```python
mnist = DataBlock(blocks = (ImageBlock(cls=PILImageBW),CategoryBlock),
                  get_items = get_image_files,
                  splitter = GrandparentSplitter(),
                  get_y = parent_label)
```

Each type comes with default transforms that will be applied
- at the base level to create items in a tuple (usually input,target) from the base elements (like filenames)
- at the item level of the datasets
- at the batch level

They are called respectively type transforms, item transforms, batch transforms. In the case of MNIST, the type transforms are the method to create a [`PILImageBW`](/vision.core.html#PILImageBW) (for the input) and the [`Categorize`](/data.transforms.html#Categorize) transform (for the target), the item transform is [`ToTensor`](/data.transforms.html#ToTensor) and the batch transforms are `Cuda` and [`IntToFloatTensor`](/data.transforms.html#IntToFloatTensor). You can add any other transforms by passing them in [`DataBlock.datasets`](/data.block.html#DataBlock.datasets) or [`DataBlock.dataloaders`](/data.block.html#DataBlock.dataloaders).

```python
test_eq(mnist.type_tfms[0], [PILImageBW.create])
test_eq(mnist.type_tfms[1].map(type), [Categorize])
test_eq(mnist.default_item_tfms.map(type), [ToTensor])
test_eq(mnist.default_batch_tfms.map(type), [IntToFloatTensor])
```

```python
dsets = mnist.datasets(untar_data(URLs.MNIST_TINY))
test_eq(dsets.vocab, ['3', '7'])
x,y = dsets.train[0]
test_eq(x.size,(28,28))
show_at(dsets.train, 0, cmap='Greys', figsize=(2,2));
```


![png](output_24_0.png)


```python
test_fail(lambda: DataBlock(wrong_kwarg=42, wrong_kwarg2='foo'))
```

We can pass any number of blocks to [`DataBlock`](/data.block.html#DataBlock), we can then define what are the input and target blocks by changing `n_inp`. For example, defining `n_inp=2` will consider the first two blocks passed as inputs and the others as targets. 

```python
mnist = DataBlock((ImageBlock, ImageBlock, CategoryBlock), get_items=get_image_files, splitter=GrandparentSplitter(),
                   get_y=parent_label)
dsets = mnist.datasets(untar_data(URLs.MNIST_TINY))
test_eq(mnist.n_inp, 2)
test_eq(len(dsets.train[0]), 3)
```

```python
test_fail(lambda: DataBlock((ImageBlock, ImageBlock, CategoryBlock), get_items=get_image_files, splitter=GrandparentSplitter(),
                  get_y=[parent_label, noop],
                  n_inp=2), msg='get_y contains 2 functions, but must contain 1 (one for each output)')
```

```python
mnist = DataBlock((ImageBlock, ImageBlock, CategoryBlock), get_items=get_image_files, splitter=GrandparentSplitter(),
                  n_inp=1,
                  get_y=[noop, Pipeline([noop, parent_label])])
dsets = mnist.datasets(untar_data(URLs.MNIST_TINY))
test_eq(len(dsets.train[0]), 3)
```

## Debugging


<h4 id="DataBlock.summary" class="doc_header"><code>DataBlock.summary</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/block.py#L156" class="source_link" style="float:right">[source]</a></h4>

> <code>DataBlock.summary</code>(**`source`**, **`bs`**=*`4`*, **`show_batch`**=*`False`*, **\*\*`kwargs`**)

Steps through the transform pipeline for one batch, and optionally calls `show_batch(**kwargs)` on the transient `Dataloaders`.


Besides stepping through the transformation, `summary()`  provides a shortcut `dls.show_batch(...)`, to see the data.  E.g.

```
pets.summary(path/"images", bs=8, show_batch=True, unique=True,...)
```

is a shortcut to:
```
pets.summary(path/"images", bs=8)
dls = pets.dataloaders(path/"images", bs=8)
dls.show_batch(unique=True,...)  # See different tfms effect on the same image.
```
