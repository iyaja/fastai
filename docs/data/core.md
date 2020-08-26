# Data core
> Core functionality for gathering data


The classes here provide functionality for applying a list of transforms to a set of items ([`TfmdLists`](/data.core.html#TfmdLists), [`Datasets`](/data.core.html#Datasets)) or a [`DataLoader`](/data.load.html#DataLoader) (`TfmdDl`) as well as the base class used to gather the data for model training: [`DataLoaders`](/data.core.html#DataLoaders).

`show_batch` is a type-dispatched function that is responsible for showing decoded `samples`. `x` and `y` are the input and the target in the batch to be shown, and are passed along to dispatch on their types. There is a different implementation of `show_batch` if `x` is a [`TensorImage`](/torch_core.html#TensorImage) or a [`TensorText`](/text.data.html#TensorText) for instance (see vision.core or text.data for more details). `ctxs` can be passed but the function is responsible to create them if necessary. `kwargs` depend on the specific implementation.

`show_results` is a type-dispatched function that is responsible for showing decoded `samples` and their corresponding `outs`. Like in `show_batch`, `x` and `y` are the input and the target in the batch to be shown, and are passed along to dispatch on their types. `ctxs` can be passed but the function is responsible to create them if necessary. `kwargs` depend on the specific implementation.


<h2 id="TfmdDL" class="doc_header"><code>class</code> <code>TfmdDL</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L40" class="source_link" style="float:right">[source]</a></h2>

> <code>TfmdDL</code>(**`dataset`**, **`bs`**=*`64`*, **`shuffle`**=*`False`*, **`num_workers`**=*`None`*, **`verbose`**=*`False`*, **`do_setup`**=*`True`*, **`pin_memory`**=*`False`*, **`timeout`**=*`0`*, **`batch_size`**=*`None`*, **`drop_last`**=*`False`*, **`indexed`**=*`None`*, **`n`**=*`None`*, **`device`**=*`None`*, **`wif`**=*`None`*, **`before_iter`**=*`None`*, **`after_item`**=*`None`*, **`before_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_iter`**=*`None`*, **`create_batches`**=*`None`*, **`create_item`**=*`None`*, **`create_batch`**=*`None`*, **`retain`**=*`None`*, **`get_idxs`**=*`None`*, **`sample`**=*`None`*, **`shuffle_fn`**=*`None`*, **`do_batch`**=*`None`*) :: [`DataLoader`](/data.load.html#DataLoader)

Transformed [`DataLoader`](/data.load.html#DataLoader)


A [`TfmdDL`](/data.core.html#TfmdDL) is a [`DataLoader`](/data.load.html#DataLoader) that creates [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) from a list of [`Transform`](https://fastcore.fast.ai/transform#Transform)s for the callbacks `after_item`, `before_batch` and `after_batch`. As a result, it can decode or show a processed `batch`.

```python
add_docs(TfmdDL,
         decode="Decode `b` using `tfms`",
         decode_batch="Decode `b` entirely",
         new="Create a new version of self with a few changed attributes",
         show_batch="Show `b` (defaults to `one_batch`), a list of lists of pipeline outputs (i.e. output of a `DataLoader`)",
         show_results="Show each item of `b` and `out`",
         before_iter="override",
         to="Put self and its transforms state on `device`")
```

```python
class _Category(int, ShowTitle): pass
```

```python
class NegTfm(Transform):
    def encodes(self, x): return torch.neg(x)
    def decodes(self, x): return torch.neg(x)
    
tdl = TfmdDL([(TensorImage([1]),)] * 4, after_batch=NegTfm(), bs=4, num_workers=4)
b = tdl.one_batch()
test_eq(type(b[0]), TensorImage)
b = (tensor([1.,1.,1.,1.]),)
test_eq(type(tdl.decode_batch(b)[0][0]), TensorImage)
```

```python
class A(Transform): 
    def encodes(self, x): return x 
    def decodes(self, x): return TitledInt(x) 

@Transform
def f(x)->None: return fastuple((x,x))

start = torch.arange(50)
test_eq_type(f(2), fastuple((2,2)))
```

```python
a = A()
tdl = TfmdDL(start, after_item=lambda x: (a(x), f(x)), bs=4)
x,y = tdl.one_batch()
test_eq(type(y), fastuple)

s = tdl.decode_batch((x,y))
test_eq(type(s[0][1]), fastuple)
```

```python
tdl = TfmdDL(torch.arange(0,50), after_item=A(), after_batch=NegTfm(), bs=4)
test_eq(tdl.dataset[0], start[0])
test_eq(len(tdl), (50-1)//4+1)
test_eq(tdl.bs, 4)
test_stdout(tdl.show_batch, '0\n1\n2\n3')
test_stdout(partial(tdl.show_batch, unique=True), '0\n0\n0\n0')
```

```python
class B(Transform):
    parameters = 'a'
    def __init__(self): self.a = torch.tensor(0.)
    def encodes(self, x): x
    
tdl = TfmdDL([(TensorImage([1]),)] * 4, after_batch=B(), bs=4)
test_eq(tdl.after_batch.fs[0].a.device, torch.device('cpu'))
tdl.to(default_device())
test_eq(tdl.after_batch.fs[0].a.device, default_device())
```

### Methods


<h4 id="DataLoader.one_batch" class="doc_header"><code>DataLoader.one_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/load.py#L130" class="source_link" style="float:right">[source]</a></h4>

> <code>DataLoader.one_batch</code>()




```python
tfm = NegTfm()
tdl = TfmdDL(start, after_batch=tfm, bs=4)
```

```python
b = tdl.one_batch()
test_eq(tensor([0,-1,-2,-3]), b)
```


<h4 id="TfmdDL.decode" class="doc_header"><code>TfmdDL.decode</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L80" class="source_link" style="float:right">[source]</a></h4>

> <code>TfmdDL.decode</code>(**`b`**)




```python
test_eq(tdl.decode(b), tensor(0,1,2,3))
```


<h4 id="TfmdDL.decode_batch" class="doc_header"><code>TfmdDL.decode_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L81" class="source_link" style="float:right">[source]</a></h4>

> <code>TfmdDL.decode_batch</code>(**`b`**, **`max_n`**=*`9`*, **`full`**=*`True`*)




```python
test_eq(tdl.decode_batch(b), [0,1,2,3])
```


<h4 id="TfmdDL.show_batch" class="doc_header"><code>TfmdDL.show_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L96" class="source_link" style="float:right">[source]</a></h4>

> <code>TfmdDL.show_batch</code>(**`b`**=*`None`*, **`max_n`**=*`9`*, **`ctxs`**=*`None`*, **`show`**=*`True`*, **`unique`**=*`False`*, **\*\*`kwargs`**)





<h4 id="TfmdDL.to" class="doc_header"><code>TfmdDL.to</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L119" class="source_link" style="float:right">[source]</a></h4>

> <code>TfmdDL.to</code>(**`device`**)





<h2 id="DataLoaders" class="doc_header"><code>class</code> <code>DataLoaders</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L127" class="source_link" style="float:right">[source]</a></h2>

> <code>DataLoaders</code>(**\*`loaders`**, **`path`**=*`'.'`*, **`device`**=*`None`*) :: [`GetAttr`](https://fastcore.fast.ai/foundation#GetAttr)

Basic wrapper around several [`DataLoader`](/data.load.html#DataLoader)s.


```python
dls = DataLoaders(tdl,tdl)
x = dls.train.one_batch()
x2 = first(tdl)
test_eq(x,x2)
x2 = dls.one_batch()
test_eq(x,x2)
```

### Methods


<h4 id="DataLoaders.__getitem__" class="doc_header"><code>DataLoaders.__getitem__</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L134" class="source_link" style="float:right">[source]</a></h4>

> <code>DataLoaders.__getitem__</code>(**`i`**)

Retrieve [`DataLoader`](/data.load.html#DataLoader) at `i` (`0` is training, `1` is validation)


```python
x2 = dls[0].one_batch()
test_eq(x,x2)
```


<h4 id="DataLoaders.train" class="doc_header"><code>DataLoaders.train</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L140" class="source_link" style="float:right">[source]</a></h4>

Training [`DataLoader`](/data.load.html#DataLoader)



<h4 id="DataLoaders.valid" class="doc_header"><code>DataLoaders.valid</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L140" class="source_link" style="float:right">[source]</a></h4>

Validation [`DataLoader`](/data.load.html#DataLoader)



<h4 id="DataLoaders.train_ds" class="doc_header"><code>DataLoaders.train_ds</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L141" class="source_link" style="float:right">[source]</a></h4>

Training `Dataset`



<h4 id="DataLoaders.valid_ds" class="doc_header"><code>DataLoaders.valid_ds</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L141" class="source_link" style="float:right">[source]</a></h4>

Validation `Dataset`



<h2 id="FilteredBase" class="doc_header"><code>class</code> <code>FilteredBase</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L184" class="source_link" style="float:right">[source]</a></h2>

> <code>FilteredBase</code>(**\*`args`**, **`dl_type`**=*`None`*, **\*\*`kwargs`**)

Base class for lists with subsets



<h2 id="TfmdLists" class="doc_header"><code>class</code> <code>TfmdLists</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L212" class="source_link" style="float:right">[source]</a></h2>

> <code>TfmdLists</code>(**`items`**, **`tfms`**, **`use_list`**=*`None`*, **`do_setup`**=*`True`*, **`split_idx`**=*`None`*, **`train_setup`**=*`True`*, **`splits`**=*`None`*, **`types`**=*`None`*, **`verbose`**=*`False`*, **`dl_type`**=*`None`*) :: [`FilteredBase`](/data.core.html#FilteredBase)

A [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) of `tfms` applied to a collection of `items`


```python
add_docs(TfmdLists,
         setup="Transform setup with self",
         decode="From `Pipeline",
         show="From `Pipeline",
         overlapping_splits="All splits that are in more than one split",
         subset="New `TfmdLists` with same tfms that only includes items in `i`th split",
         infer_idx="Finds the index where `self.tfms` can be applied to `x`, depending on the type of `x`",
         infer="Apply `self.tfms` to `x` starting at the right tfm depending on the type of `x`",
         new_empty="A new version of `self` but with no items")
```

```python
def decode_at(o, idx):
    "Decoded item at `idx`"
    return o.decode(o[idx])
```


<h4 id="decode_at" class="doc_header"><code>decode_at</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L273" class="source_link" style="float:right">[source]</a></h4>

> <code>decode_at</code>(**`o`**, **`idx`**)

Decoded item at `idx`


```python
def show_at(o, idx, **kwargs):
    "Show item at `idx`",
    return o.show(o[idx], **kwargs)
```


<h4 id="show_at" class="doc_header"><code>show_at</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L278" class="source_link" style="float:right">[source]</a></h4>

> <code>show_at</code>(**`o`**, **`idx`**, **\*\*`kwargs`**)




A [`TfmdLists`](/data.core.html#TfmdLists) combines a collection of object with a [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline). `tfms` can either be a [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) or a list of transforms, in which case, it will wrap them in a [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline). `use_list` is passed along to [`L`](https://fastcore.fast.ai/foundation#L) with the `items` and `split_idx` are passed to each transform of the [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline). `do_setup` indicates if the [`Pipeline.setup`](https://fastcore.fast.ai/transform#Pipeline.setup) method should be called during initialization.

```python
class _IntFloatTfm(Transform):
    def encodes(self, o):  return TitledInt(o)
    def decodes(self, o):  return TitledFloat(o)
int2f_tfm=_IntFloatTfm()

def _neg(o): return -o
neg_tfm = Transform(_neg, _neg)
```

```python
items = L([1.,2.,3.]); tfms = [neg_tfm, int2f_tfm]
tl = TfmdLists(items, tfms=tfms)
test_eq_type(tl[0], TitledInt(-1))
test_eq_type(tl[1], TitledInt(-2))
test_eq_type(tl.decode(tl[2]), TitledFloat(3.))
test_stdout(lambda: show_at(tl, 2), '-3')
test_eq(tl.types, [float, float, TitledInt])
tl
```




    TfmdLists: [1.0, 2.0, 3.0]
    tfms - (#2) [_neg:
    (object,object) -> _neg (object,object) -> _neg,_IntFloatTfm:
    (object,object) -> encodes
     (object,object) -> decodes
    ]



```python
splits = [[0,2],[1]]
tl = TfmdLists(items, tfms=tfms, splits=splits)
test_eq(tl.n_subsets, 2)
test_eq(tl.train, tl.subset(0))
test_eq(tl.valid, tl.subset(1))
test_eq(tl.train.items, items[splits[0]])
test_eq(tl.valid.items, items[splits[1]])
test_eq(tl.train.tfms.split_idx, 0)
test_eq(tl.valid.tfms.split_idx, 1)
test_eq(tl.train.new_empty().split_idx, 0)
test_eq(tl.valid.new_empty().split_idx, 1)
test_eq_type(tl.splits, L(splits))
assert not tl.overlapping_splits()
```

```python
df = pd.DataFrame(dict(a=[1,2,3],b=[2,3,4]))
tl = TfmdLists(df, lambda o: o.a+1, splits=[[0],[1,2]])
test_eq(tl[1,2], [3,4])
tr = tl.subset(0)
test_eq(tr[:], [2])
val = tl.subset(1)
test_eq(val[:], [3,4])
```

```python
class _B(Transform):
    def __init__(self): self.m = 0
    def encodes(self, o): return o+self.m
    def decodes(self, o): return o-self.m
    def setups(self, items): 
        print(items)
        self.m = tensor(items).float().mean().item()

# test for setup, which updates `self.m`
tl = TfmdLists(items, _B())
test_eq(tl.m, 2)
```

    TfmdLists: [1.0, 2.0, 3.0]
    tfms - (#0) []


Here's how we can use [`TfmdLists.setup`](/data.core.html#TfmdLists.setup) to implement a simple category list, getting labels from a mock file list:

```python
class _Cat(Transform):
    order = 1
    def encodes(self, o):    return int(self.o2i[o])
    def decodes(self, o):    return TitledStr(self.vocab[o])
    def setups(self, items): self.vocab,self.o2i = uniqueify(L(items), sort=True, bidir=True)
tcat = _Cat()

def _lbl(o): return TitledStr(o.split('_')[0])

# Check that tfms are sorted by `order` & `_lbl` is called first
fns = ['dog_0.jpg','cat_0.jpg','cat_2.jpg','cat_1.jpg','dog_1.jpg']
tl = TfmdLists(fns, [tcat,_lbl])
exp_voc = ['cat','dog']
test_eq(tcat.vocab, exp_voc)
test_eq(tl.tfms.vocab, exp_voc)
test_eq(tl.vocab, exp_voc)
test_eq(tl, (1,0,0,0,1))
test_eq([tl.decode(o) for o in tl], ('dog','cat','cat','cat','dog'))
```

```python
tl = TfmdLists(fns, [tcat,_lbl], splits=[[0,4], [1,2,3]])
test_eq(tcat.vocab, ['dog'])
```

```python
tfm = NegTfm(split_idx=1)
tds = TfmdLists(start, A())
tdl = TfmdDL(tds, after_batch=tfm, bs=4)
x = tdl.one_batch()
test_eq(x, torch.arange(4))
tds.split_idx = 1
x = tdl.one_batch()
test_eq(x, -torch.arange(4))
tds.split_idx = 0
x = tdl.one_batch()
test_eq(x, torch.arange(4))
```

```python
tds = TfmdLists(start, A())
tdl = TfmdDL(tds, after_batch=NegTfm(), bs=4)
test_eq(tdl.dataset[0], start[0])
test_eq(len(tdl), (len(tds)-1)//4+1)
test_eq(tdl.bs, 4)
test_stdout(tdl.show_batch, '0\n1\n2\n3')
```


<h4 id="TfmdLists.subset" class="doc_header"><code>TfmdLists.subset</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L231" class="source_link" style="float:right">[source]</a></h4>

> <code>TfmdLists.subset</code>(**`i`**)





<h4 id="TfmdLists.infer_idx" class="doc_header"><code>TfmdLists.infer_idx</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L253" class="source_link" style="float:right">[source]</a></h4>

> <code>TfmdLists.infer_idx</code>(**`x`**)





<h4 id="TfmdLists.infer" class="doc_header"><code>TfmdLists.infer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L264" class="source_link" style="float:right">[source]</a></h4>

> <code>TfmdLists.infer</code>(**`x`**)




```python
def mult(x): return x*2
mult.order = 2

fns = ['dog_0.jpg','cat_0.jpg','cat_2.jpg','cat_1.jpg','dog_1.jpg']
tl = TfmdLists(fns, [_lbl,_Cat(),mult])

test_eq(tl.infer_idx('dog_45.jpg'), 0)
test_eq(tl.infer('dog_45.jpg'), 2)

test_eq(tl.infer_idx(4), 2)
test_eq(tl.infer(4), 8)

test_fail(lambda: tl.infer_idx(2.0))
test_fail(lambda: tl.infer(2.0))
```


<h2 id="Datasets" class="doc_header"><code>class</code> <code>Datasets</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L285" class="source_link" style="float:right">[source]</a></h2>

> <code>Datasets</code>(**`items`**=*`None`*, **`tfms`**=*`None`*, **`tls`**=*`None`*, **`n_inp`**=*`None`*, **`dl_type`**=*`None`*, **`use_list`**=*`None`*, **`do_setup`**=*`True`*, **`split_idx`**=*`None`*, **`train_setup`**=*`True`*, **`splits`**=*`None`*, **`types`**=*`None`*, **`verbose`**=*`False`*) :: [`FilteredBase`](/data.core.html#FilteredBase)

A dataset that creates a tuple from each `tfms`, passed thru `item_tfms`


A [`Datasets`](/data.core.html#Datasets) creates a tuple from `items` (typically input,target) by applying to them each list of [`Transform`](https://fastcore.fast.ai/transform#Transform) (or [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline)) in `tfms`. Note that if `tfms` contains only one list of `tfms`, the items given by [`Datasets`](/data.core.html#Datasets) will be tuples of one element. 

`n_inp` is the number of elements in the tuples that should be considered part of the input and will default to 1 if `tfms` consists of one set of transforms, `len(tfms)-1` otherwise. In most cases, the number of elements in the tuples spit out by [`Datasets`](/data.core.html#Datasets) will be 2 (for input,target) but it can happen that there is 3 (Siamese networks or tabular data) in which case we need to be able to determine when the inputs end and the targets begin.

```python
items = [1,2,3,4]
dsets = Datasets(items, [[neg_tfm,int2f_tfm], [add(1)]])
t = dsets[0]
test_eq(t, (-1,2))
test_eq(dsets[0,1,2], [(-1,2),(-2,3),(-3,4)])
test_eq(dsets.n_inp, 1)
dsets.decode(t)
```




    (1.0, 2)



```python
class Norm(Transform):
    def encodes(self, o): return (o-self.m)/self.s
    def decodes(self, o): return (o*self.s)+self.m
    def setups(self, items):
        its = tensor(items).float()
        self.m,self.s = its.mean(),its.std()
```

```python
items = [1,2,3,4]
nrm = Norm()
dsets = Datasets(items, [[neg_tfm,int2f_tfm], [neg_tfm,nrm]])

x,y = zip(*dsets)
test_close(tensor(y).mean(), 0)
test_close(tensor(y).std(), 1)
test_eq(x, (-1,-2,-3,-4,))
test_eq(nrm.m, -2.5)
test_stdout(lambda:show_at(dsets, 1), '-2')

test_eq(dsets.m, nrm.m)
test_eq(dsets.norm.m, nrm.m)
test_eq(dsets.train.norm.m, nrm.m)
```

```python
test_fns = ['dog_0.jpg','cat_0.jpg','cat_2.jpg','cat_1.jpg','kid_1.jpg']
tcat = _Cat()
dsets = Datasets(test_fns, [[tcat,_lbl]], splits=[[0,1,2], [3,4]])
test_eq(tcat.vocab, ['cat','dog'])
test_eq(dsets.train, [(1,),(0,),(0,)])
test_eq(dsets.valid[0], (0,))
test_stdout(lambda: show_at(dsets.train, 0), "dog")
```

```python
inp = [0,1,2,3,4]
dsets = Datasets(inp, tfms=[None])

test_eq(*dsets[2], 2)          # Retrieve one item (subset 0 is the default)
test_eq(dsets[1,2], [(1,),(2,)])    # Retrieve two items by index
mask = [True,False,False,True,False]
test_eq(dsets[mask], [(0,),(3,)])   # Retrieve two items by mask
```

```python
inp = pd.DataFrame(dict(a=[5,1,2,3,4]))
dsets = Datasets(inp, tfms=attrgetter('a')).subset(0)
test_eq(*dsets[2], 2)          # Retrieve one item (subset 0 is the default)
test_eq(dsets[1,2], [(1,),(2,)])    # Retrieve two items by index
mask = [True,False,False,True,False]
test_eq(dsets[mask], [(5,),(3,)])   # Retrieve two items by mask
```

```python
inp = [0,1,2,3,4]
dsets = Datasets(inp, tfms=[None])
test_eq(dsets.n_inp, 1)
dsets = Datasets(inp, tfms=[[None],[None],[None]])
test_eq(dsets.n_inp, 2)
dsets = Datasets(inp, tfms=[[None],[None],[None]], n_inp=1)
test_eq(dsets.n_inp, 1)
```

```python
dsets = Datasets(range(5), tfms=[None], splits=[tensor([0,2]), [1,3,4]])

test_eq(dsets.subset(0), [(0,),(2,)])
test_eq(dsets.train, [(0,),(2,)])       # Subset 0 is aliased to `train`
test_eq(dsets.subset(1), [(1,),(3,),(4,)])
test_eq(dsets.valid, [(1,),(3,),(4,)])     # Subset 1 is aliased to `valid`
test_eq(*dsets.valid[2], 4)
#assert '[(1,),(3,),(4,)]' in str(dsets) and '[(0,),(2,)]' in str(dsets)
dsets
```




    (#5) [(0,),(1,),(2,),(3,),(4,)]



```python
splits = [[False,True,True,False,True], [True,False,False,False,False]]
dsets = Datasets(range(5), tfms=[None], splits=splits)

test_eq(dsets.train, [(1,),(2,),(4,)])
test_eq(dsets.valid, [(0,)])
```

```python
tfm = [[lambda x: x*2,lambda x: x+1]]
splits = [[1,2],[0,3,4]]
dsets = Datasets(range(5), tfm, splits=splits)
test_eq(dsets.train,[(3,),(5,)])
test_eq(dsets.valid,[(1,),(7,),(9,)])
test_eq(dsets.train[False,True], [(5,)])
```

```python
class _Tfm(Transform):
    split_idx=1
    def encodes(self, x): return x*2
    def decodes(self, x): return TitledStr(x//2)
```

```python
dsets = Datasets(range(5), [_Tfm()], splits=[[1,2],[0,3,4]])
test_eq(dsets.train,[(1,),(2,)])
test_eq(dsets.valid,[(0,),(6,),(8,)])
test_eq(dsets.train[False,True], [(2,)])
dsets
```




    (#5) [(0,),(1,),(2,),(3,),(4,)]



```python
ds = dsets.train
with ds.set_split_idx(1):
    test_eq(ds,[(2,),(4,)])
test_eq(dsets.train,[(1,),(2,)])
```

```python
dsets = Datasets(range(5), [_Tfm(),noop], splits=[[1,2],[0,3,4]])
test_eq(dsets.train,[(1,1),(2,2)])
test_eq(dsets.valid,[(0,0),(6,3),(8,4)])
```

```python
start = torch.arange(0,50)
tds = Datasets(start, [A()])
tdl = TfmdDL(tds, after_item=NegTfm(), bs=4)
b = tdl.one_batch()
test_eq(tdl.decode_batch(b), ((0,),(1,),(2,),(3,)))
test_stdout(tdl.show_batch, "0\n1\n2\n3")
```

```python
class _Tfm(Transform):
    split_idx=1
    def encodes(self, x): return x*2

dsets = Datasets(range(8), [None], splits=[[1,2,5,7],[0,3,4,6]])
```

```python
class _Tfm(Transform):
    split_idx=1
    def encodes(self, x): return x*2

dsets = Datasets(range(8), [None], splits=[[1,2,5,7],[0,3,4,6]])
dls = dsets.dataloaders(bs=4, after_batch=_Tfm(), shuffle_train=False, device=torch.device('cpu'))
test_eq(dls.train, [(tensor([1,2,5, 7]),)])
test_eq(dls.valid, [(tensor([0,6,8,12]),)])
test_eq(dls.n_inp, 1)
```

### Methods

```python
items = [1,2,3,4]
dsets = Datasets(items, [[neg_tfm,int2f_tfm]])
```


<h4 id="Datasets.dataloaders" class="doc_header"><code>Datasets.dataloaders</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L197" class="source_link" style="float:right">[source]</a></h4>

> <code>Datasets.dataloaders</code>(**`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`n`**=*`None`*, **`path`**=*`'.'`*, **`dl_type`**=*`None`*, **`dl_kwargs`**=*`None`*, **`device`**=*`None`*, **`shuffle`**=*`False`*, **`num_workers`**=*`None`*, **`verbose`**=*`False`*, **`do_setup`**=*`True`*, **`pin_memory`**=*`False`*, **`timeout`**=*`0`*, **`batch_size`**=*`None`*, **`drop_last`**=*`False`*, **`indexed`**=*`None`*, **`wif`**=*`None`*, **`before_iter`**=*`None`*, **`after_item`**=*`None`*, **`before_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_iter`**=*`None`*, **`create_batches`**=*`None`*, **`create_item`**=*`None`*, **`create_batch`**=*`None`*, **`retain`**=*`None`*, **`get_idxs`**=*`None`*, **`sample`**=*`None`*, **`shuffle_fn`**=*`None`*, **`do_batch`**=*`None`*)

Get a [`DataLoaders`](/data.core.html#DataLoaders)



<h4 id="Datasets.decode" class="doc_header"><code>Datasets.decode</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L301" class="source_link" style="float:right">[source]</a></h4>

> <code>Datasets.decode</code>(**`o`**, **`full`**=*`True`*)

Compose `decode` of all `tuple_tfms` then all `tfms` on `i`


```python
test_eq(*dsets[0], -1)
test_eq(*dsets.decode((-1,)), 1)
```


<h4 id="Datasets.show" class="doc_header"><code>Datasets.show</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L316" class="source_link" style="float:right">[source]</a></h4>

> <code>Datasets.show</code>(**`o`**, **`ctx`**=*`None`*, **\*\*`kwargs`**)

Show item `o` in `ctx`


```python
test_stdout(lambda:dsets.show(dsets[1]), '-2')
```


<h4 id="Datasets.new_empty" class="doc_header"><code>Datasets.new_empty</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L305" class="source_link" style="float:right">[source]</a></h4>

> <code>Datasets.new_empty</code>()

Create a new empty version of the `self`, keeping only the transforms


```python
items = [1,2,3,4]
nrm = Norm()
dsets = Datasets(items, [[neg_tfm,int2f_tfm], [neg_tfm]])
empty = dsets.new_empty()
test_eq(empty.items, [])
```

## Add test set for inference

```python
class _Tfm1(Transform):
    split_idx=0
    def encodes(self, x): return x*3

dsets = Datasets(range(8), [[_Tfm(),_Tfm1()]], splits=[[1,2,5,7],[0,3,4,6]])
test_eq(dsets.train, [(3,),(6,),(15,),(21,)])
test_eq(dsets.valid, [(0,),(6,),(8,),(12,)])
```


<h4 id="test_set" class="doc_header"><code>test_set</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L339" class="source_link" style="float:right">[source]</a></h4>

> <code>test_set</code>(**`dsets`**, **`test_items`**, **`rm_tfms`**=*`None`*, **`with_labels`**=*`False`*)

Create a test set from `test_items` using validation transforms of `dsets`


```python
class _Tfm1(Transform):
    split_idx=0
    def encodes(self, x): return x*3

dsets = Datasets(range(8), [[_Tfm(),_Tfm1()]], splits=[[1,2,5,7],[0,3,4,6]])
test_eq(dsets.train, [(3,),(6,),(15,),(21,)])
test_eq(dsets.valid, [(0,),(6,),(8,),(12,)])

#Tranform of the validation set are applied
tst = test_set(dsets, [1,2,3])
test_eq(tst, [(2,),(4,),(6,)])
```


<h4 id="DataLoaders.test_dl" class="doc_header"><code>DataLoaders.test_dl</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/core.py#L356" class="source_link" style="float:right">[source]</a></h4>

> <code>DataLoaders.test_dl</code>(**`test_items`**, **`rm_type_tfms`**=*`None`*, **`with_labels`**=*`False`*, **\*\*`kwargs`**)

Create a test dataloader from `test_items` using validation transforms of `dls`


```python
dsets = Datasets(range(8), [[_Tfm(),_Tfm1()]], splits=[[1,2,5,7],[0,3,4,6]])
dls = dsets.dataloaders(bs=4, device=torch.device('cpu'))
```

```python
dsets = Datasets(range(8), [[_Tfm(),_Tfm1()]], splits=[[1,2,5,7],[0,3,4,6]])
dls = dsets.dataloaders(bs=4, device=torch.device('cpu'))
tst_dl = dls.test_dl([2,3,4,5])
test_eq(tst_dl._n_inp, 1)
test_eq(list(tst_dl), [(tensor([ 4,  6,  8, 10]),)])
#Test you can change transforms
tst_dl = dls.test_dl([2,3,4,5], after_item=add1)
test_eq(list(tst_dl), [(tensor([ 5,  7,  9, 11]),)])
```
