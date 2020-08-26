# fastai DataLoader
> API compatible with PyTorch DataLoader, with a lot more callbacks and flexibility


```python
bs = 4
letters = list(string.ascii_lowercase)
```

## DataLoader helpers

fastai includes a replacement for Pytorch's *DataLoader* which is largely API-compatible, and adds a lot of useful functionality and flexibility. Before we look at the class, there are a couple of helpers we'll need to define.


<h4 id="fa_collate" class="doc_header"><code>fa_collate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/load.py#L43" class="source_link" style="float:right">[source]</a></h4>

> <code>fa_collate</code>(**`t`**)




```python
t = [(1,(2,3)),(1,(2,3))]
test_eq(fa_collate(t), default_collate(t))
test_eq(L(fa_collate(t)).map(type), [Tensor,tuple])

t = [(1,(2,(3,4))),(1,(2,(3,4)))]
test_eq(fa_collate(t), default_collate(t))
test_eq(L(fa_collate(t)).map(type), [Tensor,tuple])
test_eq(L(fa_collate(t)[1]).map(type), [Tensor,tuple])
```


<h4 id="fa_convert" class="doc_header"><code>fa_convert</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/load.py#L50" class="source_link" style="float:right">[source]</a></h4>

> <code>fa_convert</code>(**`t`**)




```python
t0 = array([1,2])
t = [t0,(t0,t0)]

test_eq(fa_convert(t), default_convert(t))
test_eq(L(fa_convert(t)).map(type), [Tensor,tuple])
```


<h3 id="SkipItemException" class="doc_header"><code>class</code> <code>SkipItemException</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/load.py#L56" class="source_link" style="float:right">[source]</a></h3>

> <code>SkipItemException</code>() :: `Exception`

Common base class for all non-exit exceptions.



<h2 id="DataLoader" class="doc_header"><code>class</code> <code>DataLoader</code><a href="torch/utils/data/dataloader.py#L60" class="source_link" style="float:right">[source]</a></h2>

> <code>DataLoader</code>(**`dataset`**, **`batch_size`**=*`1`*, **`shuffle`**=*`False`*, **`sampler`**=*`None`*, **`batch_sampler`**=*`None`*, **`num_workers`**=*`0`*, **`collate_fn`**=*`None`*, **`pin_memory`**=*`False`*, **`drop_last`**=*`False`*, **`timeout`**=*`0`*, **`worker_init_fn`**=*`None`*, **`multiprocessing_context`**=*`None`*, **`generator`**=*`None`*)

Data loader. Combines a dataset and a sampler, and provides an iterable over
the given dataset.

The :class:`~torch.utils.data.DataLoader` supports both map-style and
iterable-style datasets with single- or multi-process loading, customizing
loading order and optional automatic batching (collation) and memory pinning.

See :py:mod:`torch.utils.data` documentation page for more details.

Arguments:
    dataset (Dataset): dataset from which to load the data.
    batch_size (int, optional): how many samples per batch to load
        (default: ``1``).
    shuffle (bool, optional): set to ``True`` to have the data reshuffled
        at every epoch (default: ``False``).
    sampler (Sampler or Iterable, optional): defines the strategy to draw
        samples from the dataset. Can be any ``Iterable`` with ``__len__``
        implemented. If specified, :attr:`shuffle` must not be specified.
    batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
        returns a batch of indices at a time. Mutually exclusive with
        :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
        and :attr:`drop_last`.
    num_workers (int, optional): how many subprocesses to use for data
        loading. ``0`` means that the data will be loaded in the main process.
        (default: ``0``)
    collate_fn (callable, optional): merges a list of samples to form a
        mini-batch of Tensor(s).  Used when using batched loading from a
        map-style dataset.
    pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
        into CUDA pinned memory before returning them.  If your data elements
        are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
        see the example below.
    drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)
    timeout (numeric, optional): if positive, the timeout value for collecting a batch
        from workers. Should always be non-negative. (default: ``0``)
    worker_init_fn (callable, optional): If not ``None``, this will be called on each
        worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
        input, after seeding and before data loading. (default: ``None``)


.. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
             cannot be an unpicklable object, e.g., a lambda function. See
             :ref:`multiprocessing-best-practices` on more details related
             to multiprocessing in PyTorch.

.. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
             When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
             it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
             rounding depending on :attr:`drop_last`, regardless of multi-process loading
             configurations. This represents the best guess PyTorch can make because PyTorch
             trusts user :attr:`dataset` code in correctly handling multi-process
             loading to avoid duplicate data.

             However, if sharding results in multiple workers having incomplete last batches,
             this estimate can still be inaccurate, because (1) an otherwise complete batch can
             be broken into multiple ones and (2) more than one batch worth of samples can be
             dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
             cases in general.

             See `Dataset Types`_ for more details on these two types of datasets and how
             :class:`~torch.utils.data.IterableDataset` interacts with
             `Multi-process data loading`_.


Arguments to [`DataLoader`](/data.load.html#DataLoader):
* `dataset`: dataset from which to load the data. Can be either map-style or iterable-style dataset.
* `bs` (int): how many samples per batch to load (if `batch_size` is provided then `batch_size` will override `bs`). If `bs=None`, then it is assumed that `dataset.__getitem__` returns a batch.
* `num_workers` (int): how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process.
* `pin_memory` (bool): If `True`, the data loader will copy Tensors into CUDA pinned memory before returning them.
* `timeout` (float>0): the timeout value in seconds for collecting a batch from workers.
* `batch_size` (int): It is only provided for PyTorch compatibility. Use `bs`.
* `shuffle` (bool): If `True`, then data is shuffled every time dataloader is fully read/iterated.
* `drop_last` (bool): If `True`, then the last incomplete batch is dropped.
* `indexed` (bool): Set to `False`, if you are using iterable-style dataset. Otherwise it is set to `True` by default.
* `n` (int): Defaults to `len(dataset)`. If you are using iterable-style dataset, you can specify the size of batch using `n`.
* `device` (torch.device): Defaults to `default_device()` which is CUDA by default. You can specify device as `torch.device('cpu').

Override `item` and use the default infinite sampler to get a stream of unknown length (`stop()` when you want to stop the stream).

```python
class RandDL(DataLoader):
    def create_item(self, s):
        r = random.random()
        return r if r<0.95 else stop()

L(RandDL())
```




    (#39) [0.24813649034165253,0.7183731274355801,0.46707243201681625,0.3819386594236871,0.37001598891686693,0.00048487460559387685,0.37430607545258365,0.28122066066872486,0.7108328343496174,0.4052328635806347...]



```python
L(RandDL(bs=4, drop_last=True)).map(len)
```




    (#1) [4]



```python
dl = RandDL(bs=4, num_workers=4, drop_last=True)
L(dl).map(len)
```




    (#15) [4,4,4,4,4,4,4,4,4,4...]



```python
test_eq(dl.fake_l.num_workers, 4)
with dl.fake_l.no_multiproc(): 
    test_eq(dl.fake_l.num_workers, 0)
    L(dl).map(len)
test_eq(dl.fake_l.num_workers, 4)
```

```python
def _rand_item(s):
    r = random.random()
    return r if r<0.95 else stop()

L(DataLoader(create_item=_rand_item))
```




    (#18) [0.09369657046580104,0.022311107860009227,0.12902272918569346,0.8060082768103013,0.2512204187078644,0.40772772960651604,0.2115850693953002,0.23026583510965482,0.7840788021237788,0.18360739628018286...]



If you don't set `bs`, then `dataset` is assumed to provide an iterator or a `__getitem__` that returns a batch.

```python
ds1 = DataLoader(letters)
test_eq(L(ds1), letters)
test_eq(len(ds1), 26)

test_shuffled(L(DataLoader(letters, shuffle=True)), letters)

ds1 = DataLoader(letters, indexed=False)
test_eq(L(ds1), letters)
test_eq(len(ds1), 26)

t2 = L(tensor([0,1,2]),tensor([3,4,5]))
ds2 = DataLoader(t2)
test_eq_type(L(ds2), t2)

t3 = L(array([0,1,2]),array([3,4,5]))
ds3 = DataLoader(t3)
test_eq_type(L(ds3), t3.map(tensor))

ds4 = DataLoader(t3, create_batch=noop, after_iter=lambda: setattr(t3, 'f', 1))
test_eq_type(L(ds4), t3)
test_eq(t3.f, 1)
```

If you do set `bs`, then `dataset` is assumed to provide an iterator or a `__getitem__` that returns a single item of a batch.

```python
def twoepochs(d): return ' '.join(''.join(list(o)) for _ in range(2) for o in d)
```

```python
ds1 = DataLoader(letters, bs=4, drop_last=True, num_workers=0)
test_eq(twoepochs(ds1), 'abcd efgh ijkl mnop qrst uvwx abcd efgh ijkl mnop qrst uvwx')

ds1 = DataLoader(letters,4,num_workers=2)
test_eq(twoepochs(ds1), 'abcd efgh ijkl mnop qrst uvwx yz abcd efgh ijkl mnop qrst uvwx yz')

ds1 = DataLoader(range(12), bs=4, num_workers=3)
test_eq_type(L(ds1), L(tensor([0,1,2,3]),tensor([4,5,6,7]),tensor([8,9,10,11])))

ds1 = DataLoader([str(i) for i in range(11)], bs=4, after_iter=lambda: setattr(t3, 'f', 2))
test_eq_type(L(ds1), L(['0','1','2','3'],['4','5','6','7'],['8','9','10']))
test_eq(t3.f, 2)

it = iter(DataLoader(map(noop,range(20)), bs=4, num_workers=1))
test_eq_type([next(it) for _ in range(3)], [tensor([0,1,2,3]),tensor([4,5,6,7]),tensor([8,9,10,11])])
```

```python
class SleepyDL(list):
    def __getitem__(self,i):
        time.sleep(random.random()/50)
        return super().__getitem__(i)

t = SleepyDL(letters)

%time test_eq(DataLoader(t, num_workers=0), letters)
%time test_eq(DataLoader(t, num_workers=2), letters)
%time test_eq(DataLoader(t, num_workers=4), letters)

dl = DataLoader(t, shuffle=True, num_workers=1)
test_shuffled(L(dl), letters)
test_shuffled(L(dl), L(dl))
```

    CPU times: user 8 ms, sys: 0 ns, total: 8 ms
    Wall time: 249 ms
    CPU times: user 12 ms, sys: 16 ms, total: 28 ms
    Wall time: 160 ms
    CPU times: user 20 ms, sys: 28 ms, total: 48 ms
    Wall time: 116 ms


```python
class SleepyQueue():
    "Simulate a queue with varying latency"
    def __init__(self, q): self.q=q
    def __iter__(self):
        while True:
            time.sleep(random.random()/100)
            try: yield self.q.get_nowait()
            except queues.Empty: return

q = Queue()
for o in range(30): q.put(o)
it = SleepyQueue(q)

%time test_shuffled(L(DataLoader(it, num_workers=4)), range(30))
```

    CPU times: user 8 ms, sys: 36 ms, total: 44 ms
    Wall time: 104 ms


```python
class A(TensorBase): pass

for nw in (0,2):
    t = A(tensor([1,2]))
    dl = DataLoader([t,t,t,t,t,t,t,t], bs=4, num_workers=nw)
    b = first(dl)
    test_eq(type(b), A)

    t = (A(tensor([1,2])),)
    dl = DataLoader([t,t,t,t,t,t,t,t], bs=4, num_workers=nw)
    b = first(dl)
    test_eq(type(b[0]), A)
```

```python
class A(TensorBase): pass
t = A(tensor(1,2))

tdl = DataLoader([t,t,t,t,t,t,t,t], bs=4, num_workers=2, after_batch=to_device)
b = first(tdl)
test_eq(type(b), A)

# Unknown attributes are delegated to `dataset`
test_eq(tdl.pop(), tensor(1,2))
```
