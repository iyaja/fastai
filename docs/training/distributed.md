# Distributed and parallel training
> Callbacks and helper functions to train in parallel or use distributed training


## Parallel

Patch the parallel models so they work with RNNs


<h4 id="DataParallel.reset" class="doc_header"><code>DataParallel.reset</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L12" class="source_link" style="float:right">[source]</a></h4>

> <code>DataParallel.reset</code>()





<h2 id="ParallelTrainer" class="doc_header"><code>class</code> <code>ParallelTrainer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L18" class="source_link" style="float:right">[source]</a></h2>

> <code>ParallelTrainer</code>(**`device_ids`**) :: [`Callback`](/callback.core.html#Callback)

Basic class handling tweaks of the training loop by changing a [`Learner`](/learner.html#Learner) in various events



<h4 id="Learner.to_parallel" class="doc_header"><code>Learner.to_parallel</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L25" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.to_parallel</code>(**`device_ids`**=*`None`*)





<h4 id="Learner.detach_parallel" class="doc_header"><code>Learner.detach_parallel</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L31" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.detach_parallel</code>()

Remove ParallelTrainer callback from Learner.



<h4 id="Learner.parallel_ctx" class="doc_header"><code>Learner.parallel_ctx</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L38" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.parallel_ctx</code>(**`device_ids`**=*`None`*)

A context manager to adapt a learner to train in data parallel mode.


## Distributed

Patch the parallel models so they work with RNNs


<h4 id="DistributedDataParallel.reset" class="doc_header"><code>DistributedDataParallel.reset</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L49" class="source_link" style="float:right">[source]</a></h4>

> <code>DistributedDataParallel.reset</code>()




Convenience functions to set up/tear down torch distributed data parallel mode.


<h4 id="setup_distrib" class="doc_header"><code>setup_distrib</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L54" class="source_link" style="float:right">[source]</a></h4>

> <code>setup_distrib</code>(**`gpu`**=*`None`*)





<h4 id="teardown_distrib" class="doc_header"><code>teardown_distrib</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L63" class="source_link" style="float:right">[source]</a></h4>

> <code>teardown_distrib</code>()




### DataLoader

We need to change the dataloaders so that they only get one part of the batch each (otherwise there is no point in using distributed training).


<h2 id="DistributedDL" class="doc_header"><code>class</code> <code>DistributedDL</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L69" class="source_link" style="float:right">[source]</a></h2>

> <code>DistributedDL</code>(**`dataset`**, **`rank`**, **`world_size`**, **`bs`**=*`64`*, **`shuffle`**=*`False`*, **`num_workers`**=*`None`*, **`verbose`**=*`False`*, **`do_setup`**=*`True`*, **`pin_memory`**=*`False`*, **`timeout`**=*`0`*, **`batch_size`**=*`None`*, **`drop_last`**=*`False`*, **`indexed`**=*`None`*, **`n`**=*`None`*, **`device`**=*`None`*, **`wif`**=*`None`*, **`before_iter`**=*`None`*, **`after_item`**=*`None`*, **`before_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_iter`**=*`None`*, **`create_batches`**=*`None`*, **`create_item`**=*`None`*, **`create_batch`**=*`None`*, **`retain`**=*`None`*, **`get_idxs`**=*`None`*, **`sample`**=*`None`*, **`shuffle_fn`**=*`None`*, **`do_batch`**=*`None`*) :: [`TfmdDL`](/data.core.html#TfmdDL)

Transformed [`DataLoader`](/data.load.html#DataLoader)


```python
_tmp_file = tempfile.NamedTemporaryFile().name # i tried putting this inside self / _broadcast to no avail
# patch _broadcast with a mocked version so we can test DistributedDL w/o a proper DDP setup
@patch
def _broadcast(self:DistributedDL,t,rank):
    t = LongTensor(t)
    if rank == self.rank: torch.save(t,_tmp_file)
    else:                 t.data = torch.load(_tmp_file)
    return t.tolist()
```

```python
dl = TfmdDL(list(range(50)), bs=16, num_workers=2)
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    test_eq(list(dl1)[0], torch.arange(i, 52, 4)%50)
```

```python
dl = TfmdDL(list(range(50)), bs=16, num_workers=2, shuffle=True)
res = []
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    res += list(dl1)[0].tolist()
#All items should be sampled (we cannot test order b/c shuffle=True)
test_eq(np.unique(res), np.arange(50))
```

```python
from fastai.callback.data import WeightedDL
```

```python
dl = WeightedDL(list(range(50)), bs=16, num_workers=2, shuffle=True,wgts=list(np.arange(50)>=25))
res = []
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    res += list(dl1)[0].tolist()
test(res,[25]*len(res),operator.ge)        # all res >=25
test(res,[25]*len(res),lambda a,b: ~(a<b)) # all res NOT < 25
```


<h2 id="DistributedTrainer" class="doc_header"><code>class</code> <code>DistributedTrainer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L111" class="source_link" style="float:right">[source]</a></h2>

> <code>DistributedTrainer</code>(**`cuda_id`**=*`0`*) :: [`Callback`](/callback.core.html#Callback)

Basic class handling tweaks of the training loop by changing a [`Learner`](/learner.html#Learner) in various events


Attach, remove a callback which adapts the model to use DistributedDL to train in distributed data parallel mode.


<h4 id="Learner.to_distributed" class="doc_header"><code>Learner.to_distributed</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L137" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.to_distributed</code>(**`cuda_id`**)





<h4 id="Learner.detach_distributed" class="doc_header"><code>Learner.detach_distributed</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L144" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.detach_distributed</code>()





<h4 id="Learner.distrib_ctx" class="doc_header"><code>Learner.distrib_ctx</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L152" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.distrib_ctx</code>(**`cuda_id`**=*`None`*)

A context manager to adapt a learner to train in distributed data parallel mode.


### `distrib_ctx` context manager

**`distrib_ctx(cuda_id)`** prepares a learner to train in distributed data parallel mode.  It assumes these [environment variables](https://pytorch.org/tutorials/intermediate/dist_tuto.html#initialization-methods) have all been setup properly, such as those launched by `python -m fastai.launch`.

#### Typical usage:
```
with learn.distrib_ctx(): learn.fit(.....)
```

It attaches a [`DistributedTrainer`](/distributed.html#DistributedTrainer) callback and [`DistributedDL`](/distributed.html#DistributedDL) data loader to  the learner, then executes `learn.fit(.....)`.  Upon exiting the context, it removes the [`DistributedTrainer`](/distributed.html#DistributedTrainer) and [`DistributedDL`](/distributed.html#DistributedDL), and destroys any locally created distributed process group.  The process is still attached to the GPU though.



<h4 id="rank0_first" class="doc_header"><code>rank0_first</code><a href="https://github.com/fastai/fastai/tree/master/fastai/distributed.py#L171" class="source_link" style="float:right">[source]</a></h4>

> <code>rank0_first</code>(**`func`**)

Execute `func` in the Rank-0 process first, then in other ranks in parallel.


**`rank0_first(f)`** calls `f()` in rank-0 process first, then in parallel on the rest, in distributed training mode. In single process, non-distributed training mode, `f()` is called only once as expected.

One application of `rank0_first()` is to make fresh downloads via `untar_data()` safe in distributed training scripts launched by `python -m fastai.launch <script>`:
> <code>path = untar_data(URLs.IMDB)</code>

becomes:> <code>path = <b>rank0_first(lambda:</b> untar_data(URLs.IMDB))</code>

Some learner factory methods may use `untar_data()` to **download pretrained models** by default:
> <code>learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)</code>

becomes:> <code>learn = <b>rank0_first(lambda:</b> text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy))</code>
Otherwise, multiple processes will download at the same time and corrupt the data.

