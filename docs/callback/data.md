# Data Callbacks
> Callbacks which work with a learner's data



<h2 id="CollectDataCallback" class="doc_header"><code>class</code> <code>CollectDataCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/data.py#L9" class="source_link" style="float:right">[source]</a></h2>

> <code>CollectDataCallback</code>(**`before_fit`**=*`None`*, **`before_epoch`**=*`None`*, **`before_train`**=*`None`*, **`before_batch`**=*`None`*, **`after_pred`**=*`None`*, **`after_loss`**=*`None`*, **`before_backward`**=*`None`*, **`after_backward`**=*`None`*, **`after_step`**=*`None`*, **`after_cancel_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_cancel_train`**=*`None`*, **`after_train`**=*`None`*, **`before_validate`**=*`None`*, **`after_cancel_validate`**=*`None`*, **`after_validate`**=*`None`*, **`after_cancel_epoch`**=*`None`*, **`after_epoch`**=*`None`*, **`after_cancel_fit`**=*`None`*, **`after_fit`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

Collect all batches, along with `pred` and `loss`, into `self.data`. Mainly for testing



<h2 id="CudaCallback" class="doc_header"><code>class</code> <code>CudaCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/data.py#L15" class="source_link" style="float:right">[source]</a></h2>

> <code>CudaCallback</code>(**`device`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

Move data to CUDA device


You don't normally need to use this Callback, because fastai's [`DataLoader`](/data.load.html#DataLoader) will handle passing data to a device for you. However, if you already have a plain PyTorch DataLoader and can't change it for some reason, you can use this transform.

```python
learn = synth_learner(cbs=CudaCallback)
learn.model
learn.fit(1)
test_eq(next(learn.model.parameters()).device.type, 'cuda')
```

    (#4) [0,12.672515869140625,9.3746976852417,'00:00']



<h2 id="WeightedDL" class="doc_header"><code>class</code> <code>WeightedDL</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/data.py#L24" class="source_link" style="float:right">[source]</a></h2>

> <code>WeightedDL</code>(**`dataset`**=*`None`*, **`bs`**=*`None`*, **`wgts`**=*`None`*, **`shuffle`**=*`False`*, **`num_workers`**=*`None`*, **`verbose`**=*`False`*, **`do_setup`**=*`True`*, **`pin_memory`**=*`False`*, **`timeout`**=*`0`*, **`batch_size`**=*`None`*, **`drop_last`**=*`False`*, **`indexed`**=*`None`*, **`n`**=*`None`*, **`device`**=*`None`*, **`wif`**=*`None`*, **`before_iter`**=*`None`*, **`after_item`**=*`None`*, **`before_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_iter`**=*`None`*, **`create_batches`**=*`None`*, **`create_item`**=*`None`*, **`create_batch`**=*`None`*, **`retain`**=*`None`*, **`get_idxs`**=*`None`*, **`sample`**=*`None`*, **`shuffle_fn`**=*`None`*, **`do_batch`**=*`None`*) :: [`TfmdDL`](/data.core.html#TfmdDL)

Transformed [`DataLoader`](/data.load.html#DataLoader)



<h4 id="Datasets.weighted_dataloaders" class="doc_header"><code>Datasets.weighted_dataloaders</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/data.py#L36" class="source_link" style="float:right">[source]</a></h4>

> <code>Datasets.weighted_dataloaders</code>(**`wgts`**, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`n`**=*`None`*, **`path`**=*`'.'`*, **`dl_type`**=*`None`*, **`dl_kwargs`**=*`None`*, **`device`**=*`None`*)




```python
n = 160
dsets = Datasets(torch.arange(n).float())
dls = dsets.weighted_dataloaders(wgts=range(n), bs=16)
learn = synth_learner(data=dls, cbs=CollectDataCallback)
```

```python
learn.fit(1)
t = concat(*learn.collect_data.data.itemgot(0,0))
plt.hist(t);
```

    (#4) [0,nan,None,'00:00']



![png](output_13_1.png)



<h2 id="PartialDL" class="doc_header"><code>class</code> <code>PartialDL</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/data.py#L45" class="source_link" style="float:right">[source]</a></h2>

> <code>PartialDL</code>(**`dataset`**=*`None`*, **`bs`**=*`None`*, **`partial_n`**=*`None`*, **`shuffle`**=*`False`*, **`num_workers`**=*`None`*, **`verbose`**=*`False`*, **`do_setup`**=*`True`*, **`pin_memory`**=*`False`*, **`timeout`**=*`0`*, **`batch_size`**=*`None`*, **`drop_last`**=*`False`*, **`indexed`**=*`None`*, **`n`**=*`None`*, **`device`**=*`None`*, **`wif`**=*`None`*, **`before_iter`**=*`None`*, **`after_item`**=*`None`*, **`before_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_iter`**=*`None`*, **`create_batches`**=*`None`*, **`create_item`**=*`None`*, **`create_batch`**=*`None`*, **`retain`**=*`None`*, **`get_idxs`**=*`None`*, **`sample`**=*`None`*, **`shuffle_fn`**=*`None`*, **`do_batch`**=*`None`*) :: [`TfmdDL`](/data.core.html#TfmdDL)

Select randomly partial quantity of data at each epoch



<h4 id="FilteredBase.partial_dataloaders" class="doc_header"><code>FilteredBase.partial_dataloaders</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/data.py#L60" class="source_link" style="float:right">[source]</a></h4>

> <code>FilteredBase.partial_dataloaders</code>(**`partial_n`**, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`n`**=*`None`*, **`path`**=*`'.'`*, **`dl_type`**=*`None`*, **`dl_kwargs`**=*`None`*, **`device`**=*`None`*)

Create a partial dataloader [`PartialDL`](/callback.data.html#PartialDL) for the training set


```python
dls = dsets.partial_dataloaders(partial_n=32, bs=16)
```

```python
assert len(dls[0])==2
for batch in dls[0]:
    assert len(batch[0])==16
```
