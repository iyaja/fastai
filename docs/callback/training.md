# Training callbacks
> Various callbacks to customize training behavior



<h2 id="ShortEpochCallback" class="doc_header"><code>class</code> <code>ShortEpochCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/training.py#L12" class="source_link" style="float:right">[source]</a></h2>

> <code>ShortEpochCallback</code>(**`pct`**=*`0.01`*, **`short_valid`**=*`True`*) :: [`Callback`](/callback.core.html#Callback)

Fit just `pct` of an epoch, then stop


```python
learn = synth_learner()
learn.fit(1, cbs=ShortEpochCallback())
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


```python
learn = synth_learner()
learn.fit(1, cbs=ShortEpochCallback(short_valid=False))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>12.395771</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<h2 id="GradientAccumulation" class="doc_header"><code>class</code> <code>GradientAccumulation</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/training.py#L22" class="source_link" style="float:right">[source]</a></h2>

> <code>GradientAccumulation</code>(**`n_acc`**=*`32`*) :: [`Callback`](/callback.core.html#Callback)

Accumulate gradients before updating weights


```python
learn = synth_learner()

learn.fit(2, lr=0.01, cbs=GradientAccumulation(n_acc=2*learn.dls.bs))
# ensure train_loss decreased
assert learn.recorder.values[-1][0] < learn.recorder.values[0][0]

learn.fit(2, lr=0.01, cbs=GradientAccumulation(n_acc=1e6))
# ensure valid_loss didn't change (same weights)
assert learn.recorder.values[-1][1] == learn.recorder.values[0][1]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10.566907</td>
      <td>3.633753</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5.525984</td>
      <td>0.397483</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.476599</td>
      <td>0.397483</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.478213</td>
      <td>0.397483</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


## BnFreeze


<h4 id="set_bn_eval" class="doc_header"><code>set_bn_eval</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/training.py#L40" class="source_link" style="float:right">[source]</a></h4>

> <code>set_bn_eval</code>(**`m`**:[`Module`](/torch_core.html#Module), **`use_eval`**=*`True`*)

Set bn layers in eval mode for all recursive children of `m`.



<h2 id="BnFreeze" class="doc_header"><code>class</code> <code>BnFreeze</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/training.py#L48" class="source_link" style="float:right">[source]</a></h2>

> <code>BnFreeze</code>(**`before_fit`**=*`None`*, **`before_epoch`**=*`None`*, **`before_train`**=*`None`*, **`before_batch`**=*`None`*, **`after_pred`**=*`None`*, **`after_loss`**=*`None`*, **`before_backward`**=*`None`*, **`after_backward`**=*`None`*, **`after_step`**=*`None`*, **`after_cancel_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_cancel_train`**=*`None`*, **`after_train`**=*`None`*, **`before_validate`**=*`None`*, **`after_cancel_validate`**=*`None`*, **`after_validate`**=*`None`*, **`after_cancel_epoch`**=*`None`*, **`after_epoch`**=*`None`*, **`after_cancel_fit`**=*`None`*, **`after_fit`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

Freeze moving average statistics in all non-trainable batchnorm layers.


[`BnFreeze`](/callback.training.html#BnFreeze) is useful when you'd like to train two separate models that have a common feature extractor / body. The only part of the model that's different is the head that you attach for transfer learning. <br>

[`Learner.freeze()`](/learner.html#Learner.freeze()) doesn't suffice here as the [`BatchNorm`](/layers.html#BatchNorm) layers are trainable by default, and running mean and std of batches are tracked. For feature extractors to fully match, you need to set `train_bn=False` and these stats need to be frozen as well, which is precisely the function of [`BnFreeze`](/callback.training.html#BnFreeze).

```python
from fastai.vision.all import *
```

```python
path = untar_data(URLs.MNIST_TINY)
dls  = ImageDataLoaders.from_folder(path, valid_pct=0.2)
```

We first demonstrate the mismatch of the running stats when using only `train_bn=False`, by creating a [`Learner`](/learner.html#Learner)...:

```python
learn1 = cnn_learner(deepcopy(dls), resnet18, pretrained=True, train_bn=False)
```

...and grab the first [`BatchNorm`](/layers.html#BatchNorm) layer, and store its running mean: 

```python
m = learn1.model[0][1].running_mean.clone()
```

You can see that now that running mean has changed:

```python
learn1.fit(1, lr=0.02)
test_ne(learn1.model[0][1].running_mean, m)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.058304</td>
      <td>0.713414</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>


When we use the [`BnFreeze`](/callback.training.html#BnFreeze) callback, the running statistics will not be changed during training. This is often important for getting good results from transfer learning.

```python
learn1 = cnn_learner(deepcopy(dls), resnet18, pretrained=True, train_bn=False, cbs=BnFreeze)
m = learn1.model[0][1].running_mean.clone()
learn1.fit(1, lr=0.02)
test_eq(learn1.model[0][1].running_mean, m)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.540841</td>
      <td>0.432421</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>

