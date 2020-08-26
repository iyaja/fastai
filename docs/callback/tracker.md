# Tracking callbacks
> Callbacks that make decisions depending how a monitored metric/loss behaves



<h2 id="TerminateOnNaNCallback" class="doc_header"><code>class</code> <code>TerminateOnNaNCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/tracker.py#L12" class="source_link" style="float:right">[source]</a></h2>

> <code>TerminateOnNaNCallback</code>(**`before_fit`**=*`None`*, **`before_epoch`**=*`None`*, **`before_train`**=*`None`*, **`before_batch`**=*`None`*, **`after_pred`**=*`None`*, **`after_loss`**=*`None`*, **`before_backward`**=*`None`*, **`after_backward`**=*`None`*, **`after_step`**=*`None`*, **`after_cancel_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_cancel_train`**=*`None`*, **`after_train`**=*`None`*, **`before_validate`**=*`None`*, **`after_cancel_validate`**=*`None`*, **`after_validate`**=*`None`*, **`after_cancel_epoch`**=*`None`*, **`after_epoch`**=*`None`*, **`after_cancel_fit`**=*`None`*, **`after_fit`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

A [`Callback`](/callback.core.html#Callback) that terminates training if loss is NaN.


```python
learn = synth_learner()
learn.fit(10, lr=100, cbs=TerminateOnNaNCallback())
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
      <td>733056513731005412711487968341131264.000000</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


```python
assert len(learn.recorder.losses) < 10 * len(learn.dls.train)
for l in learn.recorder.losses:
    assert not torch.isinf(l) and not torch.isnan(l) 
```


<h2 id="TrackerCallback" class="doc_header"><code>class</code> <code>TrackerCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/tracker.py#L21" class="source_link" style="float:right">[source]</a></h2>

> <code>TrackerCallback</code>(**`monitor`**=*`'valid_loss'`*, **`comp`**=*`None`*, **`min_delta`**=*`0.0`*) :: [`Callback`](/callback.core.html#Callback)

A [`Callback`](/callback.core.html#Callback) that keeps track of the best value in `monitor`.


When implementing a [`Callback`](/callback.core.html#Callback) that has behavior that depends on the best value of a metric or loss, subclass this [`Callback`](/callback.core.html#Callback) and use its `best` (for best value so far) and `new_best` (there was a new best value this epoch) attributes. 

`comp` is the comparison operator used to determine if a value is best than another (defaults to `np.less` if 'loss' is in the name passed in `monitor`, `np.greater` otherwise) and `min_delta` is an optional float that requires a new value to go over the current best (depending on `comp`) by at least that amount.


<h2 id="EarlyStoppingCallback" class="doc_header"><code>class</code> <code>EarlyStoppingCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/tracker.py#L47" class="source_link" style="float:right">[source]</a></h2>

> <code>EarlyStoppingCallback</code>(**`monitor`**=*`'valid_loss'`*, **`comp`**=*`None`*, **`min_delta`**=*`0.0`*, **`patience`**=*`1`*) :: [`TrackerCallback`](/callback.tracker.html#TrackerCallback)

A [`TrackerCallback`](/callback.tracker.html#TrackerCallback) that terminates training when monitored quantity stops improving.


`comp` is the comparison operator used to determine if a value is best than another (defaults to `np.less` if 'loss' is in the name passed in `monitor`, `np.greater` otherwise) and `min_delta` is an optional float that requires a new value to go over the current best (depending on `comp`) by at least that amount. `patience` is the number of epochs you're willing to wait without improvement.

```python
learn = synth_learner(n_trn=2, metrics=F.mse_loss)
learn.fit(n_epoch=200, lr=1e-7, cbs=EarlyStoppingCallback(monitor='mse_loss', min_delta=0.1, patience=2))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>mse_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>19.993200</td>
      <td>24.202908</td>
      <td>24.202908</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>20.007574</td>
      <td>24.202845</td>
      <td>24.202845</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>20.021687</td>
      <td>24.202751</td>
      <td>24.202751</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


    No improvement since epoch 0: early stopping


```python
learn.validate()
```








    (#2) [24.20275115966797,24.20275115966797]



```python
learn = synth_learner(n_trn=2)
learn.fit(n_epoch=200, lr=1e-7, cbs=EarlyStoppingCallback(monitor='valid_loss', min_delta=0.1, patience=2))
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
      <td>12.963860</td>
      <td>10.800257</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>12.936502</td>
      <td>10.800226</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>12.926699</td>
      <td>10.800186</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


    No improvement since epoch 0: early stopping



<h2 id="SaveModelCallback" class="doc_header"><code>class</code> <code>SaveModelCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/tracker.py#L66" class="source_link" style="float:right">[source]</a></h2>

> <code>SaveModelCallback</code>(**`monitor`**=*`'valid_loss'`*, **`comp`**=*`None`*, **`min_delta`**=*`0.0`*, **`fname`**=*`'model'`*, **`every_epoch`**=*`False`*, **`with_opt`**=*`False`*) :: [`TrackerCallback`](/callback.tracker.html#TrackerCallback)

A [`TrackerCallback`](/callback.tracker.html#TrackerCallback) that saves the model's best during training and loads it at the end.


`comp` is the comparison operator used to determine if a value is best than another (defaults to `np.less` if 'loss' is in the name passed in `monitor`, `np.greater` otherwise) and `min_delta` is an optional float that requires a new value to go over the current best (depending on `comp`) by at least that amount. Model will be saved in `learn.path/learn.model_dir/name.pth`, maybe `every_epoch` or at each improvement of the monitored quantity. 

```python
learn = synth_learner(n_trn=2, path=Path.cwd()/'tmp')
learn.fit(n_epoch=2, cbs=SaveModelCallback())
assert (Path.cwd()/'tmp/models/model.pth').exists()
learn.fit(n_epoch=2, cbs=SaveModelCallback(every_epoch=True))
for i in range(2): assert (Path.cwd()/f'tmp/models/model_{i}.pth').exists()
shutil.rmtree(Path.cwd()/'tmp')
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
      <td>22.050220</td>
      <td>19.015476</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>21.870991</td>
      <td>18.548334</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


    Better model found at epoch 0 with valid_loss value: 19.01547622680664.
    Better model found at epoch 1 with valid_loss value: 18.5483341217041.



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
      <td>21.132572</td>
      <td>17.913414</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>20.745022</td>
      <td>17.148394</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


## ReduceLROnPlateau


<h2 id="ReduceLROnPlateau" class="doc_header"><code>class</code> <code>ReduceLROnPlateau</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/tracker.py#L92" class="source_link" style="float:right">[source]</a></h2>

> <code>ReduceLROnPlateau</code>(**`monitor`**=*`'valid_loss'`*, **`comp`**=*`None`*, **`min_delta`**=*`0.0`*, **`patience`**=*`1`*, **`factor`**=*`10.0`*, **`min_lr`**=*`0`*) :: [`TrackerCallback`](/callback.tracker.html#TrackerCallback)

A [`TrackerCallback`](/callback.tracker.html#TrackerCallback) that reduces learning rate when a metric has stopped improving.


```python
learn = synth_learner(n_trn=2)
learn.fit(n_epoch=4, lr=1e-7, cbs=ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=2))
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
      <td>19.359529</td>
      <td>19.566990</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>19.375505</td>
      <td>19.566935</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>19.362509</td>
      <td>19.566856</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>19.386513</td>
      <td>19.566845</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


    Epoch 2: reducing lr to 1e-08


```python
learn = synth_learner(n_trn=2)
learn.fit(n_epoch=6, lr=5e-8, cbs=ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=2, min_lr=1e-8))
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
      <td>11.854585</td>
      <td>8.217508</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>11.875463</td>
      <td>8.217495</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11.885604</td>
      <td>8.217478</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>11.876034</td>
      <td>8.217473</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>11.872295</td>
      <td>8.217467</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>11.874965</td>
      <td>8.217461</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


    Epoch 2: reducing lr to 1e-08

