# Progress and logging callbacks
> Callback and helper function to track progress of training or log results


```python
from fastai.test_utils import *
```


<h2 id="ProgressCallback" class="doc_header"><code>class</code> <code>ProgressCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L10" class="source_link" style="float:right">[source]</a></h2>

> <code>ProgressCallback</code>(**`before_fit`**=*`None`*, **`before_epoch`**=*`None`*, **`before_train`**=*`None`*, **`before_batch`**=*`None`*, **`after_pred`**=*`None`*, **`after_loss`**=*`None`*, **`before_backward`**=*`None`*, **`after_backward`**=*`None`*, **`after_step`**=*`None`*, **`after_cancel_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_cancel_train`**=*`None`*, **`after_train`**=*`None`*, **`before_validate`**=*`None`*, **`after_cancel_validate`**=*`None`*, **`after_validate`**=*`None`*, **`after_cancel_epoch`**=*`None`*, **`after_epoch`**=*`None`*, **`after_cancel_fit`**=*`None`*, **`after_fit`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

A [`Callback`](/callback.core.html#Callback) to handle the display of progress bars


```python
learn = synth_learner()
learn.fit(5)
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
      <td>6.149214</td>
      <td>5.585020</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5.495753</td>
      <td>4.248405</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4.743536</td>
      <td>3.020540</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4.007711</td>
      <td>2.104298</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.349151</td>
      <td>1.430777</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<h4 id="Learner.no_bar" class="doc_header"><code>Learner.no_bar</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L59" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.no_bar</code>()

Context manager that deactivates the use of progress bars


```python
learn = synth_learner()
with learn.no_bar(): learn.fit(5)
```

    (#4) [0,30.605850219726562,27.92107391357422,'00:00']
    (#4) [1,26.819326400756836,19.888404846191406,'00:00']
    (#4) [2,22.556987762451172,13.134763717651367,'00:00']
    (#4) [3,18.57308578491211,8.311532020568848,'00:00']
    (#4) [4,15.115865707397461,5.124312400817871,'00:00']



<h4 id="ProgressCallback.before_fit" class="doc_header"><code>ProgressCallback.before_fit</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L14" class="source_link" style="float:right">[source]</a></h4>

> <code>ProgressCallback.before_fit</code>()

Setup the master bar over the epochs



<h4 id="ProgressCallback.before_epoch" class="doc_header"><code>ProgressCallback.before_epoch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L22" class="source_link" style="float:right">[source]</a></h4>

> <code>ProgressCallback.before_epoch</code>()

Update the master bar



<h4 id="ProgressCallback.before_train" class="doc_header"><code>ProgressCallback.before_train</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L25" class="source_link" style="float:right">[source]</a></h4>

> <code>ProgressCallback.before_train</code>()

Launch a progress bar over the training dataloader



<h4 id="ProgressCallback.before_validate" class="doc_header"><code>ProgressCallback.before_validate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L26" class="source_link" style="float:right">[source]</a></h4>

> <code>ProgressCallback.before_validate</code>()

Launch a progress bar over the validation dataloader



<h4 id="ProgressCallback.after_batch" class="doc_header"><code>ProgressCallback.after_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L29" class="source_link" style="float:right">[source]</a></h4>

> <code>ProgressCallback.after_batch</code>()

Update the current progress bar



<h4 id="ProgressCallback.after_train" class="doc_header"><code>ProgressCallback.after_train</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L27" class="source_link" style="float:right">[source]</a></h4>

> <code>ProgressCallback.after_train</code>()

Close the progress bar over the training dataloader



<h4 id="ProgressCallback.after_validate" class="doc_header"><code>ProgressCallback.after_validate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L28" class="source_link" style="float:right">[source]</a></h4>

> <code>ProgressCallback.after_validate</code>()

Close the progress bar over the validation dataloader



<h4 id="ProgressCallback.after_fit" class="doc_header"><code>ProgressCallback.after_fit</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L37" class="source_link" style="float:right">[source]</a></h4>

> <code>ProgressCallback.after_fit</code>()

Close the master bar



<h2 id="ShowGraphCallback" class="doc_header"><code>class</code> <code>ShowGraphCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L70" class="source_link" style="float:right">[source]</a></h2>

> <code>ShowGraphCallback</code>(**`before_fit`**=*`None`*, **`before_epoch`**=*`None`*, **`before_train`**=*`None`*, **`before_batch`**=*`None`*, **`after_pred`**=*`None`*, **`after_loss`**=*`None`*, **`before_backward`**=*`None`*, **`after_backward`**=*`None`*, **`after_step`**=*`None`*, **`after_cancel_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_cancel_train`**=*`None`*, **`after_train`**=*`None`*, **`before_validate`**=*`None`*, **`after_cancel_validate`**=*`None`*, **`after_validate`**=*`None`*, **`after_cancel_epoch`**=*`None`*, **`after_epoch`**=*`None`*, **`after_cancel_fit`**=*`None`*, **`after_fit`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

Update a graph of training and validation loss


```python
learn = synth_learner(cbs=ShowGraphCallback())
learn.fit(5)
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
      <td>17.077604</td>
      <td>15.280979</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>14.917645</td>
      <td>10.782686</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>12.502088</td>
      <td>6.969840</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>10.244695</td>
      <td>4.306195</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>8.296035</td>
      <td>2.599644</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6.680259</td>
      <td>1.545244</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>5.367113</td>
      <td>0.914276</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>4.311401</td>
      <td>0.539155</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3.466442</td>
      <td>0.317982</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2.791286</td>
      <td>0.188403</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



![png](output_19_1.png)



<h2 id="CSVLogger" class="doc_header"><code>class</code> <code>CSVLogger</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L91" class="source_link" style="float:right">[source]</a></h2>

> <code>CSVLogger</code>(**`fname`**=*`'history.csv'`*, **`append`**=*`False`*) :: [`Callback`](/callback.core.html#Callback)

Basic class handling tweaks of the training loop by changing a [`Learner`](/learner.html#Learner) in various events


The results are appended to an existing file if `append`, or they overwrite it otherwise.

```python
learn = synth_learner(cbs=CSVLogger())
learn.fit(5)
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
      <td>19.039587</td>
      <td>19.701471</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>16.735422</td>
      <td>14.259439</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>14.151361</td>
      <td>9.632797</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>11.720305</td>
      <td>6.176662</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>9.583822</td>
      <td>3.861699</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



<h4 id="CSVLogger.read_log" class="doc_header"><code>CSVLogger.read_log</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L97" class="source_link" style="float:right">[source]</a></h4>

> <code>CSVLogger.read_log</code>()

Convenience method to quickly access the log.


```python
df = learn.csv_logger.read_log()
test_eq(df.columns.values, learn.recorder.metric_names)
for i,v in enumerate(learn.recorder.values):
    test_close(df.iloc[i][:3], [i] + v)
os.remove(learn.path/learn.csv_logger.fname)
```


<h4 id="CSVLogger.before_fit" class="doc_header"><code>CSVLogger.before_fit</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L101" class="source_link" style="float:right">[source]</a></h4>

> <code>CSVLogger.before_fit</code>()

Prepare file with metric names.



<h4 id="CSVLogger.after_fit" class="doc_header"><code>CSVLogger.after_fit</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/progress.py#L113" class="source_link" style="float:right">[source]</a></h4>

> <code>CSVLogger.after_fit</code>()

Close the file and clean up.

