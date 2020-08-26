# Callback
> Basic callbacks for Learner


## Events

Callbacks can occur at any of these times:: *before_fit before_epoch before_train before_batch after_pred after_loss before_backward after_backward after_step after_cancel_batch after_batch after_cancel_train after_train before_validate after_cancel_validate after_validate after_cancel_epoch after_epoch after_cancel_fit after_fit*.


<h3 id="event" class="doc_header"><code>class</code> <code>event</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>event</code>(**\*`args`**, **\*\*`kwargs`**)

All possible events as attributes to get tab-completion and typo-proofing


To ensure that you are referring to an event (that is, the name of one of the times when callbacks are called) that exists, and to get tab completion of event names, use `event`:

```python
test_eq(event.after_backward, 'after_backward')
```


<h2 id="Callback" class="doc_header"><code>class</code> <code>Callback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L31" class="source_link" style="float:right">[source]</a></h2>

> <code>Callback</code>(**`before_fit`**=*`None`*, **`before_epoch`**=*`None`*, **`before_train`**=*`None`*, **`before_batch`**=*`None`*, **`after_pred`**=*`None`*, **`after_loss`**=*`None`*, **`before_backward`**=*`None`*, **`after_backward`**=*`None`*, **`after_step`**=*`None`*, **`after_cancel_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_cancel_train`**=*`None`*, **`after_train`**=*`None`*, **`before_validate`**=*`None`*, **`after_cancel_validate`**=*`None`*, **`after_validate`**=*`None`*, **`after_cancel_epoch`**=*`None`*, **`after_epoch`**=*`None`*, **`after_cancel_fit`**=*`None`*, **`after_fit`**=*`None`*) :: [`GetAttr`](https://fastcore.fast.ai/foundation#GetAttr)

Basic class handling tweaks of the training loop by changing a [`Learner`](/learner.html#Learner) in various events


The training loop is defined in [`Learner`](/learner.html#Learner) a bit below and consists in a minimal set of instructions: looping through the data we:
- compute the output of the model from the input
- calculate a loss between this output and the desired target
- compute the gradients of this loss with respect to all the model parameters
- update the parameters accordingly
- zero all the gradients

Any tweak of this training loop is defined in a [`Callback`](/callback.core.html#Callback) to avoid over-complicating the code of the training loop, and to make it easy to mix and match different techniques (since they'll be defined in different callbacks). A callback can implement actions on the following events:

- `before_fit`: called before doing anything, ideal for initial setup.
- `before_epoch`: called at the beginning of each epoch, useful for any behavior you need to reset at each epoch.
- `before_train`: called at the beginning of the training part of an epoch.
- `before_batch`: called at the beginning of each batch, just after drawing said batch. It can be used to do any setup necessary for the batch (like hyper-parameter scheduling) or to change the input/target before it goes in the model (change of the input with techniques like mixup for instance).
- `after_pred`: called after computing the output of the model on the batch. It can be used to change that output before it's fed to the loss.
- `after_loss`: called after the loss has been computed, but before the backward pass. It can be used to add any penalty to the loss (AR or TAR in RNN training for instance).
- `before_backward`: called after the loss has been computed, but only in training mode (i.e. when the backward pass will be used)
- `after_backward`: called after the backward pass, but before the update of the parameters. It can be used to do any change to the gradients before said update (gradient clipping for instance).
- `after_step`: called after the step and before the gradients are zeroed.
- `after_batch`: called at the end of a batch, for any clean-up before the next one.
- `after_train`: called at the end of the training phase of an epoch.
- `before_validate`: called at the beginning of the validation phase of an epoch, useful for any setup needed specifically for validation.
- `after_validate`: called at the end of the validation part of an epoch.
- `after_epoch`: called at the end of an epoch, for any clean-up before the next one.
- `after_fit`: called at the end of training, for final clean-up.


<h4 id="Callback.__call__" class="doc_header"><code>Callback.__call__</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L39" class="source_link" style="float:right">[source]</a></h4>

> <code>Callback.__call__</code>(**`event_name`**)

Call `self.{event_name}` if it's defined


One way to define callbacks is through subclassing:

```python
class _T(Callback):
    def call_me(self): return "maybe"
test_eq(_T()("call_me"), "maybe")
```

Another way is by passing the callback function to the constructor:

```python
def cb(self): return "maybe"
_t = Callback(before_fit=cb)
test_eq(_t(event.before_fit), "maybe")
```


<h4 id="GetAttr.__getattr__" class="doc_header"><code>GetAttr.__getattr__</code><a href="https://github.com/fastai/fastcore/tree/master/fastcore/foundation.py#L237" class="source_link" style="float:right">[source]</a></h4>

> <code>GetAttr.__getattr__</code>(**`k`**)




This is a shortcut to avoid having to write `self.learn.bla` for any `bla` attribute we seek, and just write `self.bla`.

```python
mk_class('TstLearner', 'a')

class TstCallback(Callback):
    def batch_begin(self): print(self.a)

learn,cb = TstLearner(1),TstCallback()
cb.learn = learn
test_stdout(lambda: cb('batch_begin'), "1")
```

Note that it only works to get the value of the attribute, if you want to change it, you have to manually access it with `self.learn.bla`. In the example below, `self.a += 1` creates an `a` attribute of 2 in the callback instead of setting the `a` of the learner to 2. It also issues a warning that something is probably wrong:

```python
learn.a
```




    1



```python
class TstCallback(Callback):
    def batch_begin(self): self.a += 1

learn,cb = TstLearner(1),TstCallback()
cb.learn = learn
cb('batch_begin')
test_eq(cb.a, 2)
test_eq(cb.learn.a, 1)
```

    /home/jhoward/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:22: UserWarning: You are setting an attribute (a) that also exists in the learner, so you're not setting it in the learner but in the callback. Use `self.learn.a` otherwise.


A proper version needs to write `self.learn.a = self.a + 1`:

```python
class TstCallback(Callback):
    def batch_begin(self): self.learn.a = self.a + 1

learn,cb = TstLearner(1),TstCallback()
cb.learn = learn
cb('batch_begin')
test_eq(cb.learn.a, 2)
```


<h4 id="Callback.name" class="doc_header"><code>Callback.name</code><a href="" class="source_link" style="float:right">[source]</a></h4>

Name of the [`Callback`](/callback.core.html#Callback), camel-cased and with '*Callback*' removed


```python
test_eq(TstCallback().name, 'tst')
class ComplicatedNameCallback(Callback): pass
test_eq(ComplicatedNameCallback().name, 'complicated_name')
```


<h3 id="TrainEvalCallback" class="doc_header"><code>class</code> <code>TrainEvalCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L59" class="source_link" style="float:right">[source]</a></h3>

> <code>TrainEvalCallback</code>(**`before_fit`**=*`None`*, **`before_epoch`**=*`None`*, **`before_train`**=*`None`*, **`before_batch`**=*`None`*, **`after_pred`**=*`None`*, **`after_loss`**=*`None`*, **`before_backward`**=*`None`*, **`after_backward`**=*`None`*, **`after_step`**=*`None`*, **`after_cancel_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_cancel_train`**=*`None`*, **`after_train`**=*`None`*, **`before_validate`**=*`None`*, **`after_cancel_validate`**=*`None`*, **`after_validate`**=*`None`*, **`after_cancel_epoch`**=*`None`*, **`after_epoch`**=*`None`*, **`after_cancel_fit`**=*`None`*, **`after_fit`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

[`Callback`](/callback.core.html#Callback) that tracks the number of iterations done and properly sets training/eval mode


This [`Callback`](/callback.core.html#Callback) is automatically added in every [`Learner`](/learner.html#Learner) at initialization.


<h4 id="TrainEvalCallback.before_fit" class="doc_header"><code>TrainEvalCallback.before_fit</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L62" class="source_link" style="float:right">[source]</a></h4>

> <code>TrainEvalCallback.before_fit</code>()

Set the iter and epoch counters to 0, put the model and the right device



<h4 id="TrainEvalCallback.after_batch" class="doc_header"><code>TrainEvalCallback.after_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L68" class="source_link" style="float:right">[source]</a></h4>

> <code>TrainEvalCallback.after_batch</code>()

Update the iter counter (in training mode)



<h4 id="TrainEvalCallback.before_train" class="doc_header"><code>TrainEvalCallback.before_train</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L73" class="source_link" style="float:right">[source]</a></h4>

> <code>TrainEvalCallback.before_train</code>()

Set the model in training mode



<h3 id="GatherPredsCallback" class="doc_header"><code>class</code> <code>GatherPredsCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L89" class="source_link" style="float:right">[source]</a></h3>

> <code>GatherPredsCallback</code>(**`with_input`**=*`False`*, **`with_loss`**=*`False`*, **`save_preds`**=*`None`*, **`save_targs`**=*`None`*, **`concat_dim`**=*`0`*) :: [`Callback`](/callback.core.html#Callback)

[`Callback`](/callback.core.html#Callback) that saves the predictions and targets, optionally `with_loss`



<h4 id="GatherPredsCallback.before_validate" class="doc_header"><code>GatherPredsCallback.before_validate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L97" class="source_link" style="float:right">[source]</a></h4>

> <code>GatherPredsCallback.before_validate</code>()

Initialize containers



<h4 id="GatherPredsCallback.after_batch" class="doc_header"><code>GatherPredsCallback.after_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L103" class="source_link" style="float:right">[source]</a></h4>

> <code>GatherPredsCallback.after_batch</code>()

Save predictions, targets and potentially losses



<h4 id="GatherPredsCallback.after_validate" class="doc_header"><code>GatherPredsCallback.after_validate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L116" class="source_link" style="float:right">[source]</a></h4>

> <code>GatherPredsCallback.after_validate</code>()

Concatenate all recorded tensors



<h3 id="FetchPredsCallback" class="doc_header"><code>class</code> <code>FetchPredsCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/core.py#L131" class="source_link" style="float:right">[source]</a></h3>

> <code>FetchPredsCallback</code>(**`ds_idx`**=*`1`*, **`dl`**=*`None`*, **`with_input`**=*`False`*, **`with_decoded`**=*`False`*, **`cbs`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

A callback to fetch predictions during the training loop


When writing a callback, the following attributes of [`Learner`](/learner.html#Learner) are available:
- `model`: the model used for training/validation
- `data`: the underlying [`DataLoaders`](/data.core.html#DataLoaders)
- `loss_func`: the loss function used
- `opt`: the optimizer used to update the model parameters
- `opt_func`: the function used to create the optimizer
- `cbs`: the list containing all [`Callback`](/callback.core.html#Callback)s
- `dl`: current [`DataLoader`](/data.load.html#DataLoader) used for iteration
- `x`/`xb`: last input drawn from `self.dl` (potentially modified by callbacks). `xb` is always a tuple (potentially with one element) and `x` is detuplified. You can only assign to `xb`.
- `y`/`yb`: last target drawn from `self.dl` (potentially modified by callbacks). `yb` is always a tuple (potentially with one element) and `y` is detuplified. You can only assign to `yb`.
- `pred`: last predictions from `self.model` (potentially modified by callbacks)
- `loss`: last computed loss (potentially modified by callbacks)
- `n_epoch`: the number of epochs in this training
- `n_iter`: the number of iterations in the current `self.dl`
- `epoch`: the current epoch index (from 0 to `n_epoch-1`)
- `iter`: the current iteration index in `self.dl` (from 0 to `n_iter-1`)

The following attributes are added by [`TrainEvalCallback`](/callback.core.html#TrainEvalCallback) and should be available unless you went out of your way to remove that callback:

- `train_iter`: the number of training iterations done since the beginning of this training
- `pct_train`: from 0. to 1., the percentage of training iterations completed
- `training`:  flag to indicate if we're in training mode or not

The following attribute is added by [`Recorder`](/learner.html#Recorder) and should be available unless you went out of your way to remove that callback:

- `smooth_loss`: an exponentially-averaged version of the training loss

## Callbacks control flow

It happens that we may want to skip some of the steps of the training loop: in gradient accumulation, we don't always want to do the step/zeroing of the grads for instance. During an LR finder test, we don't want to do the validation phase of an epoch. Or if we're training with a strategy of early stopping, we want to be able to completely interrupt the training loop.

This is made possible by raising specific exceptions the training loop will look for (and properly catch).


<h3 id="CancelBatchException" class="doc_header"><code>class</code> <code>CancelBatchException</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>CancelBatchException</code>(**\*`args`**, **\*\*`kwargs`**) :: `Exception`

Interrupts training and go to `after_fit`



<h3 id="CancelTrainException" class="doc_header"><code>class</code> <code>CancelTrainException</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>CancelTrainException</code>(**\*`args`**, **\*\*`kwargs`**) :: `Exception`

Skip the rest of the validation part of the epoch and go to `after_validate`



<h3 id="CancelValidException" class="doc_header"><code>class</code> <code>CancelValidException</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>CancelValidException</code>(**\*`args`**, **\*\*`kwargs`**) :: `Exception`

Skip the rest of this epoch and go to `after_epoch`



<h3 id="CancelEpochException" class="doc_header"><code>class</code> <code>CancelEpochException</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>CancelEpochException</code>(**\*`args`**, **\*\*`kwargs`**) :: `Exception`

Skip the rest of the training part of the epoch and go to `after_train`



<h3 id="CancelFitException" class="doc_header"><code>class</code> <code>CancelFitException</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>CancelFitException</code>(**\*`args`**, **\*\*`kwargs`**) :: `Exception`

Skip the rest of this batch and go to `after_batch`


You can detect one of those exceptions occurred and add code that executes right after with the following events:
- `after_cancel_batch`: reached immediately after a `CancelBatchException` before proceeding to `after_batch`
- `after_cancel_train`: reached immediately after a `CancelTrainException` before proceeding to `after_epoch`
- `after_cancel_valid`: reached immediately after a `CancelValidException` before proceeding to `after_epoch`
- `after_cancel_epoch`: reached immediately after a `CancelEpochException` before proceeding to `after_epoch`
- `after_cancel_fit`: reached immediately after a `CancelFitException` before proceeding to `after_fit`
