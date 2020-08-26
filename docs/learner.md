# Learner
> Basic class for handling the training loop


You probably want to jump directly to the definition of [`Learner`](/learner.html#Learner).

## Utils function


<h4 id="replacing_yield" class="doc_header"><code>replacing_yield</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L29" class="source_link" style="float:right">[source]</a></h4>

> <code>replacing_yield</code>(**`o`**, **`attr`**, **`val`**)

Context manager to temporarily replace an attribute


```python
class _A:
    def __init__(self, a): self.a = a
    @contextmanager
    def a_changed(self, v): return replacing_yield(self, 'a', v)

a = _A(42)
with a.a_changed(32):
    test_eq(a.a, 32)
test_eq(a.a, 42)
```


<h4 id="mk_metric" class="doc_header"><code>mk_metric</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L36" class="source_link" style="float:right">[source]</a></h4>

> <code>mk_metric</code>(**`m`**)

Convert `m` to an [`AvgMetric`](/learner.html#AvgMetric), unless it's already a [`Metric`](/learner.html#Metric)


See the class [`Metric`](/learner.html#Metric) below for more information.


<h4 id="save_model" class="doc_header"><code>save_model</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L41" class="source_link" style="float:right">[source]</a></h4>

> <code>save_model</code>(**`file`**, **`model`**, **`opt`**, **`with_opt`**=*`True`*, **`pickle_protocol`**=*`2`*)

Save `model` to `file` along with `opt` (if available, and if `with_opt`)


`file` can be a `Path` object, a string or an opened file object. `pickle_protocol` is passed along to `torch.save`


<h4 id="load_model" class="doc_header"><code>load_model</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L50" class="source_link" style="float:right">[source]</a></h4>

> <code>load_model</code>(**`file`**, **`model`**, **`opt`**, **`with_opt`**=*`None`*, **`device`**=*`None`*, **`strict`**=*`True`*)

Load `model` from `file` along with `opt` (if available, and if `with_opt`)


`file` can be a `Path` object, a string or an opened file object. If a `device` is passed, the model is loaded on it, otherwise it's loaded on the CPU. 

If `strict` is `True`, the file must exactly contain weights for every parameter key in `model`, if `strict` is `False`, only the keys that are in the saved model are loaded in `model`.


<h2 id="Learner" class="doc_header"><code>class</code> <code>Learner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L82" class="source_link" style="float:right">[source]</a></h2>

> <code>Learner</code>(**`dls`**, **`model`**, **`loss_func`**=*`None`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`splitter`**=*`trainable_params`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*, **`moms`**=*`(0.95, 0.85, 0.95)`*)

Group together a `model`, some `dls` and a `loss_func` to handle training


`opt_func` will be used to create an optimizer when [`Learner.fit`](/learner.html#Learner.fit) is called, with `lr` as a default learning rate. `splitter` is a function that takes `self.model` and returns a list of parameter groups (or just one parameter group if there are no different parameter groups). The default is [`trainable_params`](/torch_core.html#trainable_params), which returns all trainable parameters of the model.

`cbs` is one or a list of [`Callback`](/callback.core.html#Callback)s  to pass to the [`Learner`](/learner.html#Learner). [`Callback`](/callback.core.html#Callback)s are used for every tweak of the training loop. Each [`Callback`](/callback.core.html#Callback) is registered as an attribute of [`Learner`](/learner.html#Learner) (with camel case). At creation, all the callbacks in [`defaults.callbacks`](https://fastcore.fast.ai/foundation#defaults.callbacks) ([`TrainEvalCallback`](/callback.core.html#TrainEvalCallback), [`Recorder`](/learner.html#Recorder) and [`ProgressCallback`](/callback.progress.html#ProgressCallback)) are associated to the [`Learner`](/learner.html#Learner).

[`metrics`](/metrics.html) is an optional list of metrics, that can be either functions or [`Metric`](/learner.html#Metric)s (see below). 

`path` and `model_dir` are used to save and/or load models. Often `path` will be inferred from `dls`, but you can override it or pass a `Path`  object to `model_dir`. Make sure you can write in `path/model_dir`!

`wd` is the default weight decay used when training the model; `moms`, the default momentums used in [`Learner.fit_one_cycle`](/callback.schedule.html#Learner.fit_one_cycle). `wd_bn_bias` controls if weight decay is applied to [`BatchNorm`](/layers.html#BatchNorm) layers and bias. 

Lastly, `train_bn` controls if [`BatchNorm`](/layers.html#BatchNorm) layers are trained even when they are supposed to be frozen according to the `splitter`. Our empirical experiments have shown that it's the best behavior for those layers in transfer learning.

### PyTorch interop

You can use regular PyTorch functionality for most of the arguments of the [`Learner`](/learner.html#Learner), although the experience will be smoother with pure fastai objects and you will be able to use the full functionality of the library. The expectation is that the training loop will work smoothly even if you did not use fastai end to end. What you might lose are interpretation objects or showing functionality. The list below explains how to use plain PyTorch objects for all the arguments and what you might lose.

The most important is `opt_func`. If you are not using a fastai optimizer, you will need to write a function that wraps your PyTorch optimizer in an [`OptimWrapper`](/optimizer.html#OptimWrapper). See the [optimizer module](http://docs.fast.ai/optimizer) for more details. This is to ensure the library's schedulers/freeze API work with your code.

- `dls` is a [`DataLoaders`](/data.core.html#DataLoaders) object, that you can create from standard PyTorch dataloaders. By doing so, you will lose all showing functionality like `show_batch`/`show_results`. You can check the [data block API](http://docs.fast.ai/tutorial.datablock) or the [mid-level data API tutorial](http://docs.fast.ai/tutorial.pets) to learn how to use fastai to gather your data!
- `model` is a standard PyTorch model. You can use anyone you like, just make sure it accepts the number of inputs you have in your [`DataLoaders`](/data.core.html#DataLoaders) and returns as many outputs as you have targets.
- `loss_func` can be any loss function you like. It needs to be one of fastai's if you want to use `Learn.predict` or `Learn.get_preds`, or you will have to implement special methods (see more details after the [`BaseLoss`](/layers.html#BaseLoss) documentation).

Now let's look at the main thing the [`Learner`](/learner.html#Learner) class implements: the training loop.

### Training loop


<h4 id="Learner.fit" class="doc_header"><code>Learner.fit</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L196" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.fit</code>(**`n_epoch`**, **`lr`**=*`None`*, **`wd`**=*`None`*, **`cbs`**=*`None`*, **`reset_opt`**=*`False`*)

Fit `self.model` for `n_epoch` using `cbs`. Optionally `reset_opt`.


Uses `lr` and `wd` if they are provided, otherwise use the defaults values given by the `lr` and `wd` attributes of [`Learner`](/learner.html#Learner).

All the examples use [`synth_learner`](/test_utils.html#synth_learner) which is a simple [`Learner`](/learner.html#Learner) training a linear regression model.

```python
learn = synth_learner(lr=5e-2)
learn.model = learn.model.cpu()
xb,yb = learn.dls.one_batch()
init_loss = learn.loss_func(learn.model(xb), yb)
learn.fit(6)
xb,yb = learn.dls.one_batch()
final_loss = learn.loss_func(learn.model(xb), yb)
assert final_loss < init_loss
```


<h4 id="Learner.one_batch" class="doc_header"><code>Learner.one_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L173" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.one_batch</code>(**`i`**, **`b`**)

Train or evaluate `self.model` on batch `(xb,yb)`


This is an internal method called by [`Learner.fit`](/learner.html#Learner.fit). If passed, `i` is the index of this iteration in the epoch. In training mode, this does a full training step on the batch (compute predictions, loss, gradients, update the model parameters and zero the gradients). In validation mode, it stops at the loss computation. Training or validation is controlled internally by the [`TrainEvalCallback`](/callback.core.html#TrainEvalCallback) through the `training` attribute.

Nothing is returned, but the attributes `x`, `y`, `pred`, `loss` of the [`Learner`](/learner.html#Learner) are set with the proper values:

```python
b = learn.dls.one_batch()
learn.one_batch(0, b)
test_eq(learn.x, b[0])
test_eq(learn.y, b[1])
out = learn.model(learn.x)
test_eq(learn.pred, out)
test_eq(learn.loss, learn.loss_func(out, b[1]))
```

More generally, the following attributes of [`Learner`](/learner.html#Learner) are available and updated during the training loop:
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


<h4 id="Learner.all_batches" class="doc_header"><code>Learner.all_batches</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L159" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.all_batches</code>()

Train or evaluate `self.model` on all the batches of `self.dl`



<h4 id="Learner.create_opt" class="doc_header"><code>Learner.create_opt</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L140" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.create_opt</code>()

Create an optimizer with default hyper-parameters


This method is called internally to create the optimizer, the hyper-parameters are then adjusted by what you pass to [`Learner.fit`](/learner.html#Learner.fit) or your particular schedulers (see [`callback.schedule`](/callback.schedule.html)).

```python
learn = synth_learner(n_train=5, cbs=VerboseCallback())
assert learn.opt is None
learn.create_opt()
assert learn.opt is not None
test_eq(learn.opt.hypers[0]['lr'], learn.lr)
```

### Serializing


<h4 id="Learner.save" class="doc_header"><code>Learner.save</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L277" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.save</code>(**`file`**, **`with_opt`**=*`True`*, **`pickle_protocol`**=*`2`*)

Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`


`file` can be a `Path`, a `string` or a buffer. `pickle_protocol` is passed along to `torch.save`.


<h4 id="Learner.load" class="doc_header"><code>Learner.load</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L283" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.load</code>(**`file`**, **`with_opt`**=*`None`*, **`device`**=*`None`*, **`strict`**=*`True`*)

Load model and optimizer state (if `with_opt`) from `self.path/self.model_dir/file` using `device`


`file` can be a `Path`, a `string` or a buffer. Use `device` to load the model/optimizer state on a device different from the one it was saved.

```python
with tempfile.TemporaryDirectory() as d:
    learn = synth_learner(path=d)
    learn.fit(1)
    
    #Test save created a file
    learn.save('tmp')
    assert (Path(d)/'models/tmp.pth').exists()
    
    #Test load did load the model
    learn1 = synth_learner(path=d)
    learn1 = learn1.load('tmp')
    test_eq(learn.model.a, learn1.model.a)
    test_eq(learn.model.b, learn1.model.b)
    test_eq(learn.opt.state_dict(), learn1.opt.state_dict())
```

### Callback handling

We only describe the basic functionality linked to [`Callback`](/callback.core.html#Callback)s here. To learn more about [`Callback`](/callback.core.html#Callback)s an how to write them, check the [callback.core](http://docs.fast.ai/callback.core) module documentation.

Let's first see how the [`Callback`](/callback.core.html#Callback)s become attributes of [`Learner`](/learner.html#Learner):

```python
class TstCallback(Callback):
    def batch_begin(self): self.learn.a = self.a + 1

tst_learn = synth_learner()
test_eq(len(tst_learn.cbs), 1)
assert isinstance(tst_learn.cbs[0], TrainEvalCallback)
assert hasattr(tst_learn, ('train_eval'))

tst_learn = synth_learner(cbs=TstCallback())
test_eq(len(tst_learn.cbs), 2)
assert isinstance(tst_learn.cbs[1], TstCallback)
assert hasattr(tst_learn, ('tst'))
```

A name that becomes an existing attribute of the [`Learner`](/learner.html#Learner) will throw an exception (here add_cb is a method of [`Learner`](/learner.html#Learner)).

```python
class AddCbCallback(Callback): pass
test_fail(lambda: synth_learner(cbs=AddCbCallback()))
```


<h4 id="Learner.__call__" class="doc_header"><code>Learner.__call__</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L133" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.__call__</code>(**`event_name`**)

Call `event_name` for all [`Callback`](/callback.core.html#Callback)s in `self.cbs`


This how the [`Callback`](/callback.core.html#Callback)s are called internally. For instance a [`VerboseCallback`](/test_utils.html#VerboseCallback) just prints the event names (can be useful for debugging):

```python
learn = synth_learner(cbs=VerboseCallback())
learn('after_fit')
```

    after_fit



<h4 id="Learner.add_cb" class="doc_header"><code>Learner.add_cb</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L104" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.add_cb</code>(**`cb`**)

Add `cb` to the list of [`Callback`](/callback.core.html#Callback) and register `self` as their learner


```python
learn = synth_learner()
learn.add_cb(TestTrainEvalCallback())
test_eq(len(learn.cbs), 2)
assert isinstance(learn.cbs[1], TestTrainEvalCallback)
test_eq(learn.train_eval.learn, learn)
```


<h4 id="Learner.add_cbs" class="doc_header"><code>Learner.add_cbs</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L102" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.add_cbs</code>(**`cbs`**)

Add `cbs` to the list of [`Callback`](/callback.core.html#Callback) and register `self` as their learner


```python
learn.add_cbs([TestTrainEvalCallback(), TestTrainEvalCallback()])
test_eq(len(learn.cbs), 4)
```


<h4 id="Learner.added_cbs" class="doc_header"><code>Learner.added_cbs</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L119" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.added_cbs</code>(**`cbs`**)

Context manage that temporarily adds `cbs`


```python
learn = synth_learner()
test_eq(len(learn.cbs), 1)
with learn.added_cbs(TestTrainEvalCallback()):
    test_eq(len(learn.cbs), 2)
```


<h4 id="Learner.ordered_cbs" class="doc_header"><code>Learner.ordered_cbs</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L131" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.ordered_cbs</code>(**`event`**)

List of [`Callback`](/callback.core.html#Callback)s, in order, for an `event` in the training loop


By order, we mean using the internal ordering of the [`Callback`](/callback.core.html#Callback)s (see [`callback.core`](/callback.core.html) for more information on how it works).

```python
learn = synth_learner()
learn.add_cb(TestTrainEvalCallback())
learn.ordered_cbs('before_fit')
```




    [TrainEvalCallback, TestTrainEvalCallback]




<h4 id="Learner.remove_cb" class="doc_header"><code>Learner.remove_cb</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L112" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.remove_cb</code>(**`cb`**)

Add `cb` from the list of [`Callback`](/callback.core.html#Callback) and deregister `self` as their learner


```python
learn = synth_learner()
learn.add_cb(TestTrainEvalCallback())
cb = learn.cbs[1]
learn.remove_cb(learn.cbs[1])
test_eq(len(learn.cbs), 1)
assert cb.learn is None
assert not getattr(learn,'test_train_eval',None)
```

`cb` can simply be the class of the [`Callback`](/callback.core.html#Callback) we want to remove (in which case all instances of that callback are removed).

```python
learn = synth_learner()
learn.add_cbs([TestTrainEvalCallback(), TestTrainEvalCallback()])
learn.remove_cb(TestTrainEvalCallback)
test_eq(len(learn.cbs), 1)
assert not getattr(learn,'test_train_eval',None)
```


<h4 id="Learner.remove_cbs" class="doc_header"><code>Learner.remove_cbs</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L103" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.remove_cbs</code>(**`cbs`**)

Remove `cbs` from the list of [`Callback`](/callback.core.html#Callback) and deregister `self` as their learner


Elements of `cbs` can either be types of callbacks or actual callbacks of the [`Learner`](/learner.html#Learner).

```python
learn = synth_learner()
learn.add_cbs([TestTrainEvalCallback() for _ in range(3)])
cb = learn.cbs[1]
learn.remove_cbs(learn.cbs[1:])
test_eq(len(learn.cbs), 1)
```


<h4 id="Learner.removed_cbs" class="doc_header"><code>Learner.removed_cbs</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L125" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.removed_cbs</code>(**`cbs`**)

Context manage that temporarily removes `cbs`


Elements of `cbs` can either be types of callbacks or actual callbacks of the [`Learner`](/learner.html#Learner).

```python
learn = synth_learner()
learn.add_cb(TestTrainEvalCallback())
with learn.removed_cbs(learn.cbs[1]):
    test_eq(len(learn.cbs), 1)
test_eq(len(learn.cbs), 2)
```


<h4 id="Learner.show_training_loop" class="doc_header"><code>Learner.show_training_loop</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L260" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.show_training_loop</code>()

Show each step in the training loop


At each step, callbacks are shown in order, which can help debugging.

```python
learn = synth_learner()
learn.show_training_loop()
```

    Start Fit
       - before_fit     : [TrainEvalCallback]
      Start Epoch Loop
         - before_epoch   : []
        Start Train
           - before_train   : [TrainEvalCallback]
          Start Batch Loop
             - before_batch   : []
             - after_pred     : []
             - after_loss     : []
             - before_backward: []
             - after_backward : []
             - after_step     : []
             - after_cancel_batch: []
             - after_batch    : [TrainEvalCallback]
          End Batch Loop
        End Train
         - after_cancel_train: []
         - after_train    : []
        Start Valid
           - before_validate: [TrainEvalCallback]
          Start Batch Loop
             - **CBs same as train batch**: []
          End Batch Loop
        End Valid
         - after_cancel_validate: []
         - after_validate : []
      End Epoch Loop
       - after_cancel_epoch: []
       - after_epoch    : []
    End Fit
     - after_cancel_fit: []
     - after_fit      : []



<h4 id="before_batch_cb" class="doc_header"><code>before_batch_cb</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L326" class="source_link" style="float:right">[source]</a></h4>

> <code>before_batch_cb</code>(**`f`**)

Shortcut for creating a Callback on the `before_batch` event, which takes and returns `xb,yb`


In order to change the data passed to your model, you will generally want to hook into the `before_batch` event, like so:

```python
class TstCallback(Callback):
    def before_batch(self):
        self.learn.xb = self.xb + 1000
        self.learn.yb = self.yb - 1000
```

Since that is so common, we provide the [`before_batch_cb`](/learner.html#before_batch_cb) decorator to make it easier.

```python
@before_batch_cb
def cb(self, xb, yb): return xb+1000,yb-1000
```


<h3 id="Metric" class="doc_header"><code>class</code> <code>Metric</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L332" class="source_link" style="float:right">[source]</a></h3>

> <code>Metric</code>()

Blueprint for defining a metric


Metrics can be simple averages (like accuracy) but sometimes their computation is a little bit more complex and can't be averaged over batches (like precision or recall), which is why we need a special class for them. For simple functions that can be computed as averages over batches, we can use the class [`AvgMetric`](/learner.html#AvgMetric), otherwise you'll need to implement the following methods.
{% include note.html content='If your <code>Metric</code> has state depending on tensors, don&#8217;t forget to store it on the CPU to avoid any potential memory leaks.' %}


<h4 id="Metric.reset" class="doc_header"><code>Metric.reset</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L334" class="source_link" style="float:right">[source]</a></h4>

> <code>Metric.reset</code>()

Reset inner state to prepare for new computation



<h4 id="Metric.accumulate" class="doc_header"><code>Metric.accumulate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L335" class="source_link" style="float:right">[source]</a></h4>

> <code>Metric.accumulate</code>(**`learn`**)

Use `learn` to update the state with new results



<h4 id="Metric.value" class="doc_header"><code>Metric.value</code><a href="" class="source_link" style="float:right">[source]</a></h4>

The value of the metric



<h4 id="Metric.name" class="doc_header"><code>Metric.name</code><a href="" class="source_link" style="float:right">[source]</a></h4>

Name of the [`Metric`](/learner.html#Metric), camel-cased and with Metric removed



<h3 id="AvgMetric" class="doc_header"><code>class</code> <code>AvgMetric</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L357" class="source_link" style="float:right">[source]</a></h3>

> <code>AvgMetric</code>(**`func`**) :: [`Metric`](/learner.html#Metric)

Average the values of `func` taking into account potential different batch sizes


```python
learn = synth_learner()
tst = AvgMetric(lambda x,y: (x-y).abs().mean())
t,u = torch.randn(100),torch.randn(100)
tst.reset()
for i in range(0,100,25): 
    learn.pred,learn.yb = t[i:i+25],(u[i:i+25],)
    tst.accumulate(learn)
test_close(tst.value, (t-u).abs().mean())
```


<h3 id="AvgLoss" class="doc_header"><code>class</code> <code>AvgLoss</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L371" class="source_link" style="float:right">[source]</a></h3>

> <code>AvgLoss</code>() :: [`Metric`](/learner.html#Metric)

Average the losses taking into account potential different batch sizes


```python
tst = AvgLoss()
t = torch.randn(100)
tst.reset()
for i in range(0,100,25): 
    learn.yb,learn.loss = t[i:i+25],t[i:i+25].mean()
    tst.accumulate(learn)
test_close(tst.value, t.mean())
```


<h3 id="AvgSmoothLoss" class="doc_header"><code>class</code> <code>AvgSmoothLoss</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L384" class="source_link" style="float:right">[source]</a></h3>

> <code>AvgSmoothLoss</code>(**`beta`**=*`0.98`*) :: [`Metric`](/learner.html#Metric)

Smooth average of the losses (exponentially weighted with `beta`)


```python
tst = AvgSmoothLoss()
t = torch.randn(100)
tst.reset()
val = tensor(0.)
for i in range(4): 
    learn.loss = t[i*25:(i+1)*25].mean()
    tst.accumulate(learn)
    val = val*0.98 + t[i*25:(i+1)*25].mean()*(1-0.98)
    test_close(val/(1-0.98**(i+1)), tst.value)
```


<h3 id="ValueMetric" class="doc_header"><code>class</code> <code>ValueMetric</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L395" class="source_link" style="float:right">[source]</a></h3>

> <code>ValueMetric</code>(**`func`**, **`metric_name`**=*`None`*) :: [`Metric`](/learner.html#Metric)

Use to include a pre-calculated metric value (for insance calculated in a [`Callback`](/callback.core.html#Callback)) and returned by `func`


```python
def metric_value_fn(): return 5e-3

vm = ValueMetric(metric_value_fn, 'custom_value_metric')
test_eq(vm.value, 5e-3)
test_eq(vm.name, 'custom_value_metric')

vm = ValueMetric(metric_value_fn)
test_eq(vm.name, 'metric_value_fn')
```


<h2 id="Recorder" class="doc_header"><code>class</code> <code>Recorder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L413" class="source_link" style="float:right">[source]</a></h2>

> <code>Recorder</code>(**`add_time`**=*`True`*, **`train_metrics`**=*`False`*, **`valid_metrics`**=*`True`*, **`beta`**=*`0.98`*) :: [`Callback`](/callback.core.html#Callback)

Callback that registers statistics (lr, loss and metrics) during training


By default, metrics are computed on the validation set only, although that can be changed by adjusting `train_metrics` and `valid_metrics`. `beta` is the weight used to compute the exponentially weighted average of the losses (which gives the `smooth_loss` attribute to [`Learner`](/learner.html#Learner)).

The `logger` attribute of a [`Learner`](/learner.html#Learner) determines what happens to those metrics. By default, it just print them:

```python
def tst_metric(out, targ): return F.mse_loss(out, targ)
learn = synth_learner(n_train=5, metrics=tst_metric)
pat = r"[tensor\(\d.\d*\), tensor\(\d.\d*\), tensor\(\d.\d*\), 'dd:dd']"
test_stdout(lambda: learn.fit(1), pat, regex=True)
```

### Internals


<h4 id="Recorder.before_fit" class="doc_header"><code>Recorder.before_fit</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L421" class="source_link" style="float:right">[source]</a></h4>

> <code>Recorder.before_fit</code>()

Prepare state for training



<h4 id="Recorder.before_epoch" class="doc_header"><code>Recorder.before_epoch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L444" class="source_link" style="float:right">[source]</a></h4>

> <code>Recorder.before_epoch</code>()

Set timer if `self.add_time=True`



<h4 id="Recorder.before_validate" class="doc_header"><code>Recorder.before_validate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L451" class="source_link" style="float:right">[source]</a></h4>

> <code>Recorder.before_validate</code>()

Reset loss and metrics state



<h4 id="Recorder.after_batch" class="doc_header"><code>Recorder.after_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L434" class="source_link" style="float:right">[source]</a></h4>

> <code>Recorder.after_batch</code>()

Update all metrics and records lr and smooth loss in training



<h4 id="Recorder.after_epoch" class="doc_header"><code>Recorder.after_epoch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L457" class="source_link" style="float:right">[source]</a></h4>

> <code>Recorder.after_epoch</code>()

Store and log the loss/metric values


### Plotting tools


<h4 id="Recorder.plot_loss" class="doc_header"><code>Recorder.plot_loss</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L475" class="source_link" style="float:right">[source]</a></h4>

> <code>Recorder.plot_loss</code>(**`skip_start`**=*`5`*, **`with_valid`**=*`True`*)

Plot the losses from `skip_start` and onward


## Inference functions


<h4 id="Learner.validate" class="doc_header"><code>Learner.validate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L216" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.validate</code>(**`ds_idx`**=*`1`*, **`dl`**=*`None`*, **`cbs`**=*`None`*)

Validate on `dl` with potential new `cbs`.


```python
learn = synth_learner(n_train=5, metrics=tst_metric)
res = learn.validate()
test_eq(res[0], res[1])
x,y = learn.dls.valid_ds.tensors
test_close(res[0], F.mse_loss(learn.model(x), y))
```


<h4 id="Learner.get_preds" class="doc_header"><code>Learner.get_preds</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L221" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.get_preds</code>(**`ds_idx`**=*`1`*, **`dl`**=*`None`*, **`with_input`**=*`False`*, **`with_decoded`**=*`False`*, **`with_loss`**=*`False`*, **`act`**=*`None`*, **`inner`**=*`False`*, **`reorder`**=*`True`*, **`cbs`**=*`None`*, **`save_preds`**=*`None`*, **`save_targs`**=*`None`*, **`concat_dim`**=*`0`*)

Get the predictions and targets on the `ds_idx`-th dbunchset or `dl`, optionally `with_input` and `with_loss`


`with_decoded` will also return the decoded predictions using the <code>decodes</code> function of the loss function (if it exists). For instance, fastai's `CrossEntropyFlat` takes the argmax or predictions in its decodes. 

Depending on the `loss_func` attribute of [`Learner`](/learner.html#Learner), an activation function will be picked automatically so that the predictions make sense. For instance if the loss is a case of cross-entropy, a softmax will be applied, or if the loss is binary cross entropy with logits, a sigmoid will be applied. If you want to make sure a certain activation function is applied, you can pass it with `act`.

`save_preds` and `save_targs` should be used when your predictions are too big to fit all in memory. Give a `Path` object that points to a folder where the predictions and targets will be saved.

`concat_dim` is the batch dimension, where all the tensors will be concatenated.

`inner` is an internal attribute that tells `get_preds` it's called internally, inside another training loop, to avoid recursion errors.

{% include note.html content='If you want to use the option `with_loss=True` on a custom loss function, make sure you have implemented a `reduction` attribute that supports &#8217;none&#8217; ' %}

```python
learn = synth_learner(n_train=5, metrics=tst_metric)
preds,targs = learn.get_preds()
x,y = learn.dls.valid_ds.tensors
test_eq(targs, y)
test_close(preds, learn.model(x))

preds,targs = learn.get_preds(act = torch.sigmoid)
test_eq(targs, y)
test_close(preds, torch.sigmoid(learn.model(x)))
```


<h4 id="Learner.predict" class="doc_header"><code>Learner.predict</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L243" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.predict</code>(**`item`**, **`rm_type_tfms`**=*`None`*, **`with_input`**=*`False`*)

Prediction on `item`, fully decoded, loss function decoded and probabilities


It returns a tuple of three elements with, in reverse order,
- the prediction from the model, potentially passed through the activation of the loss function (if it has one)
- the decoded prediction, using the potential <code>decodes</code> method from it
- the fully decoded prediction, using the transforms used to build the [`Datasets`](/data.core.html#Datasets)/[`DataLoaders`](/data.core.html#DataLoaders)

`rm_type_tfms` is a deprecated argument that should not be used and will be removed in a future version. `with_input` will add the decoded inputs to the result.

```python
class _FakeLossFunc(Module):
    reduction = 'none'
    def forward(self, x, y): return F.mse_loss(x,y)
    def activation(self, x): return x+1
    def decodes(self, x):    return 2*x

class _Add1(Transform):
    def encodes(self, x): return x+1
    def decodes(self, x): return x-1
    
learn = synth_learner(n_train=5)
dl = TfmdDL(Datasets(torch.arange(50), tfms = [L(), [_Add1()]]))
learn.dls = DataLoaders(dl, dl)
learn.loss_func = _FakeLossFunc()

inp = tensor([2.])
out = learn.model(inp).detach()+1  #applying model + activation
dec = 2*out                        #decodes from loss function
full_dec = dec-1                   #decodes from _Add1
test_eq(learn.predict(inp), [full_dec,dec,out])
test_eq(learn.predict(inp, with_input=True), [inp,full_dec,dec,out])
```


<h4 id="Learner.show_results" class="doc_header"><code>Learner.show_results</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L254" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.show_results</code>(**`ds_idx`**=*`1`*, **`dl`**=*`None`*, **`max_n`**=*`9`*, **`shuffle`**=*`True`*, **\*\*`kwargs`**)

Show some predictions on `ds_idx`-th dataset or `dl`


Will show `max_n` samples (unless the batch size of `ds_idx` or `dl` is less than `max_n`, in which case it will show as many samples) and `shuffle` the data unless you pass `false` to that flag. `kwargs` are application-dependant.

We can't show an example on our synthetic [`Learner`](/learner.html#Learner), but check all the beginners tutorials which will show you how that method works across applications.

The last functions in this section are used internally for inference, but should be less useful to you.


<h4 id="Learner.no_logging" class="doc_header"><code>Learner.no_logging</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L267" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.no_logging</code>()

Context manager to temporarily remove `logger`


```python
learn = synth_learner(n_train=5, metrics=tst_metric)
with learn.no_logging():
    test_stdout(lambda: learn.fit(1), '')
test_eq(learn.logger, print)
```


<h4 id="Learner.loss_not_reduced" class="doc_header"><code>Learner.loss_not_reduced</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L272" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.loss_not_reduced</code>()

A context manager to evaluate `loss_func` with reduction set to none.


This requires your loss function to either have a `reduction` attribute or a `reduction` argument (like all fastai and PyTorch loss functions).

## Transfer learning


<h4 id="Learner.freeze_to" class="doc_header"><code>Learner.freeze_to</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L496" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.freeze_to</code>(**`n`**)

Freeze parameter groups up to `n`



<h4 id="Learner.freeze" class="doc_header"><code>Learner.freeze</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L502" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.freeze</code>()

Freeze up to last parameter group



<h4 id="Learner.unfreeze" class="doc_header"><code>Learner.unfreeze</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L505" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.unfreeze</code>()

Unfreeze the entire model


### Exporting a [`Learner`](/learner.html#Learner)


<h4 id="Learner.export" class="doc_header"><code>Learner.export</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L514" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.export</code>(**`fname`**=*`'export.pkl'`*, **`pickle_protocol`**=*`2`*)

Export the content of `self` without the items and the optimizer state for inference


The [`Learner`](/learner.html#Learner) is saved in `self.path/fname`, using `pickle_protocol`. Note that serialization in Python saves the names of functions, not the code itself. Therefore, any custom code you have for models, data transformation, loss function etc... should be put in a module that you will import in your training environment before exporting, and in your deployment environment before loading it.


<h4 id="load_learner" class="doc_header"><code>load_learner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L532" class="source_link" style="float:right">[source]</a></h4>

> <code>load_learner</code>(**`fname`**, **`cpu`**=*`True`*)

Load a [`Learner`](/learner.html#Learner) object in `fname`, optionally putting it on the `cpu`


{% include warning.html content='[`load_learner`](/learner.html#load_learner) requires all your custom code be in the exact same place as when exporting your [`Learner`](/learner.html#Learner) (the main script, or the module you imported it from).' %}

## TTA


<h4 id="Learner.tta" class="doc_header"><code>Learner.tta</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L554" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.tta</code>(**`ds_idx`**=*`1`*, **`dl`**=*`None`*, **`n`**=*`4`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`beta`**=*`0.25`*, **`use_max`**=*`False`*)

Return predictions on the `ds_idx` dataset or `dl` using Test Time Augmentation


In practice, we get the predictions `n` times with the transforms of the training set and average those. The final predictions are `(1-beta)` multiplied by this average + `beta` multiplied by the predictions obtained with the transforms of the dataset. Set `beta` to `None` to get a tuple of the predictions and tta results. You can also use the maximum of all predictions instead of an average by setting `use_max=True`.

If you want to use new transforms, you can pass them with `item_tfms` and `batch_tfms`.

## Gather arguments


<h4 id="Learner.gather_args" class="doc_header"><code>Learner.gather_args</code><a href="https://github.com/fastai/fastai/tree/master/fastai/learner.py#L579" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.gather_args</code>()

Gather config parameters accessible to the learner


```python
learn = synth_learner(lr=1e-2)
test_eq(learn.init_args['Learner.__init__.lr'], 0.01)
```
