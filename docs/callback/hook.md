# Model hooks
> Callback and helper function to add hooks in models


```python
from fastai.test_utils import *
```

## What are hooks?

Hooks are functions you can attach to a particular layer in your model and that will be executed in the forward pass (for forward hooks) or backward pass (for backward hooks). Here we begin with an introduction around hooks, but you should jump to [`HookCallback`](/callback.hook.html#HookCallback) if you quickly want to implement one (and read the following example [`ActivationStats`](/callback.hook.html#ActivationStats)).

Forward hooks are functions that take three arguments: the layer it's applied to, the input of that layer and the output of that layer.

```python
tst_model = nn.Linear(5,3)
def example_forward_hook(m,i,o): print(m,i,o)
    
x = torch.randn(4,5)
hook = tst_model.register_forward_hook(example_forward_hook)
y = tst_model(x)
hook.remove()
```

    Linear(in_features=5, out_features=3, bias=True) (tensor([[ 0.1800,  0.3520, -0.4931, -1.7999,  1.4483],
            [-0.7465, -1.1864, -0.8031,  0.0359,  1.4905],
            [ 1.4436, -0.0306,  0.5153, -0.3651,  1.2812],
            [-0.2445,  0.2442,  1.6167, -0.0388,  0.4127]]),) tensor([[-0.7753, -1.3809,  0.2013],
            [-0.2287, -0.5858,  0.0458],
            [-0.6677, -1.3756,  0.1527],
            [ 0.3419, -0.2536,  0.4552]], grad_fn=<AddmmBackward>)


Backward hooks are functions that take three arguments: the layer it's applied to, the gradients of the loss with respect to the input, and the gradients with respect to the output.

```python
def example_backward_hook(m,gi,go): print(m,gi,go)
hook = tst_model.register_backward_hook(example_backward_hook)

x = torch.randn(4,5)
y = tst_model(x)
loss = y.pow(2).mean()
loss.backward()
hook.remove()
```

    Linear(in_features=5, out_features=3, bias=True) (tensor([-0.0540, -0.3042, -0.0928]), None, tensor([[-0.2777, -0.6483, -0.5103],
            [ 0.1898,  0.3238,  0.3187],
            [ 0.1839,  0.2838,  0.3302],
            [-0.0485, -0.0478, -0.1163],
            [-0.0107, -0.0062,  0.0186]])) (tensor([[-0.1005, -0.2200, -0.1706],
            [ 0.0101, -0.0002, -0.0116],
            [ 0.0199,  0.0020,  0.0450],
            [ 0.0165, -0.0860,  0.0444]]),)


Hooks can change the input/output of a layer, or the gradients, print values or shapes. If you want to store something related to theses inputs/outputs, it's best to have your hook associated to a class so that it can put it in the state of an instance of that class.


<h2 id="Hook" class="doc_header"><code>class</code> <code>Hook</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L11" class="source_link" style="float:right">[source]</a></h2>

> <code>Hook</code>(**`m`**, **`hook_func`**, **`is_forward`**=*`True`*, **`detach`**=*`True`*, **`cpu`**=*`False`*, **`gather`**=*`False`*)

Create a hook on `m` with `hook_func`.


This will be called during the forward pass if `is_forward=True`, the backward pass otherwise, and will optionally `detach`, `gather` and put on the `cpu` the (gradient of the) input/output of the model before passing them to `hook_func`. The result of `hook_func` will be stored in the `stored` attribute of the [`Hook`](/callback.hook.html#Hook).

```python
tst_model = nn.Linear(5,3)
hook = Hook(tst_model, lambda m,i,o: o)
y = tst_model(x)
test_eq(hook.stored, y)
```


<h4 id="Hook.hook_fn" class="doc_header"><code>Hook.hook_fn</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L19" class="source_link" style="float:right">[source]</a></h4>

> <code>Hook.hook_fn</code>(**`module`**, **`input`**, **`output`**)

Applies `hook_func` to `module`, `input`, `output`.



<h4 id="Hook.remove" class="doc_header"><code>Hook.remove</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L25" class="source_link" style="float:right">[source]</a></h4>

> <code>Hook.remove</code>()

Remove the hook from the model.


{% include note.html content='It&#8217;s important to properly remove your hooks for your model when you&#8217;re done to avoid them being called again next time your model is applied to some inputs, and to free the memory that go with their state.' %}

```python
tst_model = nn.Linear(5,10)
x = torch.randn(4,5)
y = tst_model(x)
hook = Hook(tst_model, example_forward_hook)
test_stdout(lambda: tst_model(x), f"{tst_model} ({x},) {y.detach()}")
hook.remove()
test_stdout(lambda: tst_model(x), "")
```

### Context Manager

Since it's very important to remove your [`Hook`](/callback.hook.html#Hook) even if your code is interrupted by some bug, [`Hook`](/callback.hook.html#Hook) can be used as context managers.


<h4 id="Hook.__enter__" class="doc_header"><code>Hook.__enter__</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L31" class="source_link" style="float:right">[source]</a></h4>

> <code>Hook.__enter__</code>(**\*`args`**)

Register the hook



<h4 id="Hook.__exit__" class="doc_header"><code>Hook.__exit__</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L32" class="source_link" style="float:right">[source]</a></h4>

> <code>Hook.__exit__</code>(**\*`args`**)

Remove the hook


```python
tst_model = nn.Linear(5,10)
x = torch.randn(4,5)
y = tst_model(x)
with Hook(tst_model, example_forward_hook) as h:
    test_stdout(lambda: tst_model(x), f"{tst_model} ({x},) {y.detach()}")
test_stdout(lambda: tst_model(x), "")
```


<h4 id="hook_output" class="doc_header"><code>hook_output</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L40" class="source_link" style="float:right">[source]</a></h4>

> <code>hook_output</code>(**`module`**, **`detach`**=*`True`*, **`cpu`**=*`False`*, **`grad`**=*`False`*)

Return a [`Hook`](/callback.hook.html#Hook) that stores activations of `module` in `self.stored`


The activations stored are the gradients if `grad=True`, otherwise the output of [`module`](/layers.html#module). If `detach=True` they are detached from their history, and if `cpu=True`, they're put on the CPU.

```python
tst_model = nn.Linear(5,10)
x = torch.randn(4,5)
with hook_output(tst_model) as h:
    y = tst_model(x)
    test_eq(y, h.stored)
    assert not h.stored.requires_grad
    
with hook_output(tst_model, grad=True) as h:
    y = tst_model(x)
    loss = y.pow(2).mean()
    loss.backward()
    test_close(2*y / y.numel(), h.stored[0])
```

```python
with hook_output(tst_model, cpu=True) as h:
    y = tst_model.cuda()(x.cuda())
    test_eq(h.stored.device, torch.device('cpu'))
```


<h2 id="Hooks" class="doc_header"><code>class</code> <code>Hooks</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L46" class="source_link" style="float:right">[source]</a></h2>

> <code>Hooks</code>(**`ms`**, **`hook_func`**, **`is_forward`**=*`True`*, **`detach`**=*`True`*, **`cpu`**=*`False`*)

Create several hooks on the modules in `ms` with `hook_func`.


```python
layers = [nn.Linear(5,10), nn.ReLU(), nn.Linear(10,3)]
tst_model = nn.Sequential(*layers)
hooks = Hooks(tst_model, lambda m,i,o: o)
y = tst_model(x)
test_eq(hooks.stored[0], layers[0](x))
test_eq(hooks.stored[1], F.relu(layers[0](x)))
test_eq(hooks.stored[2], y)
hooks.remove()
```


<h4 id="Hooks.stored" class="doc_header"><code>Hooks.stored</code><a href="" class="source_link" style="float:right">[source]</a></h4>

The states saved in each hook.



<h4 id="Hooks.remove" class="doc_header"><code>Hooks.remove</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L57" class="source_link" style="float:right">[source]</a></h4>

> <code>Hooks.remove</code>()

Remove the hooks from the model.


### Context Manager

Like [`Hook`](/callback.hook.html#Hook) , you can use [`Hooks`](/callback.hook.html#Hooks) as context managers.


<h4 id="Hooks.__enter__" class="doc_header"><code>Hooks.__enter__</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L61" class="source_link" style="float:right">[source]</a></h4>

> <code>Hooks.__enter__</code>(**\*`args`**)

Register the hooks



<h4 id="Hooks.__exit__" class="doc_header"><code>Hooks.__exit__</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L62" class="source_link" style="float:right">[source]</a></h4>

> <code>Hooks.__exit__</code>(**\*`args`**)

Remove the hooks


```python
layers = [nn.Linear(5,10), nn.ReLU(), nn.Linear(10,3)]
tst_model = nn.Sequential(*layers)
with Hooks(layers, lambda m,i,o: o) as h:
    y = tst_model(x)
    test_eq(h.stored[0], layers[0](x))
    test_eq(h.stored[1], F.relu(layers[0](x)))
    test_eq(h.stored[2], y)
```


<h4 id="hook_outputs" class="doc_header"><code>hook_outputs</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L69" class="source_link" style="float:right">[source]</a></h4>

> <code>hook_outputs</code>(**`modules`**, **`detach`**=*`True`*, **`cpu`**=*`False`*, **`grad`**=*`False`*)

Return [`Hooks`](/callback.hook.html#Hooks) that store activations of all `modules` in `self.stored`


The activations stored are the gradients if `grad=True`, otherwise the output of `modules`. If `detach=True` they are detached from their history, and if `cpu=True`, they're put on the CPU.

```python
layers = [nn.Linear(5,10), nn.ReLU(), nn.Linear(10,3)]
tst_model = nn.Sequential(*layers)
x = torch.randn(4,5)
with hook_outputs(layers) as h:
    y = tst_model(x)
    test_eq(h.stored[0], layers[0](x))
    test_eq(h.stored[1], F.relu(layers[0](x)))
    test_eq(h.stored[2], y)
    for s in h.stored: assert not s.requires_grad
    
with hook_outputs(layers, grad=True) as h:
    y = tst_model(x)
    loss = y.pow(2).mean()
    loss.backward()
    g = 2*y / y.numel()
    test_close(g, h.stored[2][0])
    g = g @ layers[2].weight.data
    test_close(g, h.stored[1][0])
    g = g * (layers[0](x) > 0).float()
    test_close(g, h.stored[0][0])
```

```python
with hook_outputs(tst_model, cpu=True) as h:
    y = tst_model.cuda()(x.cuda())
    for s in h.stored: test_eq(s.device, torch.device('cpu'))
```


<h4 id="dummy_eval" class="doc_header"><code>dummy_eval</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L74" class="source_link" style="float:right">[source]</a></h4>

> <code>dummy_eval</code>(**`m`**, **`size`**=*`(64, 64)`*)

Evaluate `m` on a dummy input of a certain `size`



<h4 id="model_sizes" class="doc_header"><code>model_sizes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L81" class="source_link" style="float:right">[source]</a></h4>

> <code>model_sizes</code>(**`m`**, **`size`**=*`(64, 64)`*)

Pass a dummy input through the model `m` to get the various sizes of activations.


```python
m = nn.Sequential(ConvLayer(3, 16), ConvLayer(16, 32, stride=2), ConvLayer(32, 32))
test_eq(model_sizes(m), [[1, 16, 64, 64], [1, 32, 32, 32], [1, 32, 32, 32]])
```


<h4 id="num_features_model" class="doc_header"><code>num_features_model</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L88" class="source_link" style="float:right">[source]</a></h4>

> <code>num_features_model</code>(**`m`**)

Return the number of output features for `m`.


```python
m = nn.Sequential(nn.Conv2d(5,4,3), nn.Conv2d(4,3,3))
test_eq(num_features_model(m), 3)
m = nn.Sequential(ConvLayer(3, 16), ConvLayer(16, 32, stride=2), ConvLayer(32, 32))
test_eq(num_features_model(m), 32)
```

To make hooks easy to use, we wrapped a version in a Callback where you just have to implement a `hook` function (plus any element you might need).


<h4 id="has_params" class="doc_header"><code>has_params</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L100" class="source_link" style="float:right">[source]</a></h4>

> <code>has_params</code>(**`m`**)

Check if `m` has at least one parameter


```python
assert has_params(nn.Linear(3,4))
assert has_params(nn.LSTM(4,5,2))
assert not has_params(nn.ReLU())
```


<h2 id="HookCallback" class="doc_header"><code>class</code> <code>HookCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L106" class="source_link" style="float:right">[source]</a></h2>

> <code>HookCallback</code>(**`modules`**=*`None`*, **`every`**=*`None`*, **`remove_end`**=*`True`*, **`is_forward`**=*`True`*, **`detach`**=*`True`*, **`cpu`**=*`True`*, **`hook`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

[`Callback`](/callback.core.html#Callback) that can be used to register hooks on `modules`


You can either subclass and implement a `hook` function (along with any event you want) or pass that a `hook` function when initializing. Such a function needs to take three argument: a layer, input and output (for a backward hook, input means gradient with respect to the inputs, output, gradient with respect to the output) and can either modify them or update the state according to them.

If not provided, `modules` will default to the layers of `self.model` that have a `weight` attribute. Depending on `do_remove`, the hooks will be properly removed at the end of training (or in case of error). `is_forward` , `detach` and `cpu` are passed to [`Hooks`](/callback.hook.html#Hooks).

The function called at each forward (or backward) pass is `self.hook` and must be implemented when subclassing this callback.

```python
class TstCallback(HookCallback):
    def hook(self, m, i, o): return o
    def after_batch(self): test_eq(self.hooks.stored[0], self.pred)
        
learn = synth_learner(n_trn=5, cbs = TstCallback())
learn.fit(1)
```

    (#4) [0,14.337016105651855,9.152174949645996,'00:00']


```python
class TstCallback(HookCallback):
    def __init__(self, modules=None, remove_end=True, detach=True, cpu=False):
        super().__init__(modules, None, remove_end, False, detach, cpu)
    def hook(self, m, i, o): return o
    def after_batch(self):
        if self.training:
            test_eq(self.hooks.stored[0][0], 2*(self.pred-self.y)/self.pred.shape[0])
        
learn = synth_learner(n_trn=5, cbs = TstCallback())
learn.fit(1)
```

    (#4) [0,28.941675186157227,30.372051239013672,'00:00']



<h4 id="HookCallback.before_fit" class="doc_header"><code>HookCallback.before_fit</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L114" class="source_link" style="float:right">[source]</a></h4>

> <code>HookCallback.before_fit</code>()

Register the [`Hooks`](/callback.hook.html#Hooks) on `self.modules`.



<h4 id="HookCallback.after_fit" class="doc_header"><code>HookCallback.after_fit</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L127" class="source_link" style="float:right">[source]</a></h4>

> <code>HookCallback.after_fit</code>()

Remove the [`Hooks`](/callback.hook.html#Hooks).


## Model summary


<h4 id="total_params" class="doc_header"><code>total_params</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L138" class="source_link" style="float:right">[source]</a></h4>

> <code>total_params</code>(**`m`**)

Give the number of parameters of a module and if it's trainable or not


```python
test_eq(total_params(nn.Linear(10,32)), (32*10+32,True))
test_eq(total_params(nn.Linear(10,32, bias=False)), (32*10,True))
test_eq(total_params(nn.BatchNorm2d(20)), (20*2, True))
test_eq(total_params(nn.BatchNorm2d(20, affine=False)), (0,False))
test_eq(total_params(nn.Conv2d(16, 32, 3)), (16*32*3*3 + 32, True))
test_eq(total_params(nn.Conv2d(16, 32, 3, bias=False)), (16*32*3*3, True))
#First ih layer 20--10, all else 10--10. *4 for the four gates
test_eq(total_params(nn.LSTM(20, 10, 2)), (4 * (20*10 + 10) + 3 * 4 * (10*10 + 10), True))
```


<h4 id="layer_info" class="doc_header"><code>layer_info</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L145" class="source_link" style="float:right">[source]</a></h4>

> <code>layer_info</code>(**`learn`**, **\*`xb`**)

Return layer infos of `model` on `xb` (only support batch first inputs)


```python
def _m(): return nn.Sequential(nn.Linear(1,50), nn.ReLU(), nn.BatchNorm1d(50), nn.Linear(50, 1))
sample_input = torch.randn((16, 1))
test_eq(layer_info(synth_learner(model=_m()), sample_input), [
    ('Linear', 100, True, [1, 50]),
    ('ReLU', 0, False, [1, 50]),
    ('BatchNorm1d', 100, True, [1, 50]),
    ('Linear', 51, True, [1, 1])
])
```

    (#4) [0,None,'00:00','00:00']



<h4 id="module_summary" class="doc_header"><code>module_summary</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L159" class="source_link" style="float:right">[source]</a></h4>

> <code>module_summary</code>(**`learn`**, **\*`xb`**)

Print a summary of `model` using `xb`



<h4 id="Learner.summary" class="doc_header"><code>Learner.summary</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L185" class="source_link" style="float:right">[source]</a></h4>

> <code>Learner.summary</code>()

Print a summary of the model, optimizer and loss function.


```python
learn = synth_learner(model=_m())
learn.summary()
```

    (#4) [0,None,'00:00','00:00']





    Sequential (Input shape: ['16 x 1'])
    ================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ================================================================
    Linear               16 x 50              100        True      
    ________________________________________________________________
    ReLU                 16 x 50              0          False     
    ________________________________________________________________
    BatchNorm1d          16 x 50              100        True      
    ________________________________________________________________
    Linear               16 x 1               51         True      
    ________________________________________________________________
    
    Total params: 251
    Total trainable params: 251
    Total non-trainable params: 0
    
    Optimizer used: functools.partial(<function SGD at 0x7fb8d1340440>, mom=0.9)
    Loss function: FlattenedLoss of MSELoss()
    
    Callbacks:
      - TrainEvalCallback
      - Recorder



## Activation graphs

This is an example of a [`HookCallback`](/callback.hook.html#HookCallback), that stores the mean, stds and histograms of activations that go through the network.

```python
@delegates()
class ActivationStats(HookCallback):
    "Callback that record the mean and std of activations."
    run_before=TrainEvalCallback
    def __init__(self, with_hist=False, **kwargs):
        super().__init__(**kwargs)
        self.with_hist = with_hist

    def before_fit(self):
        "Initialize stats."
        super().before_fit()
        self.stats = L()

    def hook(self, m, i, o):
        o = o.float()
        res = {'mean': o.mean().item(), 'std': o.std().item(),
               'near_zero': (o<=0.05).long().sum().item()/o.numel()}
        if self.with_hist: res['hist'] = o.histc(40,0,10)
        return res

    def after_batch(self):
        "Take the stored results and puts it in `self.stats`"
        if self.training and (self.every is None or self.train_iter%self.every == 0):
            self.stats.append(self.hooks.stored)
        super().after_batch()

    def layer_stats(self, idx):
        lstats = self.stats.itemgot(idx)
        return L(lstats.itemgot(o) for o in ('mean','std','near_zero'))

    def hist(self, idx):
        res = self.stats.itemgot(idx).itemgot('hist')
        return torch.stack(tuple(res)).t().float().log1p()

    def color_dim(self, idx, figsize=(10,5), ax=None):
        "The 'colorful dimension' plot"
        res = self.hist(idx)
        if ax is None: ax = subplots(figsize=figsize)[1][0]
        ax.imshow(res, origin='lower')
        ax.axis('off')

    def plot_layer_stats(self, idx):
        _,axs = subplots(1, 3, figsize=(12,3))
        for o,ax,title in zip(self.layer_stats(idx),axs,('mean','std','% near zero')):
            ax.plot(o)
            ax.set_title(title)
```


<h2 id="ActivationStats" class="doc_header"><code>class</code> <code>ActivationStats</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/hook.py#L198" class="source_link" style="float:right">[source]</a></h2>

> <code>ActivationStats</code>(**`with_hist`**=*`False`*, **`modules`**=*`None`*, **`every`**=*`None`*, **`remove_end`**=*`True`*, **`is_forward`**=*`True`*, **`detach`**=*`True`*, **`cpu`**=*`True`*, **`hook`**=*`None`*) :: [`HookCallback`](/callback.hook.html#HookCallback)

Callback that record the mean and std of activations.


```python
learn = synth_learner(n_trn=5, cbs = ActivationStats(every=4))
learn.fit(1)
```

    (#4) [0,21.134803771972656,21.965744018554688,'00:00']


```python
learn.activation_stats.stats
```




    (#2) [(#1) [{'mean': -0.8755316734313965, 'std': 0.7497552037239075, 'near_zero': 0.875}],(#1) [{'mean': -0.8900261521339417, 'std': 0.8982859253883362, 'near_zero': 0.875}]]



The first line contains the means of the outputs of the model for each batch in the training set, the second line their standard deviations.

```python
import math
```

```python
def test_every(n_tr, every):
    "create a learner, fit, then check number of stats collected"
    learn = synth_learner(n_trn=n_tr, cbs=ActivationStats(every=every))
    learn.fit(1)
    expected_stats_len = math.ceil(n_tr / every)
    test_eq(expected_stats_len, len(learn.activation_stats.stats))
    
for n_tr in [11, 12, 13]:
    test_every(n_tr, 4)
    test_every(n_tr, 1)
```

    (#4) [0,4.477747440338135,3.7352094650268555,'00:00']
    (#4) [0,10.678933143615723,9.956993103027344,'00:00']
    (#4) [0,13.817577362060547,11.273590087890625,'00:00']
    (#4) [0,6.385880470275879,6.805768966674805,'00:00']
    (#4) [0,4.044031620025635,3.037524700164795,'00:00']
    (#4) [0,15.187829971313477,14.165755271911621,'00:00']

