# Optimizer
> Define the general fastai optimizer and the variants


```python
add_docs(_BaseOptimizer, 
         all_params="List of param_groups, parameters, and hypers",
         freeze_to="Freeze parameter groups up to `n`",
         freeze="Freeze up to last parameter group",
         set_freeze="Set `rg` for parameter group `n` only",
         unfreeze="Unfreeze the entire model",
         set_hypers="`set_hyper` for all `kwargs`",
         set_hyper="Set the value(s) in `v` for hyper-parameter `k`")
```


<h2 id="Optimizer" class="doc_header"><code>class</code> <code>Optimizer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L64" class="source_link" style="float:right">[source]</a></h2>

> <code>Optimizer</code>(**`params`**, **`cbs`**, **`train_bn`**=*`True`*, **\*\*`defaults`**) :: `_BaseOptimizer`

Base optimizer class for the fastai library, updating `params` with `cbs`


```python
add_docs(Optimizer, 
         zero_grad="Standard PyTorch API: Zero all the grad attributes of the parameters",
         step="Standard PyTorch API: Update the stats and execute the steppers in on all parameters that have a grad",
         state_dict="Return the state of the optimizer in a dictionary",
         load_state_dict="Load the content of `sd`",
         clear_state="Reset the state of the optimizer")
```

### Initializing an Optimizer

[`params`](/torch_core.html#params) will be used to create the `param_groups` of the optimizer. If it's a collection (or a generator) of parameters, it will be a [`L`](https://fastcore.fast.ai/foundation#L) containing one [`L`](https://fastcore.fast.ai/foundation#L) with all the parameters. To define multiple parameter groups [`params`](/torch_core.html#params) should be passed as a collection (or a generator) of [`L`](https://fastcore.fast.ai/foundation#L)s.
{% include note.html content='In PyTorch, <code>model.parameters()</code> returns a generator with all the parameters, that you can directly pass to <code>Optimizer</code>.' %}

```python
opt = Optimizer([1,2,3], noop)
test_eq(opt.param_lists, [[1,2,3]])
opt = Optimizer(range(3), noop)
test_eq(opt.param_lists, [[0,1,2]])
opt = Optimizer([[1,2],[3]], noop)
test_eq(opt.param_lists, [[1,2],[3]])
opt = Optimizer(([o,o+1] for o in range(0,4,2)), noop)
test_eq(opt.param_lists, [[0,1],[2,3]])
```

`cbs` is a list of functions that will be composed when applying the step. For instance, you can compose a function making the SGD step, with another one applying weight decay. Additionally, each `cb` can have a [`defaults`](https://fastcore.fast.ai/foundation#defaults) attribute that contains hyper-parameters and their default value. Those are all gathered at initialization, and new values can be passed to override those defaults with the [`defaults`](https://fastcore.fast.ai/foundation#defaults) kwargs. The steppers will be called by [`Optimizer.step`](/optimizer.html#Optimizer.step) (which is the standard PyTorch name), and gradients can be cleared with [`Optimizer.zero_grad`](/optimizer.html#Optimizer.zero_grad) (also a standard PyTorch name).

Once the defaults have all been pulled off, they are copied as many times as there are `param_groups` and stored in `hypers`. To apply different hyper-parameters to different groups (differential learning rates, or no weight decay for certain layers for instance), you will need to adjust those values after the init. 

```python
def tst_arg(p, lr=0, **kwargs): return p
tst_arg.defaults = dict(lr=1e-2)

def tst_arg2(p, lr2=0, **kwargs): return p
tst_arg2.defaults = dict(lr2=1e-3)

def tst_arg3(p, mom=0, **kwargs): return p
tst_arg3.defaults = dict(mom=0.9)

def tst_arg4(p, **kwargs): return p

opt = Optimizer([1,2,3], [tst_arg,tst_arg2, tst_arg3])
test_eq(opt.hypers, [{'lr2': 1e-3, 'mom': 0.9, 'lr': 1e-2}])
opt = Optimizer([1,2,3], tst_arg, lr=0.1)
test_eq(opt.hypers, [{'lr': 0.1}])
opt = Optimizer([[1,2],[3]], tst_arg)
test_eq(opt.hypers, [{'lr': 1e-2}, {'lr': 1e-2}])
opt = Optimizer([[1,2],[3]], tst_arg, lr=0.1)
test_eq(opt.hypers, [{'lr': 0.1}, {'lr': 0.1}])
```

For each hyper-parameter, you can pass a slice or a collection to set them, if there are multiple parameter groups. A slice will be converted to a log-uniform collection from its beginning to its end, or if it only has an end `e`, to a collection of as many values as there are parameter groups that are `...,e/10,e/10,e`.

Setting an hyper-parameter with a collection that has a different number of elements than the optimizer has parameter groups will raise an error.

```python
opt = Optimizer([[1,2],[3]], tst_arg, lr=[0.1,0.2])
test_eq(opt.hypers, [{'lr': 0.1}, {'lr': 0.2}])
opt = Optimizer([[1,2],[3],[4]], tst_arg, lr=slice(1e-2))
test_eq(opt.hypers, [{'lr': 1e-3}, {'lr': 1e-3}, {'lr': 1e-2}])
opt = Optimizer([[1,2],[3],[4]], tst_arg, lr=slice(1e-4,1e-2))
test_eq(opt.hypers, [{'lr': 1e-4}, {'lr': 1e-3}, {'lr': 1e-2}])
test_eq(opt.param_groups, [{'params': [1,2], 'lr': 1e-4}, {'params': [3], 'lr': 1e-3}, {'params': [4], 'lr': 1e-2}])
test_fail(lambda: Optimizer([[1,2],[3],[4]], tst_arg, lr=np.array([0.1,0.2])))
```

### Basic steppers

To be able to give examples of optimizer steps, we will need some steppers, like the following:


<h4 id="sgd_step" class="doc_header"><code>sgd_step</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L101" class="source_link" style="float:right">[source]</a></h4>

> <code>sgd_step</code>(**`p`**, **`lr`**, **\*\*`kwargs`**)




```python
def tst_param(val, grad=None):
    "Create a tensor with `val` and a gradient of `grad` for testing"
    res = tensor([val]).float()
    res.grad = tensor([val/10 if grad is None else grad]).float()
    return res
```

```python
p = tst_param(1., 0.1)
sgd_step(p, 1.)
test_eq(p, tensor([0.9]))
test_eq(p.grad, tensor([0.1]))
```


<h4 id="weight_decay" class="doc_header"><code>weight_decay</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L105" class="source_link" style="float:right">[source]</a></h4>

> <code>weight_decay</code>(**`p`**, **`lr`**, **`wd`**, **`do_wd`**=*`True`*, **\*\*`kwargs`**)

Weight decay as decaying `p` with `lr*wd`


```python
p = tst_param(1., 0.1)
weight_decay(p, 1., 0.1)
test_eq(p, tensor([0.9]))
test_eq(p.grad, tensor([0.1]))
```


<h4 id="l2_reg" class="doc_header"><code>l2_reg</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L112" class="source_link" style="float:right">[source]</a></h4>

> <code>l2_reg</code>(**`p`**, **`lr`**, **`wd`**, **`do_wd`**=*`True`*, **\*\*`kwargs`**)

L2 regularization as adding `wd*p` to `p.grad`


```python
p = tst_param(1., 0.1)
l2_reg(p, 1., 0.1)
test_eq(p, tensor([1.]))
test_eq(p.grad, tensor([0.2]))
```

{% include warning.html content='Weight decay and L2 regularization is the same thing for basic SGD, but for more complex optimizers, they are very different.' %}

### Making the step


<h4 id="Optimizer.step" class="doc_header"><code>Optimizer.step</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L81" class="source_link" style="float:right">[source]</a></h4>

> <code>Optimizer.step</code>()




This method will loop over all param groups, then all parameters for which `grad` is not None and call each function in `stepper`, passing it the parameter `p` with the hyper-parameters in the corresponding dict in `hypers`.

```python
r = L.range(4)
def tst_params(): return r.map(tst_param)

params = tst_params()
opt = Optimizer(params, sgd_step, lr=0.1)
opt.step()
test_close([p.item() for p in params], r.map(mul(0.99)))
```

```python
params = tst_params()
opt = Optimizer(params, [weight_decay, sgd_step], lr=0.1, wd=0.1)
opt.step()
test_close([p.item() for p in params], r.map(mul(0.98)))
```

```python
params = tst_params()
opt = Optimizer(params, sgd_step, lr=0.1)
params[-1].grad = None
opt.step()
test_close([p.item() for p in params], [0., 0.99, 1.98, 3.])
```

```python
params = tst_params()
opt = Optimizer([params[:2], params[2:]], sgd_step, lr=0.1)
opt.hypers[0]['lr'] = 0.01
opt.step()
test_close([p.item() for p in params], [0., 0.999, 1.98, 2.97])
```


<h4 id="Optimizer.zero_grad" class="doc_header"><code>Optimizer.zero_grad</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L76" class="source_link" style="float:right">[source]</a></h4>

> <code>Optimizer.zero_grad</code>()




```python
params = tst_params()
opt = Optimizer(params, [weight_decay, sgd_step], lr=0.1, wd=0.1)
opt.zero_grad()
[test_eq(p.grad, tensor([0.])) for p in params];
```

Some of the [`Optimizer`](/optimizer.html#Optimizer) `cbs` can be functions updating the state associated with a parameter. That state can then be used by any stepper. The best example is a momentum calculation.

```python
def tst_stat(p, **kwargs): 
    s = kwargs.get('sum', torch.zeros_like(p)) + p.data
    return {'sum': s}
tst_stat.defaults = {'mom': 0.9}

#Test Optimizer init
opt = Optimizer([1,2,3], tst_stat)
test_eq(opt.hypers, [{'mom': 0.9}])
opt = Optimizer([1,2,3], tst_stat, mom=0.99)
test_eq(opt.hypers, [{'mom': 0.99}])

#Test stat
x = torch.randn(4,5)
state = tst_stat(x)
assert 'sum' in state
test_eq(x, state['sum'])
state = tst_stat(x, **state)
test_eq(state['sum'], 2*x)
```

## Statistics


<h4 id="average_grad" class="doc_header"><code>average_grad</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L119" class="source_link" style="float:right">[source]</a></h4>

> <code>average_grad</code>(**`p`**, **`mom`**, **`dampening`**=*`False`*, **`grad_avg`**=*`None`*, **\*\*`kwargs`**)

Keeps track of the avg grads of `p` in `state` with `mom`.


`dampening=False` gives the classical formula for momentum in SGD: 
```
new_val = old_val * mom + grad
```
whereas `dampening=True` makes it an exponential moving average:
```
new_val = old_val * mom + grad * (1-mom)
```

```python
p = tst_param([1,2,3], [4,5,6])
state = {}
state = average_grad(p, mom=0.9, **state)
test_eq(state['grad_avg'], p.grad)
state = average_grad(p, mom=0.9, **state)
test_eq(state['grad_avg'], p.grad * 1.9)

#Test dampening
state = {}
state = average_grad(p,  mom=0.9, dampening=True, **state)
test_eq(state['grad_avg'], 0.1*p.grad)
state = average_grad(p, mom=0.9, dampening=True, **state)
test_close(state['grad_avg'], (0.1*0.9+0.1)*p.grad)
```


<h4 id="average_sqr_grad" class="doc_header"><code>average_sqr_grad</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L129" class="source_link" style="float:right">[source]</a></h4>

> <code>average_sqr_grad</code>(**`p`**, **`sqr_mom`**, **`dampening`**=*`True`*, **`sqr_avg`**=*`None`*, **\*\*`kwargs`**)




`dampening=False` gives the classical formula for momentum in SGD: 
```
new_val = old_val * mom + grad**2
```
whereas `dampening=True` makes it an exponential moving average:
```
new_val = old_val * mom + (grad**2) * (1-mom)
```

```python
p = tst_param([1,2,3], [4,5,6])
state = {}
state = average_sqr_grad(p, sqr_mom=0.99, dampening=False, **state)
test_eq(state['sqr_avg'], p.grad.pow(2))
state = average_sqr_grad(p, sqr_mom=0.99, dampening=False, **state)
test_eq(state['sqr_avg'], p.grad.pow(2) * 1.99)

#Test dampening
state = {}
state = average_sqr_grad(p, sqr_mom=0.99, **state)
test_close(state['sqr_avg'], 0.01*p.grad.pow(2))
state = average_sqr_grad(p, sqr_mom=0.99, **state)
test_close(state['sqr_avg'], (0.01*0.99+0.01)*p.grad.pow(2))
```

### Freezing part of the model


<h4 id="Optimizer.freeze" class="doc_header"><code>Optimizer.freeze</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L26" class="source_link" style="float:right">[source]</a></h4>

> <code>Optimizer.freeze</code>()





<h4 id="Optimizer.freeze_to" class="doc_header"><code>Optimizer.freeze_to</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L19" class="source_link" style="float:right">[source]</a></h4>

> <code>Optimizer.freeze_to</code>(**`n`**)





<h4 id="Optimizer.unfreeze" class="doc_header"><code>Optimizer.unfreeze</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L33" class="source_link" style="float:right">[source]</a></h4>

> <code>Optimizer.unfreeze</code>()




```python
params = [tst_params(), tst_params(), tst_params()]
opt = Optimizer(params, sgd_step, lr=0.1)
opt.freeze_to(1)
req_grad = Self.requires_grad()
test_eq(L(params[0]).map(req_grad), [False]*4)
for i in {1,2}: test_eq(L(params[i]).map(req_grad), [True]*4)
    
#Unfreezing
opt.unfreeze()
for i in range(2): test_eq(L(params[i]).map(req_grad), [True]*4)

#TODO: test warning
# opt.freeze_to(3)
```

Parameters such as batchnorm weights/bias can be marked to always be in training mode, just put `force_train=true` in their state.

```python
params = [tst_params(), tst_params(), tst_params()]
opt = Optimizer(params, sgd_step, lr=0.1)
for p in L(params[1])[[1,3]]: opt.state[p] = {'force_train': True}
opt.freeze()
test_eq(L(params[0]).map(req_grad), [False]*4)
test_eq(L(params[1]).map(req_grad), [False, True, False, True])
test_eq(L(params[2]).map(req_grad), [True]*4)
```

### Serializing


<h4 id="Optimizer.state_dict" class="doc_header"><code>Optimizer.state_dict</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L90" class="source_link" style="float:right">[source]</a></h4>

> <code>Optimizer.state_dict</code>()





<h4 id="Optimizer.load_state_dict" class="doc_header"><code>Optimizer.load_state_dict</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L94" class="source_link" style="float:right">[source]</a></h4>

> <code>Optimizer.load_state_dict</code>(**`sd`**)




```python
p = tst_param([1,2,3], [4,5,6])
opt = Optimizer(p, average_grad)
opt.step()
test_eq(opt.state[p]['grad_avg'], tensor([[4., 5., 6.]]))

sd = opt.state_dict()
p1 = tst_param([10,20,30], [40,50,60])
opt = Optimizer(p1, average_grad, mom=0.99)
test_eq(opt.hypers[0]['mom'], 0.99)
test_eq(opt.state, {})

opt.load_state_dict(sd)
test_eq(opt.hypers[0]['mom'], 0.9)
test_eq(opt.state[p1]['grad_avg'], tensor([[4., 5., 6.]]))
```


<h4 id="Optimizer.clear_state" class="doc_header"><code>Optimizer.clear_state</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L86" class="source_link" style="float:right">[source]</a></h4>

> <code>Optimizer.clear_state</code>()




```python
p = tst_param([1,2,3], [4,5,6])
opt = Optimizer(p, average_grad)
opt.state[p] = {'force_train': True}
opt.step()
test_eq(opt.state[p]['grad_avg'], tensor([[4., 5., 6.]]))

opt.clear_state()
test_eq(opt.state[p], {'force_train': True})
```

## Optimizers

### SGD with momentum


<h4 id="momentum_step" class="doc_header"><code>momentum_step</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L138" class="source_link" style="float:right">[source]</a></h4>

> <code>momentum_step</code>(**`p`**, **`lr`**, **`grad_avg`**, **\*\*`kwargs`**)

Step for SGD with momentum with `lr`



<h4 id="SGD" class="doc_header"><code>SGD</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L143" class="source_link" style="float:right">[source]</a></h4>

> <code>SGD</code>(**`params`**, **`lr`**, **`mom`**=*`0.0`*, **`wd`**=*`0.0`*, **`decouple_wd`**=*`True`*)

A [`Optimizer`](/optimizer.html#Optimizer) for SGD with `lr` and `mom` and `params`


Optional weight decay of `wd` is applied, as true weight decay (decay the weights directly) if `decouple_wd=True` else as L2 regularization (add the decay to the gradients).

```python
params = tst_params()
opt = SGD(params, lr=0.1)
opt.step()
test_close([p.item() for p in params], [i*0.99 for i in range(4)])
opt.step()
[p.item() for p in params]
test_close([p.item() for p in params], [i*0.98 for i in range(4)])
```

```python
params = tst_params()
opt = SGD(params, lr=0.1, mom=0.9)
assert isinstance(opt, Optimizer)
opt.step()
test_close([p.item() for p in params], [i*0.99 for i in range(4)])
opt.step()
[p.item() for p in params]
test_close([p.item() for p in params], [i*(1 - 0.1 * (0.1 + 0.1*1.9)) for i in range(4)])
for i,p in enumerate(params): test_close(opt.state[p]['grad_avg'].item(), i*0.19)
```

Test weight decay, notice how we can see that L2 regularization is different from weight decay even for simple SGD with momentum.

```python
params = tst_params()
#Weight decay
opt = SGD(params, lr=0.1, mom=0.9, wd=0.1)
opt.step()
test_close([p.item() for p in params], [i*0.98 for i in range(4)])
#L2 reg
opt = SGD(params, lr=0.1, mom=0.9, wd=0.1, decouple_wd=False)
opt.step()
#TODO: fix cause this formula was wrong
#test_close([p.item() for p in params], [i*0.97 for i in range(4)])
```

### RMSProp


<h4 id="rms_prop_step" class="doc_header"><code>rms_prop_step</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L152" class="source_link" style="float:right">[source]</a></h4>

> <code>rms_prop_step</code>(**`p`**, **`lr`**, **`sqr_avg`**, **`eps`**, **`grad_avg`**=*`None`*, **\*\*`kwargs`**)

Step for SGD with momentum with `lr`



<h4 id="RMSProp" class="doc_header"><code>RMSProp</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L160" class="source_link" style="float:right">[source]</a></h4>

> <code>RMSProp</code>(**`params`**, **`lr`**, **`sqr_mom`**=*`0.99`*, **`mom`**=*`0.0`*, **`wd`**=*`0.0`*, **`decouple_wd`**=*`True`*)

A [`Optimizer`](/optimizer.html#Optimizer) for RMSProp with `lr`, `sqr_mom`, `mom` and `params`


RMSProp was introduced by Geoffrey Hinton in his [course](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf). What is named `sqr_mom` here is the `alpha` in the course. Optional weight decay of `wd` is applied, as true weight decay (decay the weights directly) if `decouple_wd=True` else as L2 regularization (add the decay to the gradients).

```python
params = tst_param([1,2,3], [0.1,0.2,0.3])
opt = RMSProp(params, lr=0.1)
opt.step()
test_close(params[0], tensor([0.,1.,2.]))
opt.step()
step = - 0.1 * 0.1 / (math.sqrt((0.01*0.99+0.01) * 0.1**2) + 1e-8)
test_close(params[0], tensor([step, 1+step, 2+step]))
```

```python
params = tst_param([1,2,3], [0.1,0.2,0.3])
opt = RMSProp(params, lr=0.1, mom=0.9)
opt.step()
test_close(params[0], tensor([0.,1.,2.]))
opt.step()
step = - 0.1 * (0.1 + 0.9*0.1) / (math.sqrt((0.01*0.99+0.01) * 0.1**2) + 1e-8)
test_close(params[0], tensor([step, 1+step, 2+step]))
```

### Adam


<h4 id="step_stat" class="doc_header"><code>step_stat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L169" class="source_link" style="float:right">[source]</a></h4>

> <code>step_stat</code>(**`p`**, **`step`**=*`0`*, **\*\*`kwargs`**)

Register the number of steps done in `state` for `p`


```python
p = tst_param(1,0.1)
state = {}
state = step_stat(p, **state)
test_eq(state['step'], 1)
for _ in range(5): state = step_stat(p, **state)
test_eq(state['step'], 6)
```


<h4 id="debias" class="doc_header"><code>debias</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L175" class="source_link" style="float:right">[source]</a></h4>

> <code>debias</code>(**`mom`**, **`damp`**, **`step`**)





<h4 id="adam_step" class="doc_header"><code>adam_step</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L178" class="source_link" style="float:right">[source]</a></h4>

> <code>adam_step</code>(**`p`**, **`lr`**, **`mom`**, **`step`**, **`sqr_mom`**, **`grad_avg`**, **`sqr_avg`**, **`eps`**, **\*\*`kwargs`**)

Step for Adam with `lr` on `p`



<h4 id="Adam" class="doc_header"><code>Adam</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L188" class="source_link" style="float:right">[source]</a></h4>

> <code>Adam</code>(**`params`**, **`lr`**, **`mom`**=*`0.9`*, **`sqr_mom`**=*`0.99`*, **`eps`**=*`1e-05`*, **`wd`**=*`0.01`*, **`decouple_wd`**=*`True`*)

A [`Optimizer`](/optimizer.html#Optimizer) for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`


Adam was introduced by Diederik P. Kingma and Jimmy Ba in [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). For consistency across optimizers, we renamed `beta1` and `beta2` in the paper to `mom` and  `sqr_mom`. Note that our defaults also differ from the paper (0.99 for `sqr_mom` or `beta2`, 1e-5 for `eps`). Those values seem to be better from our experiments in a wide range of situations.

Optional weight decay of `wd` is applied, as true weight decay (decay the weights directly) if `decouple_wd=True` else as L2 regularization (add the decay to the gradients).
{% include note.html content='Don&#8217;t forget that `eps` is an hyper-parameter you can change. Some models won&#8217;t train without a very high `eps` like 0.1 (intuitively, the higher `eps` is, the closer we are to normal SGD). The usual default of 1e-8 is often too extreme in the sense we don&#8217;t manage to get as good results as with SGD. ' %}

```python
params = tst_param([1,2,3], [0.1,0.2,0.3])
opt = Adam(params, lr=0.1, wd=0)
opt.step()
step = -0.1 * 0.1 / (math.sqrt(0.1**2) + 1e-8)
test_close(params[0], tensor([1+step, 2+step, 3+step]))
opt.step()
test_close(params[0], tensor([1+2*step, 2+2*step, 3+2*step]), eps=1e-3)
```

### RAdam

RAdam (for rectified Adam) was introduced by Zhang et al. in [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1907.08610) to slightly modify the Adam optimizer to be more stable at the beginning of training (and thus not require a long warmup). They use an estimate of the variance of the moving average of the squared gradients (the term in the denominator of traditional Adam) and rescale this moving average by this term before performing the update.

This version also incorporates [SAdam](https://arxiv.org/abs/1908.00700); set `beta` to enable this (definition same as in the paper).


<h4 id="radam_step" class="doc_header"><code>radam_step</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L196" class="source_link" style="float:right">[source]</a></h4>

> <code>radam_step</code>(**`p`**, **`lr`**, **`mom`**, **`step`**, **`sqr_mom`**, **`grad_avg`**, **`sqr_avg`**, **`eps`**, **`beta`**, **\*\*`kwargs`**)

Step for RAdam with `lr` on `p`



<h4 id="RAdam" class="doc_header"><code>RAdam</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L214" class="source_link" style="float:right">[source]</a></h4>

> <code>RAdam</code>(**`params`**, **`lr`**, **`mom`**=*`0.9`*, **`sqr_mom`**=*`0.99`*, **`eps`**=*`1e-05`*, **`wd`**=*`0.0`*, **`beta`**=*`0.0`*, **`decouple_wd`**=*`True`*)

A [`Optimizer`](/optimizer.html#Optimizer) for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`


This is the effective correction reported to the adam step for 500 iterations in RAdam. We can see how it goes from 0 to 1, mimicking the effect of a warm-up.

```python
beta = 0.99
r_inf = 2/(1-beta) - 1
rs = np.array([r_inf - 2*s*beta**s/(1-beta**s) for s in range(5,500)])
v = np.sqrt(((rs-4) * (rs-2) * r_inf)/((r_inf-4)*(r_inf-2)*rs))
plt.plot(v);
```


![png](output_99_0.png)


```python
params = tst_param([1,2,3], [0.1,0.2,0.3])
opt = RAdam(params, lr=0.1)
#The r factor is lower than 5 during the first 5 steps so updates use the average of gradients (all the same)
r_inf = 2/(1-0.99) - 1
for i in range(5): 
    r = r_inf - 2*(i+1)*0.99**(i+1)/(1-0.99**(i+1))
    assert r <= 5
    opt.step()
p = tensor([0.95, 1.9, 2.85])
test_close(params[0], p)

#The r factor is greater than 5 for the sixth step so we update with RAdam
r = r_inf - 2*6*0.99**6/(1-0.99**6)
assert r > 5
opt.step()
v = math.sqrt(((r-4) * (r-2) * r_inf)/((r_inf-4)*(r_inf-2)*r))
step = -0.1*0.1*v/(math.sqrt(0.1**2) + 1e-8)
test_close(params[0], p+step)
```

### QHAdam

QHAdam (for Quasi-Hyperbolic Adam) was introduced by Ma & Yarats in [Quasi-Hyperbolic Momentum and Adam for Deep Learning](https://arxiv.org/pdf/1810.06801.pdf) as a *"computationally cheap, intuitive to interpret, and simple to implement"* optimizer. Additional code can be found in their [qhoptim repo](https://github.com/facebookresearch/qhoptim). QHAdam is based on QH-Momentum, which introduces the immediate discount factor `nu`, encapsulating plain SGD (`nu = 0`) and momentum (`nu = 1`). QH-Momentum is defined below, where g_t+1 is the update of the moment. An interpretation of QHM is as a nu-weighted average of the momentum update step and the plain SGD update step.

> θ_t+1 ← θ_t − lr * [(1 − nu) · ∇L_t(θ_t) + nu · g_t+1]

QHAdam takes the concept behind QHM above and applies it to Adam, replacing both of Adam’s moment estimators with quasi-hyperbolic terms. 

The paper's suggested default parameters are `mom = 0.999`, `sqr_mom = 0.999`, `nu_1 = 0.7` and `and nu_2 = 1.0`. When training is not stable, it is possible that setting `nu_2 < 1` can improve stability by imposing a tighter step size bound. Note that QHAdam recovers Adam when `nu_1 = nu_2 = 1.0`. QHAdam recovers RMSProp (Hinton et al., 2012) when `nu_1 = 0` and `nu_2 = 1`, and NAdam (Dozat, 2016) when `nu_1 = mom` and `nu_2 = 1`.

Optional weight decay of `wd` is applied, as true weight decay (decay the weights directly) if `decouple_wd=True` else as L2 regularization (add the decay to the gradients).


<h4 id="qhadam_step" class="doc_header"><code>qhadam_step</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L222" class="source_link" style="float:right">[source]</a></h4>

> <code>qhadam_step</code>(**`p`**, **`lr`**, **`mom`**, **`sqr_mom`**, **`sqr_avg`**, **`nu_1`**, **`nu_2`**, **`step`**, **`grad_avg`**, **`eps`**, **\*\*`kwargs`**)





<h4 id="QHAdam" class="doc_header"><code>QHAdam</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L233" class="source_link" style="float:right">[source]</a></h4>

> <code>QHAdam</code>(**`params`**, **`lr`**, **`mom`**=*`0.999`*, **`sqr_mom`**=*`0.999`*, **`nu_1`**=*`0.7`*, **`nu_2`**=*`1.0`*, **`eps`**=*`1e-08`*, **`wd`**=*`0.0`*, **`decouple_wd`**=*`True`*)

An [`Optimizer`](/optimizer.html#Optimizer) for Adam with `lr`, `mom`, `sqr_mom`, `nus`, eps` and `params`


```python
params = tst_param([1,2,3], [0.1,0.2,0.3])
opt = QHAdam(params, lr=0.1)
opt.step()
step = -0.1 * (((1-0.7) * 0.1) + (0.7 * 0.1)) / (
     math.sqrt(((1-1.0) * 0.1**2) + (1.0 * 0.1**2)) + 1e-8) 
test_close(params[0], tensor([1+step, 2+step, 3+step]))
opt.step()
test_close(params[0], tensor([1+2*step, 2+2*step, 3+2*step]), eps=1e-3)
```

### LARS/LARC


<h4 id="larc_layer_lr" class="doc_header"><code>larc_layer_lr</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L242" class="source_link" style="float:right">[source]</a></h4>

> <code>larc_layer_lr</code>(**`p`**, **`lr`**, **`trust_coeff`**, **`wd`**, **`eps`**, **`clip`**=*`True`*, **\*\*`kwargs`**)

Computes the local lr before weight decay is applied



<h4 id="larc_step" class="doc_header"><code>larc_step</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L251" class="source_link" style="float:right">[source]</a></h4>

> <code>larc_step</code>(**`p`**, **`local_lr`**, **`grad_avg`**=*`None`*, **\*\*`kwargs`**)

Step for LARC `local_lr` on `p`



<h4 id="Larc" class="doc_header"><code>Larc</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L256" class="source_link" style="float:right">[source]</a></h4>

> <code>Larc</code>(**`params`**, **`lr`**, **`mom`**=*`0.9`*, **`clip`**=*`True`*, **`trust_coeff`**=*`0.02`*, **`eps`**=*`1e-08`*, **`wd`**=*`0.0`*, **`decouple_wd`**=*`True`*)

A [`Optimizer`](/optimizer.html#Optimizer) for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`


The LARS optimizer was first introduced in [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888) then refined in its LARC variant (original LARS is with `clip=False`). A learning rate is computed for each individual layer with a certain `trust_coefficient`, then clipped to be always less than `lr`.

Optional weight decay of `wd` is applied, as true weight decay (decay the weights directly) if `decouple_wd=True` else as L2 regularization (add the decay to the gradients).

```python
params = [tst_param([1,2,3], [0.1,0.2,0.3]), tst_param([1,2,3], [0.01,0.02,0.03])]
opt = Larc(params, lr=0.1)
opt.step()
#First param local lr is 0.02 < lr so it's not clipped
test_close(opt.state[params[0]]['local_lr'], 0.02)
#Second param local lr is 0.2 > lr so it's clipped
test_eq(opt.state[params[1]]['local_lr'], 0.1)
test_close(params[0], tensor([0.998,1.996,2.994]))
test_close(params[1], tensor([0.999,1.998,2.997]))
```

```python
params = [tst_param([1,2,3], [0.1,0.2,0.3]), tst_param([1,2,3], [0.01,0.02,0.03])]
opt = Larc(params, lr=0.1, clip=False)
opt.step()
#No clipping
test_close(opt.state[params[0]]['local_lr'], 0.02)
test_close(opt.state[params[1]]['local_lr'], 0.2)
test_close(params[0], tensor([0.998,1.996,2.994]))
test_close(params[1], tensor([0.998,1.996,2.994]))
```

### LAMB


<h4 id="lamb_step" class="doc_header"><code>lamb_step</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L265" class="source_link" style="float:right">[source]</a></h4>

> <code>lamb_step</code>(**`p`**, **`lr`**, **`mom`**, **`step`**, **`sqr_mom`**, **`grad_avg`**, **`sqr_avg`**, **`eps`**, **\*\*`kwargs`**)

Step for LAMB with `lr` on `p`



<h4 id="Lamb" class="doc_header"><code>Lamb</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L278" class="source_link" style="float:right">[source]</a></h4>

> <code>Lamb</code>(**`params`**, **`lr`**, **`mom`**=*`0.9`*, **`sqr_mom`**=*`0.99`*, **`eps`**=*`1e-05`*, **`wd`**=*`0.0`*, **`decouple_wd`**=*`True`*)

A [`Optimizer`](/optimizer.html#Optimizer) for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`


LAMB was introduced in [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962). Intuitively, it's LARC applied to Adam. As in [`Adam`](/optimizer.html#Adam), we renamed `beta1` and `beta2` in the paper to `mom` and  `sqr_mom`. Note that our defaults also differ from the paper (0.99 for `sqr_mom` or `beta2`, 1e-5 for `eps`). Those values seem to be better from our experiments in a wide range of situations.

Optional weight decay of `wd` is applied, as true weight decay (decay the weights directly) if `decouple_wd=True` else as L2 regularization (add the decay to the gradients).

```python
params = tst_param([1,2,3], [0.1,0.2,0.3])
opt = Lamb(params, lr=0.1)
opt.step()
test_close(params[0], tensor([0.7840,1.7840,2.7840]), eps=1e-3)
```

Lookahead was introduced by Zhang et al. in [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610). It can be run on top of any optimizer and consists in having the final weights of the model be a moving average. In practice, we update our model using the internal optimizer but keep a copy of old weights that and every `k` steps, we change the weights by a moving average of the *fast weights* (the ones updated by the inner optimizer) with the *slow weights* (the copy of old weights). Those *slow weights* act like a stability mechanism.


<h2 id="Lookahead" class="doc_header"><code>class</code> <code>Lookahead</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L287" class="source_link" style="float:right">[source]</a></h2>

> <code>Lookahead</code>(**`opt`**, **`k`**=*`6`*, **`alpha`**=*`0.5`*) :: [`Optimizer`](/optimizer.html#Optimizer)

Wrap `opt` in a lookahead optimizer


```python
params = tst_param([1,2,3], [0.1,0.2,0.3])
p,g = params[0].data.clone(),tensor([0.1,0.2,0.3])
opt = Lookahead(SGD(params, lr=0.1))
for k in range(5): opt.step()
#first 5 steps are normal SGD steps
test_close(params[0], p - 0.5*g)
#Since k=6, sixth step is a moving average of the 6 SGD steps with the initial weight
opt.step()
test_close(params[0], p * 0.5 + (p-0.6*g) * 0.5)
```


<h4 id="ranger" class="doc_header"><code>ranger</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L327" class="source_link" style="float:right">[source]</a></h4>

> <code>ranger</code>(**`p`**, **`lr`**, **`mom`**=*`0.95`*, **`wd`**=*`0.01`*, **`eps`**=*`1e-06`*, **`sqr_mom`**=*`0.99`*, **`beta`**=*`0.0`*, **`decouple_wd`**=*`True`*)

Convenience method for [`Lookahead`](/optimizer.html#Lookahead) with [`RAdam`](/optimizer.html#RAdam)



<h4 id="detuplify_pg" class="doc_header"><code>detuplify_pg</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L333" class="source_link" style="float:right">[source]</a></h4>

> <code>detuplify_pg</code>(**`d`**)




```python
tst = {'lr': 1e-2, 'mom': 0.9, 'params':[0,1,2]}
test_eq(detuplify_pg(tst), {'lr': 1e-2, 'mom': 0.9})
tst = {'lr': 1e-2, 'betas': (0.9,0.999), 'params':[0,1,2]}
test_eq(detuplify_pg(tst), {'lr': 1e-2, 'betas__0': 0.9, 'betas__1': 0.999})
```


<h4 id="set_item_pg" class="doc_header"><code>set_item_pg</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L342" class="source_link" style="float:right">[source]</a></h4>

> <code>set_item_pg</code>(**`pg`**, **`k`**, **`v`**)




```python
tst = {'lr': 1e-2, 'mom': 0.9, 'params':[0,1,2]}
test_eq(set_item_pg(tst, 'lr', 1e-3), {'lr': 1e-3, 'mom': 0.9, 'params':[0,1,2]})
tst = {'lr': 1e-2, 'betas': (0.9,0.999), 'params':[0,1,2]}
test_eq(set_item_pg(tst, 'betas__0', 0.95), {'lr': 1e-2, 'betas': (0.95,0.999), 'params':[0,1,2]})
```


<h2 id="OptimWrapper" class="doc_header"><code>class</code> <code>OptimWrapper</code><a href="https://github.com/fastai/fastai/tree/master/fastai/optimizer.py#L353" class="source_link" style="float:right">[source]</a></h2>

> <code>OptimWrapper</code>(**`opt`**, **`hp_map`**=*`None`*) :: `_BaseOptimizer`

Common functionality between [`Optimizer`](/optimizer.html#Optimizer) and [`OptimWrapper`](/optimizer.html#OptimWrapper)


```python
sgd = SGD([tensor([1,2,3])], lr=1e-3, mom=0.9, wd=1e-2)
tst_sgd = OptimWrapper(torch.optim.SGD([tensor([1,2,3])], lr=1e-3, momentum=0.9, weight_decay=1e-2))
#Access to param_groups
test_eq(tst_sgd.param_lists, sgd.param_lists)
#Set param_groups
tst_sgd.param_lists = [[tensor([4,5,6])]]
test_eq(tst_sgd.opt.param_groups[0]['params'], [tensor(4,5,6)])
#Access to hypers
test_eq(tst_sgd.hypers, [{**sgd.hypers[0], 'dampening': 0., 'nesterov': False}])
#Set hypers
tst_sgd.set_hyper('mom', 0.95)
test_eq(tst_sgd.opt.param_groups[0]['momentum'], 0.95)
```

```python
tst_sgd = OptimWrapper(torch.optim.SGD([{'params': [tensor([1,2,3])], 'lr': 1e-3}, 
                                        {'params': [tensor([4,5,6])], 'lr': 1e-2}], momentum=0.9, weight_decay=1e-2))
sgd = SGD([[tensor([1,2,3])], [tensor([4,5,6])]], lr=[1e-3, 1e-2], mom=0.9, wd=1e-2)
#Access to param_groups
test_eq(tst_sgd.param_lists, sgd.param_lists)
#Set param_groups
tst_sgd.param_lists = [[tensor([4,5,6])], [tensor([1,2,3])]]
test_eq(tst_sgd.opt.param_groups[0]['params'], [tensor(4,5,6)])
test_eq(tst_sgd.opt.param_groups[1]['params'], [tensor(1,2,3)])
#Access to hypers
test_eq(tst_sgd.hypers, [{**sgd.hypers[i], 'dampening': 0., 'nesterov': False} for i in range(2)])
#Set hypers
tst_sgd.set_hyper('mom', 0.95)
test_eq([pg['momentum'] for pg in tst_sgd.opt.param_groups], [0.95,0.95])
tst_sgd.set_hyper('lr', [1e-4,1e-3])
test_eq([pg['lr'] for pg in tst_sgd.opt.param_groups], [1e-4,1e-3])
```

```python
def _mock_train(m, x, y, opt):
    m.train()
    for i in range(0, 100, 25):
        z = m(x[i:i+25])
        loss = F.mse_loss(z, y[i:i+25])
        loss.backward()
        opt.step()
        opt.zero_grad()
```

```python
m = nn.Linear(4,5)
x = torch.randn(100, 3, 4)
y = torch.randn(100, 3, 5)
try:
    torch.save(m.state_dict(), 'tmp.pth')
    wgt,bias = m.weight.data.clone(),m.bias.data.clone()

    m.load_state_dict(torch.load('tmp.pth'))
    opt1 = OptimWrapper(torch.optim.AdamW(m.parameters(), betas=(0.9, 0.99), eps=1e-5, weight_decay=1e-2))
    _mock_train(m, x.clone(), y.clone(), opt1)
    wgt1,bias1 = m.weight.data.clone(),m.bias.data.clone()

    m.load_state_dict(torch.load('tmp.pth'))
    opt2 = Adam(m.parameters(), 1e-3, wd=1e-2)
    _mock_train(m, x.clone(), y.clone(), opt2)
    wgt2,bias2 = m.weight.data.clone(),m.bias.data.clone()
    
    test_close(wgt1,wgt2,eps=1e-3)
    test_close(bias1,bias2,eps=1e-3)
finally: os.remove('tmp.pth')
```

```python
m = nn.Linear(4,5)
x = torch.randn(100, 3, 4)
y = torch.randn(100, 3, 5)
try:
    torch.save(m.state_dict(), 'tmp.pth')
    wgt,bias = m.weight.data.clone(),m.bias.data.clone()

    m.load_state_dict(torch.load('tmp.pth'))
    opt1 = OptimWrapper(torch.optim.Adam(m.parameters(), betas=(0.9, 0.99), eps=1e-5, weight_decay=1e-2))
    _mock_train(m, x.clone(), y.clone(), opt1)
    wgt1,bias1 = m.weight.data.clone(),m.bias.data.clone()

    m.load_state_dict(torch.load('tmp.pth'))
    opt2 = Adam(m.parameters(), 1e-3, wd=1e-2, decouple_wd=False)
    _mock_train(m, x.clone(), y.clone(), opt2)
    wgt2,bias2 = m.weight.data.clone(),m.bias.data.clone()
    
    test_close(wgt1,wgt2,eps=1e-3)
    test_close(bias1,bias2,eps=1e-3)
finally: os.remove('tmp.pth')
```
