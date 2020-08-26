# GAN
> Basic support for <a href='https://arxiv.org/abs/1406.2661'>Generative Adversarial Networks</a>


GAN stands for [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) and were invented by Ian Goodfellow. The concept is that we train two models at the same time: a generator and a critic. The generator will try to make new images similar to the ones in a dataset, and the critic will try to classify real images from the ones the generator does. The generator returns images, the critic a single number (usually a probability, 0. for fake images and 1. for real ones).

We train them against each other in the sense that at each step (more or less), we:
1. Freeze the generator and train the critic for one step by:
  - getting one batch of true images (let's call that `real`)
  - generating one batch of fake images (let's call that `fake`)
  - have the critic evaluate each batch and compute a loss function from that; the important part is that it rewards positively the detection of real images and penalizes the fake ones
  - update the weights of the critic with the gradients of this loss
  
  
2. Freeze the critic and train the generator for one step by:
  - generating one batch of fake images
  - evaluate the critic on it
  - return a loss that rewards positively the critic thinking those are real images
  - update the weights of the generator with the gradients of this loss

{% include note.html content='The fastai library provides support for training GANs through the GANTrainer, but doesn&#8217;t include more than basic models.' %}

## Wrapping the modules


<h3 id="GANModule" class="doc_header"><code>class</code> <code>GANModule</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L13" class="source_link" style="float:right">[source]</a></h3>

> <code>GANModule</code>(**`generator`**=*`None`*, **`critic`**=*`None`*, **`gen_mode`**=*`False`*) :: [`Module`](/torch_core.html#Module)

Wrapper around a `generator` and a `critic` to create a GAN.


This is just a shell to contain the two models. When called, it will either delegate the input to the `generator` or the `critic` depending of the value of `gen_mode`.


<h4 id="GANModule.switch" class="doc_header"><code>GANModule.switch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L23" class="source_link" style="float:right">[source]</a></h4>

> <code>GANModule.switch</code>(**`gen_mode`**=*`None`*)

Put the module in generator mode if `gen_mode`, in critic mode otherwise.


By default (leaving `gen_mode` to `None`), this will put the module in the other mode (critic mode if it was in generator mode and vice versa).


<h4 id="basic_critic" class="doc_header"><code>basic_critic</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L28" class="source_link" style="float:right">[source]</a></h4>

> <code>basic_critic</code>(**`in_size`**, **`n_channels`**, **`n_features`**=*`64`*, **`n_extra_layers`**=*`0`*, **`norm_type`**=*`<NormType.Batch: 1>`*, **`ks`**=*`3`*, **`stride`**=*`1`*, **`padding`**=*`None`*, **`bias`**=*`None`*, **`ndim`**=*`2`*, **`bn_1st`**=*`True`*, **`act_cls`**=*`ReLU`*, **`transpose`**=*`False`*, **`init`**=*`'auto'`*, **`xtra`**=*`None`*, **`bias_std`**=*`0.01`*, **`dilation`**:`Union`\[`int`, `Tuple`\[`int`, `int`\]\]=*`1`*, **`groups`**:`int`=*`1`*, **`padding_mode`**:`str`=*`'zeros'`*)

A basic critic for images `n_channels` x `in_size` x `in_size`.



<h3 id="AddChannels" class="doc_header"><code>class</code> <code>AddChannels</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L42" class="source_link" style="float:right">[source]</a></h3>

> <code>AddChannels</code>(**`n_dim`**) :: [`Module`](/torch_core.html#Module)

Add `n_dim` channels at the end of the input.



<h4 id="basic_generator" class="doc_header"><code>basic_generator</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L48" class="source_link" style="float:right">[source]</a></h4>

> <code>basic_generator</code>(**`out_size`**, **`n_channels`**, **`in_sz`**=*`100`*, **`n_features`**=*`64`*, **`n_extra_layers`**=*`0`*, **`ks`**=*`3`*, **`stride`**=*`1`*, **`padding`**=*`None`*, **`bias`**=*`None`*, **`ndim`**=*`2`*, **`norm_type`**=*`<NormType.Batch: 1>`*, **`bn_1st`**=*`True`*, **`act_cls`**=*`ReLU`*, **`transpose`**=*`False`*, **`init`**=*`'auto'`*, **`xtra`**=*`None`*, **`bias_std`**=*`0.01`*, **`dilation`**:`Union`\[`int`, `Tuple`\[`int`, `int`\]\]=*`1`*, **`groups`**:`int`=*`1`*, **`padding_mode`**:`str`=*`'zeros'`*)

A basic generator from `in_sz` to images `n_channels` x `out_size` x `out_size`.


```python
critic = basic_critic(64, 3)
generator = basic_generator(64, 3)
tst = GANModule(critic=critic, generator=generator)
real = torch.randn(2, 3, 64, 64)
real_p = tst(real)
test_eq(real_p.shape, [2,1])

tst.switch() #tst is now in generator mode
noise = torch.randn(2, 100)
fake = tst(noise)
test_eq(fake.shape, real.shape)

tst.switch() #tst is back in critic mode
fake_p = tst(fake)
test_eq(fake_p.shape, [2,1])
```


<h4 id="DenseResBlock" class="doc_header"><code>DenseResBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L70" class="source_link" style="float:right">[source]</a></h4>

> <code>DenseResBlock</code>(**`nf`**, **`norm_type`**=*`<NormType.Batch: 1>`*, **`ks`**=*`3`*, **`stride`**=*`1`*, **`padding`**=*`None`*, **`bias`**=*`None`*, **`ndim`**=*`2`*, **`bn_1st`**=*`True`*, **`act_cls`**=*`ReLU`*, **`transpose`**=*`False`*, **`init`**=*`'auto'`*, **`xtra`**=*`None`*, **`bias_std`**=*`0.01`*, **`dilation`**:`Union`\[`int`, `Tuple`\[`int`, `int`\]\]=*`1`*, **`groups`**:`int`=*`1`*, **`padding_mode`**:`str`=*`'zeros'`*)

Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`.



<h4 id="gan_critic" class="doc_header"><code>gan_critic</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L78" class="source_link" style="float:right">[source]</a></h4>

> <code>gan_critic</code>(**`n_channels`**=*`3`*, **`nf`**=*`128`*, **`n_blocks`**=*`3`*, **`p`**=*`0.15`*)

Critic to train a `GAN`.



<h3 id="GANLoss" class="doc_header"><code>class</code> <code>GANLoss</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L96" class="source_link" style="float:right">[source]</a></h3>

> <code>GANLoss</code>(**`gen_loss_func`**, **`crit_loss_func`**, **`gan_model`**) :: [`GANModule`](/vision.gan.html#GANModule)

Wrapper around `crit_loss_func` and `gen_loss_func`


In generator mode, this loss function expects the `output` of the generator and some `target` (a batch of real images). It will evaluate if the generator successfully fooled the critic using `gen_loss_func`. This loss function has the following signature
``` 
def gen_loss_func(fake_pred, output, target):
```
to be able to combine the output of the critic on `output` (which the first argument `fake_pred`) with `output` and `target` (if you want to mix the GAN loss with other losses for instance).

In critic mode, this loss function expects the `real_pred` given by the critic and some `input` (the noise fed to the generator). It will evaluate the critic using `crit_loss_func`. This loss function has the following signature
``` 
def crit_loss_func(real_pred, fake_pred):
```
where `real_pred` is the output of the critic on a batch of real images and `fake_pred` is generated from the noise using the generator.


<h3 id="AdaptiveLoss" class="doc_header"><code>class</code> <code>AdaptiveLoss</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L116" class="source_link" style="float:right">[source]</a></h3>

> <code>AdaptiveLoss</code>(**`crit`**) :: [`Module`](/torch_core.html#Module)

Expand the `target` to match the `output` size before applying `crit`.



<h4 id="accuracy_thresh_expand" class="doc_header"><code>accuracy_thresh_expand</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L123" class="source_link" style="float:right">[source]</a></h4>

> <code>accuracy_thresh_expand</code>(**`y_pred`**, **`y_true`**, **`thresh`**=*`0.5`*, **`sigmoid`**=*`True`*)

Compute accuracy after expanding `y_true` to the size of `y_pred`.


## Callbacks for GAN training


<h4 id="set_freeze_model" class="doc_header"><code>set_freeze_model</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L129" class="source_link" style="float:right">[source]</a></h4>

> <code>set_freeze_model</code>(**`m`**, **`rg`**)





<h3 id="GANTrainer" class="doc_header"><code>class</code> <code>GANTrainer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L133" class="source_link" style="float:right">[source]</a></h3>

> <code>GANTrainer</code>(**`switch_eval`**=*`False`*, **`clip`**=*`None`*, **`beta`**=*`0.98`*, **`gen_first`**=*`False`*, **`show_img`**=*`True`*) :: [`Callback`](/callback.core.html#Callback)

Handles GAN Training.


{% include warning.html content='The GANTrainer is useless on its own, you need to complete it with one of the following switchers' %}


<h3 id="FixedGANSwitcher" class="doc_header"><code>class</code> <code>FixedGANSwitcher</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L207" class="source_link" style="float:right">[source]</a></h3>

> <code>FixedGANSwitcher</code>(**`n_crit`**=*`1`*, **`n_gen`**=*`1`*) :: [`Callback`](/callback.core.html#Callback)

Switcher to do `n_crit` iterations of the critic then `n_gen` iterations of the generator.



<h3 id="AdaptiveGANSwitcher" class="doc_header"><code>class</code> <code>AdaptiveGANSwitcher</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L228" class="source_link" style="float:right">[source]</a></h3>

> <code>AdaptiveGANSwitcher</code>(**`gen_thresh`**=*`None`*, **`critic_thresh`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

Switcher that goes back to generator/critic when the loss goes below `gen_thresh`/`crit_thresh`.



<h3 id="GANDiscriminativeLR" class="doc_header"><code>class</code> <code>GANDiscriminativeLR</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L243" class="source_link" style="float:right">[source]</a></h3>

> <code>GANDiscriminativeLR</code>(**`mult_lr`**=*`5.0`*) :: [`Callback`](/callback.core.html#Callback)

[`Callback`](/callback.core.html#Callback) that handles multiplying the learning rate by `mult_lr` for the critic.


## GAN data


<h3 id="InvisibleTensor" class="doc_header"><code>class</code> <code>InvisibleTensor</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L258" class="source_link" style="float:right">[source]</a></h3>

> <code>InvisibleTensor</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorBase`](/torch_core.html#TensorBase)





<h4 id="generate_noise" class="doc_header"><code>generate_noise</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L262" class="source_link" style="float:right">[source]</a></h4>

> <code>generate_noise</code>(**`fn`**, **`size`**=*`100`*)




```python
bs = 128
size = 64
```

```python
dblock = DataBlock(blocks = (TransformBlock, ImageBlock),
                   get_x = generate_noise,
                   get_items = get_image_files,
                   splitter = IndexSplitter([]),
                   item_tfms=Resize(size, method=ResizeMethod.Crop), 
                   batch_tfms = Normalize.from_stats(torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])))
```

```python
path = untar_data(URLs.LSUN_BEDROOMS)
```

```python
dls = dblock.dataloaders(path, path=path, bs=bs)
```

```python
dls.show_batch(max_n=16)
```

## GAN Learner


<h4 id="gan_loss_from_func" class="doc_header"><code>gan_loss_from_func</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L279" class="source_link" style="float:right">[source]</a></h4>

> <code>gan_loss_from_func</code>(**`loss_gen`**, **`loss_crit`**, **`weights_gen`**=*`None`*)

Define loss functions for a GAN from `loss_gen` and `loss_crit`.



<h3 id="GANLearner" class="doc_header"><code>class</code> <code>GANLearner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/gan.py#L299" class="source_link" style="float:right">[source]</a></h3>

> <code>GANLearner</code>(**`dls`**, **`generator`**, **`critic`**, **`gen_loss_func`**, **`crit_loss_func`**, **`switcher`**=*`None`*, **`gen_first`**=*`False`*, **`switch_eval`**=*`True`*, **`show_img`**=*`True`*, **`clip`**=*`None`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`loss_func`**=*`None`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`splitter`**=*`trainable_params`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*, **`moms`**=*`(0.95, 0.85, 0.95)`*) :: [`Learner`](/learner.html#Learner)

A [`Learner`](/learner.html#Learner) suitable for GANs.


```python
from fastai.callback.all import *
```

```python
generator = basic_generator(64, n_channels=3, n_extra_layers=1)
critic    = basic_critic   (64, n_channels=3, n_extra_layers=1, act_cls=partial(nn.LeakyReLU, negative_slope=0.2))
```

```python
learn = GANLearner.wgan(dls, generator, critic, opt_func = RMSProp)
```

```python
learn.recorder.train_metrics=True
learn.recorder.valid_metrics=False
```

```python
learn.fit(1, 2e-4, wd=0.)
```

```python
learn.show_results(max_n=9, ds_idx=0)
```
