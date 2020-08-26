# Learner for the vision applications
> All the functions necessary to build `Learner` suitable for transfer learning in computer vision


The most important functions of this module are [`cnn_learner`](/vision.learner.html#cnn_learner) and [`unet_learner`](/vision.learner.html#unet_learner). They will help you define a [`Learner`](/learner.html#Learner) using a pretrained model. See the [vision tutorial](http://docs.fast.ai/tutorial.vision) for examples of use.

## Cut a pretrained model

By default, the fastai library cuts a pretrained model at the pooling layer. This function helps detecting it. 


<h4 id="has_pool_type" class="doc_header"><code>has_pool_type</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/learner.py#L17" class="source_link" style="float:right">[source]</a></h4>

> <code>has_pool_type</code>(**`m`**)

Return `True` if `m` is a pooling layer or has one in its children


```python
m = nn.Sequential(nn.AdaptiveAvgPool2d(5), nn.Linear(2,3), nn.Conv2d(2,3,1), nn.MaxPool3d(5))
assert has_pool_type(m)
test_eq([has_pool_type(m_) for m_ in m.children()], [True,False,False,True])
```


<h4 id="create_body" class="doc_header"><code>create_body</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/learner.py#L63" class="source_link" style="float:right">[source]</a></h4>

> <code>create_body</code>(**`arch`**, **`n_in`**=*`3`*, **`pretrained`**=*`True`*, **`cut`**=*`None`*)

Cut off the body of a typically pretrained `arch` as determined by `cut`


`cut` can either be an integer, in which case we cut the model at the corresponding layer, or a function, in which case, this function returns `cut(model)`. It defaults to the first layer that contains some pooling otherwise.

```python
tst = lambda pretrained : nn.Sequential(nn.Conv2d(3,5,3), nn.BatchNorm2d(5), nn.AvgPool2d(1), nn.Linear(3,4))
m = create_body(tst)
test_eq(len(m), 2)

m = create_body(tst, cut=3)
test_eq(len(m), 3)

m = create_body(tst, cut=noop)
test_eq(len(m), 4)

for n in range(1,5):    
    m = create_body(tst, n_in=n)
    test_eq(_get_first_layer(m)[0].in_channels, n)
```

## Head and model


<h4 id="create_head" class="doc_header"><code>create_head</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/learner.py#L76" class="source_link" style="float:right">[source]</a></h4>

> <code>create_head</code>(**`nf`**, **`n_out`**, **`lin_ftrs`**=*`None`*, **`ps`**=*`0.5`*, **`concat_pool`**=*`True`*, **`bn_final`**=*`False`*, **`lin_first`**=*`False`*, **`y_range`**=*`None`*)

Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes.


The head begins with fastai's [`AdaptiveConcatPool2d`](/layers.html#AdaptiveConcatPool2d) if `concat_pool=True` otherwise, it uses traditional average pooling. Then it uses a [`Flatten`](/layers.html#Flatten) layer before going on blocks of [`BatchNorm`](/layers.html#BatchNorm), `Dropout` and `Linear` layers (if `lin_first=True`, those are `Linear`, [`BatchNorm`](/layers.html#BatchNorm), `Dropout`).

Those blocks start at `nf`, then every element of `lin_ftrs` (defaults to `[512]`) and end at `n_out`. `ps` is a list of probabilities used for the dropouts (if you only pass 1, it will use half the value then that value as many times as necessary).

If `bn_final=True`, a final [`BatchNorm`](/layers.html#BatchNorm) layer is added. If `y_range` is passed, the function adds a [`SigmoidRange`](/layers.html#SigmoidRange) to that range.

```python
tst = create_head(5, 10)
tst
```




    Sequential(
      (0): AdaptiveConcatPool2d(
        (ap): AdaptiveAvgPool2d(output_size=1)
        (mp): AdaptiveMaxPool2d(output_size=1)
      )
      (1): Flatten(full=False)
      (2): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Dropout(p=0.25, inplace=False)
      (4): Linear(in_features=5, out_features=512, bias=False)
      (5): ReLU(inplace=True)
      (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): Dropout(p=0.5, inplace=False)
      (8): Linear(in_features=512, out_features=10, bias=False)
    )




<h4 id="create_cnn_model" class="doc_header"><code>create_cnn_model</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/learner.py#L96" class="source_link" style="float:right">[source]</a></h4>

> <code>create_cnn_model</code>(**`arch`**, **`n_out`**, **`cut`**=*`None`*, **`pretrained`**=*`True`*, **`n_in`**=*`3`*, **`init`**=*`kaiming_normal_`*, **`custom_head`**=*`None`*, **`concat_pool`**=*`True`*, **`lin_ftrs`**=*`None`*, **`ps`**=*`0.5`*, **`bn_final`**=*`False`*, **`lin_first`**=*`False`*, **`y_range`**=*`None`*)

Create custom convnet architecture using `arch`, `n_in` and `n_out`


The model is cut according to `cut` and it may be `pretrained`, in which case, the proper set of weights is downloaded then loaded. `init` is applied to the head of the model, which is either created by [`create_head`](/vision.learner.html#create_head) (with `lin_ftrs`, `ps`, `concat_pool`, `bn_final`, `lin_first` and `y_range`) or is `custom_head`.

```python
tst = create_cnn_model(models.resnet18, 10, None, True)
tst = create_cnn_model(models.resnet18, 10, None, True, n_in=1)
```


<h4 id="cnn_config" class="doc_header"><code>cnn_config</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/learner.py#L110" class="source_link" style="float:right">[source]</a></h4>

> <code>cnn_config</code>(**`cut`**=*`None`*, **`pretrained`**=*`True`*, **`n_in`**=*`3`*, **`init`**=*`kaiming_normal_`*, **`custom_head`**=*`None`*, **`concat_pool`**=*`True`*, **`lin_ftrs`**=*`None`*, **`ps`**=*`0.5`*, **`bn_final`**=*`False`*, **`lin_first`**=*`False`*, **`y_range`**=*`None`*)

Convenience function to easily create a config for [`create_cnn_model`](/vision.learner.html#create_cnn_model)


```python
pets = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=RegexLabeller(pat = r'/([^/]+)_\d+.jpg$'))

dls = pets.dataloaders(untar_data(URLs.PETS)/"images", item_tfms=RandomResizedCrop(300, min_scale=0.5), bs=64,
                        batch_tfms=[*aug_transforms(size=224)])
```

    @log_args had an issue on DataLoader.__init__ -> got an unexpected keyword argument 'item_tfms'
    @log_args had an issue on TfmdDL.__init__ -> got an unexpected keyword argument 'item_tfms'


```python
# class ModelSplitter():
#     def __init__(self, idx): self.idx = idx
#     def split(self, m): return L(m[:self.idx], m[self.idx:]).map(params)
#     def __call__(self,): return {'cut':self.idx, 'split':self.split}
```


<h4 id="default_split" class="doc_header"><code>default_split</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/learner.py#L116" class="source_link" style="float:right">[source]</a></h4>

> <code>default_split</code>(**`m`**)

Default split of a model between body and head


To do transfer learning, you need to pass a `splitter` to [`Learner`](/learner.html#Learner). This should be a function taking the model and returning a collection of parameter groups, e.g. a list of list of parameters.

## [`Learner`](/learner.html#Learner) convenience functions


<h4 id="cnn_learner" class="doc_header"><code>cnn_learner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/learner.py#L164" class="source_link" style="float:right">[source]</a></h4>

> <code>cnn_learner</code>(**`dls`**, **`arch`**, **`loss_func`**=*`None`*, **`pretrained`**=*`True`*, **`cut`**=*`None`*, **`splitter`**=*`None`*, **`y_range`**=*`None`*, **`config`**=*`None`*, **`n_out`**=*`None`*, **`normalize`**=*`True`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*, **`moms`**=*`(0.95, 0.85, 0.95)`*)

Build a convnet style learner from `dls` and `arch`


The model is built from `arch` using the number of final activations inferred from `dls` if possible (otherwise pass a value to `n_out`). It might be `pretrained` and the architecture is cut and split using the default metadata of the model architecture (this can be customized by passing a `cut` or a `splitter`). 

To customize the model creation, use [`cnn_config`](/vision.learner.html#cnn_config) and pass the result to the `config` argument. There is just easy access to `y_range` because this argument is often used.

If `normalize` and `pretrained` are `True`, this function adds a `Normalization` transform to the `dls` (if there is not already one) using the statistics of the pretrained model. That way, you won't ever forget to normalize your data in transfer learning.

All other arguments are passed to [`Learner`](/learner.html#Learner).

```python
path = untar_data(URLs.PETS)
fnames = get_image_files(path/"images")
pat = r'^(.*)_\d+.jpg$'
dls = ImageDataLoaders.from_name_re(path, fnames, pat, item_tfms=Resize(224))
```

```python
learn = cnn_learner(dls, models.resnet34, loss_func=CrossEntropyLossFlat(), config=cnn_config(ps=0.25))
```


<h4 id="unet_config" class="doc_header"><code>unet_config</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/learner.py#L181" class="source_link" style="float:right">[source]</a></h4>

> <code>unet_config</code>(**`blur`**=*`False`*, **`blur_final`**=*`True`*, **`self_attention`**=*`False`*, **`y_range`**=*`None`*, **`last_cross`**=*`True`*, **`bottle`**=*`False`*, **`act_cls`**=*`ReLU`*, **`init`**=*`kaiming_normal_`*, **`norm_type`**=*`None`*)

Convenience function to easily create a config for [`DynamicUnet`](/vision.models.unet.html#DynamicUnet)



<h4 id="unet_learner" class="doc_header"><code>unet_learner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/learner.py#L187" class="source_link" style="float:right">[source]</a></h4>

> <code>unet_learner</code>(**`dls`**, **`arch`**, **`loss_func`**=*`None`*, **`pretrained`**=*`True`*, **`cut`**=*`None`*, **`splitter`**=*`None`*, **`config`**=*`None`*, **`n_in`**=*`3`*, **`n_out`**=*`None`*, **`normalize`**=*`True`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*, **`moms`**=*`(0.95, 0.85, 0.95)`*)

Build a unet learner from `dls` and `arch`


The model is built from `arch` using the number of final filters inferred from `dls` if possible (otherwise pass a value to `n_out`). It might be `pretrained` and the architecture is cut and split using the default metadata of the model architecture (this can be customized by passing a `cut` or a `splitter`). 

To customize the model creation, use [`unet_config`](/vision.learner.html#unet_config) and pass the result to the `config` argument. 

If `normalize` and `pretrained` are `True`, this function adds a `Normalization` transform to the `dls` (if there is not already one) using the statistics of the pretrained model. That way, you won't ever forget to normalize your data in transfer learning.

All other arguments are passed to [`Learner`](/learner.html#Learner).

```python
path = untar_data(URLs.CAMVID_TINY)
fnames = get_image_files(path/'images')
def label_func(x): return path/'labels'/f'{x.stem}_P{x.suffix}'
codes = np.loadtxt(path/'codes.txt', dtype=str)
    
dls = SegmentationDataLoaders.from_label_func(path, fnames, label_func, codes=codes)
```

```python
learn = unet_learner(dls, models.resnet34, loss_func=CrossEntropyLossFlat(axis=1))
```
