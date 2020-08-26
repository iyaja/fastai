# CutMix Callback
> Callback to apply <a href='https://arxiv.org/pdf/1905.04899.pdf'>CutMix</a> data augmentation technique to the training data.


From the [research paper](https://arxiv.org/pdf/1905.04899.pdf), [`CutMix`](/callback.cutmix.html#CutMix) is a way to combine two images. It comes from [`MixUp`](/callback.mixup.html#MixUp) and `Cutout`. In this data augmentation technique:
> patches are cut and pasted among training images where the ground truth labels are also mixed proportionally to the area of the patches

Also, from the paper:> By making efficient use of training pixels and retaining the regularization effect of regional dropout, CutMix consistently outperforms the state-of-the-art augmentation strategies on CIFAR and ImageNet classification tasks, as well as on the ImageNet weakly-supervised localization task. Moreover, unlike previous augmentation methods, our CutMix-trained ImageNet classifier, when used as a pretrained model, results in consistent performance gains in Pascal detection and MS-COCO image captioning benchmarks. We also show that CutMix improves the model robustness against input corruptions and its out-of-distribution detection performances. 


<h2 id="CutMix" class="doc_header"><code>class</code> <code>CutMix</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/cutmix.py#L10" class="source_link" style="float:right">[source]</a></h2>

> <code>CutMix</code>(**`alpha`**=*`1.0`*) :: [`Callback`](/callback.core.html#Callback)

Implementation of `https://arxiv.org/abs/1905.04899`


## How does the batch with [`CutMix`](/callback.cutmix.html#CutMix) data augmentation technique look like?

First, let's quickly create the `dls` using [`ImageDataLoaders.from_name_re`](/vision.data.html#ImageDataLoaders.from_name_re) DataBlocks API.

```python
path = untar_data(URLs.PETS)
pat        = r'([^/]+)_\d+.*$'
fnames     = get_image_files(path/'images')
item_tfms  = [Resize(256, method='crop')]
batch_tfms = [*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
dls = ImageDataLoaders.from_name_re(path, fnames, pat, bs=64, item_tfms=item_tfms, 
                                    batch_tfms=batch_tfms)
```

Next, let's initialize the callback [`CutMix`](/callback.cutmix.html#CutMix), create a learner, do one batch and display the images with the labels. [`CutMix`](/callback.cutmix.html#CutMix) inside updates the loss function based on the ratio of the cutout bbox to the complete image.

```python
cutmix = CutMix(alpha=1.)
```

```python
with Learner(dls, resnet18(), loss_func=CrossEntropyLossFlat(), cbs=cutmix) as learn:
    learn.epoch,learn.training = 0,True
    learn.dl = dls.train
    b = dls.one_batch()
    learn._split(b)
    learn('before_batch')

_,axs = plt.subplots(3,3, figsize=(9,9))
dls.show_batch(b=(cutmix.x,cutmix.y), ctxs=axs.flatten())
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



![png](output_10_1.png)


## Using [`CutMix`](/callback.cutmix.html#CutMix) in Training

```python
learn = cnn_learner(dls, resnet18, loss_func=CrossEntropyLossFlat(), cbs=cutmix, metrics=[accuracy, error_rate])
# learn.fit_one_cycle(1)
```
