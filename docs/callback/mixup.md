# Mixup callback
> Callback to apply MixUp data augmentation to your training



<h4 id="reduce_loss" class="doc_header"><code>reduce_loss</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/mixup.py#L14" class="source_link" style="float:right">[source]</a></h4>

> <code>reduce_loss</code>(**`loss`**, **`reduction`**=*`'mean'`*)





<h2 id="MixUp" class="doc_header"><code>class</code> <code>MixUp</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/mixup.py#L19" class="source_link" style="float:right">[source]</a></h2>

> <code>MixUp</code>(**`alpha`**=*`0.4`*) :: [`Callback`](/callback.core.html#Callback)

Basic class handling tweaks of the training loop by changing a [`Learner`](/learner.html#Learner) in various events


```python
from fastai.vision.core import *
```

```python
path = untar_data(URLs.MNIST_TINY)
items = get_image_files(path)
tds = Datasets(items, [PILImageBW.create, [parent_label, Categorize()]], splits=GrandparentSplitter()(items))
dls = tds.dataloaders(after_item=[ToTensor(), IntToFloatTensor()])
```

```python
mixup = MixUp(0.5)
with Learner(dls, nn.Linear(3,4), loss_func=CrossEntropyLossFlat(), cbs=mixup) as learn:
    learn.epoch,learn.training = 0,True
    learn.dl = dls.train
    b = dls.one_batch()
    learn._split(b)
    learn('before_batch')

_,axs = plt.subplots(3,3, figsize=(9,9))
dls.show_batch(b=(mixup.x,mixup.y), ctxs=axs.flatten())
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



![png](output_8_1.png)

