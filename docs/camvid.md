# Title



```python
import torch
```

```python
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
```

```python
path = untar_data(URLs.CAMVID)
```

```python
valid_fnames = (path/'valid.txt').read().split('\n')
```

```python
def ListSplitter(valid_items):
    def _inner(items):
        val_mask = tensor([o.name in valid_items for o in items])
        return [~val_mask,val_mask]
    return _inner
```

```python
codes = np.loadtxt(path/'codes.txt', dtype=str)
```

```python
camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=ListSplitter(valid_fnames),
                   get_y=lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
                   batch_tfms=[*aug_transforms(size=(360,480)), Normalize.from_stats(*imagenet_stats)])

dls = camvid.dataloaders(path/"images", bs=8)
```

```python
dls = SegmentationDataLoaders.from_label_func(path, bs=8,
    fnames = get_image_files(path/"images"), 
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',                                     
    codes = codes,                         
    batch_tfms=[*aug_transforms(size=(360,480)), Normalize.from_stats(*imagenet_stats)])
```

```python
dls.show_batch(max_n=2, rows=1, figsize=(20, 7))
```


![png](output_9_0.png)


```python
dls.show_batch(max_n=4, figsize=(20, 14))
```


![png](output_10_0.png)


```python
codes = np.loadtxt(path/'codes.txt', dtype=str)
dls.vocab = codes
```

```python
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
```

```python
opt_func = partial(Adam, lr=3e-3, wd=0.01)#, eps=1e-8)

learn = unet_learner(dls, resnet34, loss_func=CrossEntropyLossFlat(axis=1), opt_func=opt_func, path=path, metrics=acc_camvid,
                     config = unet_config(norm_type=None), wd_bn_bias=True)
```

```python
get_c(dls)
```




    32



```python
learn.lr_find()
```






![png](output_15_1.png)


```python
lr= 3e-3
learn.freeze()
```

```python
learn.fit_one_cycle(10, slice(lr), pct_start=0.9, wd=1e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>acc_camvid</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.200469</td>
      <td>0.869627</td>
      <td>0.769983</td>
      <td>00:57</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.840649</td>
      <td>0.809244</td>
      <td>0.776909</td>
      <td>00:47</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.716685</td>
      <td>0.638332</td>
      <td>0.838415</td>
      <td>00:47</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.670508</td>
      <td>0.551083</td>
      <td>0.851559</td>
      <td>00:47</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.664709</td>
      <td>0.588863</td>
      <td>0.849711</td>
      <td>00:47</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.603191</td>
      <td>0.502482</td>
      <td>0.867659</td>
      <td>00:47</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.592773</td>
      <td>0.507730</td>
      <td>0.869631</td>
      <td>00:47</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.541870</td>
      <td>0.540163</td>
      <td>0.863005</td>
      <td>00:47</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.531527</td>
      <td>0.429516</td>
      <td>0.878525</td>
      <td>00:47</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.463456</td>
      <td>0.345390</td>
      <td>0.900292</td>
      <td>00:47</td>
    </tr>
  </tbody>
</table>


```python
learn.show_results(max_n=2, rows=2, vmin=1, vmax=30, figsize=(14, 10))
```






![png](output_18_1.png)


```python
learn.save('stage-1')
```

```python
learn.load('stage-1')
learn.unfreeze()
```

```python

```

```python
lrs = slice(lr/400,lr/4)
```

```python
learn.fit_one_cycle(12, lrs, pct_start=0.8, wd=1e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>acc_camvid</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.415170</td>
      <td>0.350871</td>
      <td>0.897328</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.405012</td>
      <td>0.341905</td>
      <td>0.899924</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.400426</td>
      <td>0.330662</td>
      <td>0.904413</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.385431</td>
      <td>0.329282</td>
      <td>0.904444</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.372985</td>
      <td>0.322414</td>
      <td>0.912512</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.366623</td>
      <td>0.306477</td>
      <td>0.916740</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.362156</td>
      <td>0.298581</td>
      <td>0.913030</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.343045</td>
      <td>0.290931</td>
      <td>0.919178</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.327369</td>
      <td>0.295092</td>
      <td>0.921611</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.334783</td>
      <td>0.280629</td>
      <td>0.922483</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.295628</td>
      <td>0.260418</td>
      <td>0.929844</td>
      <td>00:42</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.269825</td>
      <td>0.260967</td>
      <td>0.928652</td>
      <td>00:42</td>
    </tr>
  </tbody>
</table>


```python
learn.show_results(max_n=4, vmin=1, vmax=30, figsize=(15,6))
```






![png](output_24_1.png)

