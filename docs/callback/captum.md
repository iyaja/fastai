# Captum




<h4 id="json_clean" class="doc_header"><code>json_clean</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/captum.py#L16" class="source_link" style="float:right">[source]</a></h4>

> <code>json_clean</code>(**`o`**)




In all this notebook, we will use the following data:

```python
from fastai.vision.all import *
```

```python
path = untar_data(URLs.PETS)/'images'
fnames = get_image_files(path)
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, fnames, valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(128))
```

```python
from random import randint
```

```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.382507</td>
      <td>0.125592</td>
      <td>0.043978</td>
      <td>00:13</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.146569</td>
      <td>0.060744</td>
      <td>0.025710</td>
      <td>00:13</td>
    </tr>
  </tbody>
</table>


# Captum Interpretation

The Distill Article [here](https://distill.pub/2020/attribution-baselines/) provides a good overview of what baseline image to choose. We can try them one by one.


<h2 id="CaptumInterpretation" class="doc_header"><code>class</code> <code>CaptumInterpretation</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/captum.py#L33" class="source_link" style="float:right">[source]</a></h2>

> <code>CaptumInterpretation</code>(**`learn`**, **`cmap_name`**=*`'custom blue'`*, **`colors`**=*`None`*, **`N`**=*`256`*, **`methods`**=*`['original_image', 'heat_map']`*, **`signs`**=*`['all', 'positive']`*, **`outlier_perc`**=*`1`*)

Captum Interpretation for Resnet


## Interpretation

```python
captum=CaptumInterpretation(learn)
idx=randint(0,len(fnames))
captum.visualize(fnames[idx])
```


![png](output_16_0.png)


```python
captum.visualize(fnames[idx],baseline_type='uniform')
```


![png](output_17_0.png)


```python
captum.visualize(fnames[idx],baseline_type='gauss')
```


![png](output_18_0.png)


```python
captum.visualize(fnames[idx],metric='NT',baseline_type='uniform')
```


![png](output_19_0.png)


```python
captum.visualize(fnames[idx],metric='Occl',baseline_type='gauss')
```


![png](output_20_0.png)


## Captum Insights Callback

```python
@patch
def _formatted_data_iter(x: CaptumInterpretation,dl,normalize_func):
    dl_iter=iter(dl)
    while True:
        images,labels=next(dl_iter)
        images=normalize_func.decode(images).to(dl.device)
        yield Batch(inputs=images, labels=labels)
```


<h4 id="CaptumInterpretation.insights" class="doc_header"><code>CaptumInterpretation.insights</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/captum.py#L104" class="source_link" style="float:right">[source]</a></h4>

> <code>CaptumInterpretation.insights</code>(**`x`**:[`CaptumInterpretation`](/callback.captum.html#CaptumInterpretation), **`inp_data`**, **`debug`**=*`True`*)




```python
captum=CaptumInterpretation(learn)
captum.insights(fnames)
```
