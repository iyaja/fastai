# Vision widgets
> ipywidgets for images



<h4 id="Box.__getitem__" class="doc_header"><code>Box.__getitem__</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/widgets.py#L16" class="source_link" style="float:right">[source]</a></h4>

> <code>Box.__getitem__</code>(**`i`**)





<h4 id="widget" class="doc_header"><code>widget</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/widgets.py#L20" class="source_link" style="float:right">[source]</a></h4>

> <code>widget</code>(**`im`**, **\*`args`**, **\*\*`layout`**)

Convert anything that can be `display`ed by IPython into a widget


```python
im = Image.open('images/puppy.jpg').to_thumb(256,512)
VBox([widgets.HTML('Puppy'),
      widget(im, max_width="192px")])
```


<h4 id="carousel" class="doc_header"><code>carousel</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/widgets.py#L32" class="source_link" style="float:right">[source]</a></h4>

> <code>carousel</code>(**`children`**=*`()`*, **\*\*`layout`**)

A horizontally scrolling carousel


```python
ts = [VBox([widget(im, max_width='192px'), Button(description='click')])
      for o in range(3)]

carousel(ts, width='450px')
```


<h2 id="ImagesCleaner" class="doc_header"><code>class</code> <code>ImagesCleaner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/widgets.py#L44" class="source_link" style="float:right">[source]</a></h2>

> <code>ImagesCleaner</code>(**`opts`**=*`()`*, **`height`**=*`128`*, **`width`**=*`256`*, **`max_n`**=*`30`*)

A widget that displays all images in `fns` along with a `Dropdown`


```python
fns = get_image_files('images')
w = ImagesCleaner(('A','B'))
w.set_fns(fns)
w
```

```python
w.delete(),w.change()
```




    ((#0) [], (#0) [])




<h2 id="ImageClassifierCleaner" class="doc_header"><code>class</code> <code>ImageClassifierCleaner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/widgets.py#L74" class="source_link" style="float:right">[source]</a></h2>

> <code>ImageClassifierCleaner</code>(**`learn`**, **`opts`**=*`()`*, **`height`**=*`128`*, **`width`**=*`256`*, **`max_n`**=*`30`*) :: [`GetAttr`](https://fastcore.fast.ai/foundation#GetAttr)

A widget that provides an [`ImagesCleaner`](/vision.widgets.html#ImagesCleaner) with a CNN [`Learner`](/learner.html#Learner)

