# Core vision
> Basic image opening/processing functionality


```python
from fastai.data.external import *
```

```python
#TODO: investigate
```

## Helpers

```python
im = Image.open(TEST_IMAGE).resize((30,20))
```


<h4 id="n_px" class="doc_header"><code>n_px</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L30" class="source_link" style="float:right">[source]</a></h4>

> <code>n_px</code>(**`x`**:`Image`)




#### `Image.n_px`

> `Image.n_px` (property)

Number of pixels in image

```python
test_eq(im.n_px, 30*20)
```


<h4 id="shape" class="doc_header"><code>shape</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L34" class="source_link" style="float:right">[source]</a></h4>

> <code>shape</code>(**`x`**:`Image`)




#### `Image.shape`
> `Image.shape` (property)

Image (height,width) tuple (NB:opposite order of `Image.size()`, same order as numpy array and pytorch tensor)

```python
test_eq(im.shape, (20,30))
```


<h4 id="aspect" class="doc_header"><code>aspect</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L38" class="source_link" style="float:right">[source]</a></h4>

> <code>aspect</code>(**`x`**:`Image`)




#### `Image.aspect`

> `Image.aspect` (property)

Aspect ratio of the image, i.e. `width/height`

```python
test_eq(im.aspect, 30/20)
```


<h4 id="Image.reshape" class="doc_header"><code>Image.reshape</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L42" class="source_link" style="float:right">[source]</a></h4>

> <code>Image.reshape</code>(**`x`**:`Image`, **`h`**, **`w`**, **`resample`**=*`0`*)

`resize` `x` to `(w,h)`


```python
test_eq(im.reshape(12,10).shape, (12,10))
```


<h4 id="Image.to_bytes_format" class="doc_header"><code>Image.to_bytes_format</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L48" class="source_link" style="float:right">[source]</a></h4>

> <code>Image.to_bytes_format</code>(**`im`**:`Image`, **`format`**=*`'png'`*)

Convert to bytes, default to PNG format



<h4 id="Image.to_thumb" class="doc_header"><code>Image.to_thumb</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L56" class="source_link" style="float:right">[source]</a></h4>

> <code>Image.to_thumb</code>(**`h`**, **`w`**=*`None`*)

Same as `thumbnail`, but uses a copy


```python
test_eq(im.resize_max(max_px=20*30).shape, (20,30))
test_eq(im.resize_max(max_px=300).n_px, 294)
test_eq(im.resize_max(max_px=500, max_h=10, max_w=20).shape, (10,15))
test_eq(im.resize_max(max_h=14, max_w=15).shape, (10,15))
test_eq(im.resize_max(max_px=300, max_h=10, max_w=25).shape, (10,15))
```


<h4 id="Image.resize_max" class="doc_header"><code>Image.resize_max</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L65" class="source_link" style="float:right">[source]</a></h4>

> <code>Image.resize_max</code>(**`x`**:`Image`, **`resample`**=*`0`*, **`max_px`**=*`None`*, **`max_h`**=*`None`*, **`max_w`**=*`None`*)

`resize` `x` to `max_px`, or `max_h`, or `max_w`


## Basic types

This section regroups the basic types used in vision with the transform that create objects of those types.


<h4 id="to_image" class="doc_header"><code>to_image</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L75" class="source_link" style="float:right">[source]</a></h4>

> <code>to_image</code>(**`x`**)

Convert a tensor or array to a PIL int8 Image



<h4 id="load_image" class="doc_header"><code>load_image</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L83" class="source_link" style="float:right">[source]</a></h4>

> <code>load_image</code>(**`fn`**, **`mode`**=*`None`*)

Open and load a `PIL.Image` and convert to `mode`



<h4 id="image2tensor" class="doc_header"><code>image2tensor</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L91" class="source_link" style="float:right">[source]</a></h4>

> <code>image2tensor</code>(**`img`**)

Transform image to byte tensor in `c*h*w` dim order.



<h3 id="PILBase" class="doc_header"><code>class</code> <code>PILBase</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L98" class="source_link" style="float:right">[source]</a></h3>

> <code>PILBase</code>() :: `Image`

This class represents an image object.  To create
:py:class:`~PIL.Image.Image` objects, use the appropriate factory
functions.  There's hardly ever any reason to call the Image constructor
directly.

* :py:func:`~PIL.Image.open`
* :py:func:`~PIL.Image.new`
* :py:func:`~PIL.Image.frombytes`



<h3 id="PILImage" class="doc_header"><code>class</code> <code>PILImage</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L119" class="source_link" style="float:right">[source]</a></h3>

> <code>PILImage</code>() :: [`PILBase`](/vision.core.html#PILBase)

This class represents an image object.  To create
:py:class:`~PIL.Image.Image` objects, use the appropriate factory
functions.  There's hardly ever any reason to call the Image constructor
directly.

* :py:func:`~PIL.Image.open`
* :py:func:`~PIL.Image.new`
* :py:func:`~PIL.Image.frombytes`



<h3 id="PILImageBW" class="doc_header"><code>class</code> <code>PILImageBW</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L122" class="source_link" style="float:right">[source]</a></h3>

> <code>PILImageBW</code>() :: [`PILImage`](/vision.core.html#PILImage)

This class represents an image object.  To create
:py:class:`~PIL.Image.Image` objects, use the appropriate factory
functions.  There's hardly ever any reason to call the Image constructor
directly.

* :py:func:`~PIL.Image.open`
* :py:func:`~PIL.Image.new`
* :py:func:`~PIL.Image.frombytes`


```python
im = PILImage.create(TEST_IMAGE)
test_eq(type(im), PILImage)
test_eq(im.mode, 'RGB')
test_eq(str(im), 'PILImage mode=RGB size=1200x803')
```

```python
im.resize((64,64))
```




![png](output_46_0.png)



```python
ax = im.show(figsize=(1,1))
```


![png](output_47_0.png)


```python
test_fig_exists(ax)
```

```python
timg = TensorImage(image2tensor(im))
tpil = PILImage.create(timg)
```

```python
tpil.resize((64,64))
```




![png](output_50_0.png)




<h3 id="PILMask" class="doc_header"><code>class</code> <code>PILMask</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L125" class="source_link" style="float:right">[source]</a></h3>

> <code>PILMask</code>() :: [`PILBase`](/vision.core.html#PILBase)

This class represents an image object.  To create
:py:class:`~PIL.Image.Image` objects, use the appropriate factory
functions.  There's hardly ever any reason to call the Image constructor
directly.

* :py:func:`~PIL.Image.open`
* :py:func:`~PIL.Image.new`
* :py:func:`~PIL.Image.frombytes`


```python
im = PILMask.create(TEST_IMAGE)
test_eq(type(im), PILMask)
test_eq(im.mode, 'L')
test_eq(str(im), 'PILMask mode=L size=1200x803')
```

### Images

```python
mnist = untar_data(URLs.MNIST_TINY)
fns = get_image_files(mnist)
mnist_fn = TEST_IMAGE_BW
```

```python
timg = Transform(PILImageBW.create)
mnist_img = timg(mnist_fn)
test_eq(mnist_img.size, (28,28))
assert isinstance(mnist_img, PILImageBW)
mnist_img
```




![png](output_57_0.png)



### Segmentation masks


<h3 id="AddMaskCodes" class="doc_header"><code>class</code> <code>AddMaskCodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L133" class="source_link" style="float:right">[source]</a></h3>

> <code>AddMaskCodes</code>(**`codes`**=*`None`*) :: [`Transform`](https://fastcore.fast.ai/transform#Transform)

Add the code metadata to a [`TensorMask`](/torch_core.html#TensorMask)


```python
camvid = untar_data(URLs.CAMVID_TINY)
fns = get_image_files(camvid/'images')
cam_fn = fns[0]
mask_fn = camvid/'labels'/f'{cam_fn.stem}_P{cam_fn.suffix}'
```

```python
cam_img = PILImage.create(cam_fn)
test_eq(cam_img.size, (128,96))
tmask = Transform(PILMask.create)
mask = tmask(mask_fn)
test_eq(type(mask), PILMask)
test_eq(mask.size, (128,96))
```

```python
_,axs = plt.subplots(1,3, figsize=(12,3))
cam_img.show(ctx=axs[0], title='image')
mask.show(alpha=1, ctx=axs[1], vmin=1, vmax=30, title='mask')
cam_img.show(ctx=axs[2], title='superimposed')
mask.show(ctx=axs[2], vmin=1, vmax=30);
```


![png](output_63_0.png)


### Points


<h3 id="TensorPoint" class="doc_header"><code>class</code> <code>TensorPoint</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L144" class="source_link" style="float:right">[source]</a></h3>

> <code>TensorPoint</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorBase`](/torch_core.html#TensorBase)

Basic type for points in an image


Points are expected to come as an array/tensor of shape `(n,2)` or as a list of lists with two elements. Unless you change the defaults in [`PointScaler`](/vision.core.html#PointScaler) (see later on), coordinates should go from 0 to width/height, with the first one being the column index (so from 0 to width) and the second one being the row index (so from 0 to height).
{% include note.html content='This is different from the usual indexing convention for arrays in numpy or in PyTorch, but it&#8217;s the way points are expected by matplotlib or the internal functions in PyTorch like `F.grid_sample`.' %}

```python
pnt_img = TensorImage(mnist_img.resize((28,35)))
pnts = np.array([[0,0], [0,35], [28,0], [28,35], [9, 17]])
tfm = Transform(TensorPoint.create)
tpnts = tfm(pnts)
test_eq(tpnts.shape, [5,2])
test_eq(tpnts.dtype, torch.float32)
```

```python
ctx = pnt_img.show(figsize=(1,1), cmap='Greys')
tpnts.show(ctx=ctx);
```


![png](output_70_0.png)


### Bounding boxes


<h4 id="get_annotations" class="doc_header"><code>get_annotations</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L165" class="source_link" style="float:right">[source]</a></h4>

> <code>get_annotations</code>(**`fname`**, **`prefix`**=*`None`*)

Open a COCO style json in `fname` and returns the lists of filenames (with maybe `prefix`) and labelled bboxes.



<h3 id="TensorBBox" class="doc_header"><code>class</code> <code>TensorBBox</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L195" class="source_link" style="float:right">[source]</a></h3>

> <code>TensorBBox</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorPoint`](/vision.core.html#TensorPoint)

Basic type for a tensor of bounding boxes in an image


Bounding boxes are expected to come as tuple with an array/tensor of shape `(n,4)` or as a list of lists with four elements and a list of corresponding labels. Unless you change the defaults in [`PointScaler`](/vision.core.html#PointScaler) (see later on), coordinates for each bounding box should go from 0 to width/height, with the following convention: x1, y1, x2, y2 where (x1,y1) is your top-left corner and (x2,y2) is your bottom-right corner.
{% include note.html content='We use the same convention as for points with x going from 0 to width and y going from 0 to height.' %}


<h3 id="LabeledBBox" class="doc_header"><code>class</code> <code>LabeledBBox</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L206" class="source_link" style="float:right">[source]</a></h3>

> <code>LabeledBBox</code>(**`items`**=*`None`*, **\*`rest`**, **`use_list`**=*`False`*, **`match`**=*`None`*) :: [`L`](https://fastcore.fast.ai/foundation#L)

Basic type for a list of bounding boxes in an image


```python
coco = untar_data(URLs.COCO_TINY)
images, lbl_bbox = get_annotations(coco/'train.json')
idx=2
coco_fn,bbox = coco/'train'/images[idx],lbl_bbox[idx]
coco_img = timg(coco_fn)
```

```python
tbbox = LabeledBBox(TensorBBox(bbox[0]), bbox[1])
ctx = coco_img.show(figsize=(3,3), cmap='Greys')
tbbox.show(ctx=ctx);
```


![png](output_82_0.png)


## Basic Transforms

Unless specifically mentioned, all the following transforms can be used as single-item transforms (in one of the list in the `tfms` you pass to a `TfmdDS` or a `Datasource`) or tuple transforms (in the `tuple_tfms` you pass to a `TfmdDS` or a `Datasource`). The safest way that will work across applications is to always use them as `tuple_tfms`. For instance, if you have points or bounding boxes as targets and use [`Resize`](/vision.augment.html#Resize) as a single-item transform, when you get to [`PointScaler`](/vision.core.html#PointScaler) (which is a tuple transform) you won't have the correct size of the image to properly scale your points.


<h4 id="encodes" class="doc_header"><code>encodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L281" class="source_link" style="float:right">[source]</a></h4>

> <code>encodes</code>(**`x`**:[`TensorBBox`](/vision.core.html#TensorBBox))





<h4 id="encodes" class="doc_header"><code>encodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L281" class="source_link" style="float:right">[source]</a></h4>

> <code>encodes</code>(**`x`**:[`TensorBBox`](/vision.core.html#TensorBBox))




Any data augmentation transform that runs on PIL Images must be run before this transform.

```python
tfm = ToTensor()
print(tfm)
print(type(mnist_img))
print(type(tfm(mnist_img)))
```

    ToTensor:
    (PILMask,object) -> encodes
    (PILBase,object) -> encodes
     
    <class '__main__.PILImageBW'>
    <class 'fastai.torch_core.TensorImageBW'>


```python
tfm = ToTensor()
test_eq(tfm(mnist_img).shape, (1,28,28))
test_eq(type(tfm(mnist_img)), TensorImageBW)
test_eq(tfm(mask).shape, (96,128))
test_eq(type(tfm(mask)), TensorMask)
```

Let's confirm we can pipeline this with [`PILImage.create`](/vision.core.html#PILImage.create).

```python
pipe_img = Pipeline([PILImageBW.create, ToTensor()])
img = pipe_img(mnist_fn)
test_eq(type(img), TensorImageBW)
pipe_img.show(img, figsize=(1,1));
```


![png](output_93_0.png)


```python
def _cam_lbl(x): return mask_fn
cam_tds = Datasets([cam_fn], [[PILImage.create, ToTensor()], [_cam_lbl, PILMask.create, ToTensor()]])
show_at(cam_tds, 0);
```


![png](output_94_0.png)


To work with data augmentation, and in particular the `grid_sample` method, points need to be represented with coordinates going from -1 to 1 (-1 being top or left, 1 bottom or right), which will be done unless you pass `do_scale=False`. We also need to make sure they are following our convention of points being x,y coordinates, so pass along `y_first=True` if you have your data in an y,x format to add a flip.
{% include warning.html content='This transform needs to run on the tuple level, before any transform that changes the image size.' %}


<h3 id="PointScaler" class="doc_header"><code>class</code> <code>PointScaler</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L235" class="source_link" style="float:right">[source]</a></h3>

> <code>PointScaler</code>(**`do_scale`**=*`True`*, **`y_first`**=*`False`*) :: [`Transform`](https://fastcore.fast.ai/transform#Transform)

Scale a tensor representing points


To work with data augmentation, and in particular the `grid_sample` method, points need to be represented with coordinates going from -1 to 1 (-1 being top or left, 1 bottom or right), which will be done unless you pass `do_scale=False`. We also need to make sure they are following our convention of points being x,y coordinates, so pass along `y_first=True` if you have your data in an y,x format to add a flip.
{% include note.html content='This transform automatically grabs the sizes of the images it sees before a <code>TensorPoint</code> object and embeds it in them. For this to work, those images need to be before any points in the order of your final tuple. If you don&#8217;t have such images, you need to embed the size of the corresponding image when creating a <code>TensorPoint</code> by passing it with `sz=...`.' %}

```python
def _pnt_lbl(x): return TensorPoint.create(pnts)
def _pnt_open(fn): return PILImage(PILImage.create(fn).resize((28,35)))
pnt_tds = Datasets([mnist_fn], [_pnt_open, [_pnt_lbl]])
pnt_tdl = TfmdDL(pnt_tds, bs=1, after_item=[PointScaler(), ToTensor()])
```

```python
test_eq(pnt_tdl.after_item.c, 10)
```

```python
x,y = pnt_tdl.one_batch()
#Scaling and flipping properly done
#NB: we added a point earlier at (9,17); formula below scales to (-1,1) coords
test_close(y[0], tensor([[-1., -1.], [-1.,  1.], [1.,  -1.], [1., 1.], [9/14-1, 17/17.5-1]]))
a,b = pnt_tdl.decode_batch((x,y))[0]
test_eq(b, tensor(pnts).float())
#Check types
test_eq(type(x), TensorImage)
test_eq(type(y), TensorPoint)
test_eq(type(a), TensorImage)
test_eq(type(b), TensorPoint)
test_eq(b.get_meta('img_size'), (28,35)) #Automatically picked the size of the input
```

```python
pnt_tdl.show_batch(figsize=(2,2), cmap='Greys');
```


![png](output_103_0.png)



<h3 id="BBoxLabeler" class="doc_header"><code>class</code> <code>BBoxLabeler</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L260" class="source_link" style="float:right">[source]</a></h3>

> <code>BBoxLabeler</code>(**`enc`**=*`None`*, **`dec`**=*`None`*, **`split_idx`**=*`None`*, **`order`**=*`None`*) :: [`Transform`](https://fastcore.fast.ai/transform#Transform)

Delegates (`__call__`,`decode`,`setup`) to ([`encodes`](/tabular.core.html#encodes),[`decodes`](/tabular.core.html#decodes),[`setups`](/tabular.core.html#setups)) if `split_idx` matches



<h4 id="decodes" class="doc_header"><code>decodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L286" class="source_link" style="float:right">[source]</a></h4>

> <code>decodes</code>(**`x`**:[`TensorBBox`](/vision.core.html#TensorBBox))





<h4 id="encodes" class="doc_header"><code>encodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L281" class="source_link" style="float:right">[source]</a></h4>

> <code>encodes</code>(**`x`**:[`TensorBBox`](/vision.core.html#TensorBBox))





<h4 id="decodes" class="doc_header"><code>decodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L286" class="source_link" style="float:right">[source]</a></h4>

> <code>decodes</code>(**`x`**:[`TensorBBox`](/vision.core.html#TensorBBox))




```python
def _coco_bb(x):  return TensorBBox.create(bbox[0])
def _coco_lbl(x): return bbox[1]

coco_tds = Datasets([coco_fn], [PILImage.create, [_coco_bb], [_coco_lbl, MultiCategorize(add_na=True)]], n_inp=1)
coco_tdl = TfmdDL(coco_tds, bs=1, after_item=[BBoxLabeler(), PointScaler(), ToTensor()])
```

```python
Categorize(add_na=True)
```




    Categorize: {'vocab': None, 'add_na': True}:
    (object,object) -> encodes
     (object,object) -> decodes



```python
coco_tds.tfms
```




    (#3) [Pipeline: PILBase.create,Pipeline: _coco_bb,Pipeline: _coco_lbl -> MultiCategorize: {'vocab': (#2) ['#na#','vase'], 'add_na': True}]



```python
x,y,z
```




    (PILImage mode=RGB size=128x128,
     TensorBBox([[-0.9011, -0.4606,  0.1416,  0.6764],
             [ 0.2000, -0.2405,  1.0000,  0.9102],
             [ 0.4909, -0.9325,  0.9284, -0.5011]]),
     TensorMultiCategory([1, 1, 1]))



```python
x,y,z = coco_tdl.one_batch()
test_close(y[0], -1+tensor(bbox[0])/64)
test_eq(z[0], tensor([1,1,1]))
a,b,c = coco_tdl.decode_batch((x,y,z))[0]
test_close(b, tensor(bbox[0]).float())
test_eq(c.bbox, b)
test_eq(c.lbl, bbox[1])

#Check types
test_eq(type(x), TensorImage)
test_eq(type(y), TensorBBox)
test_eq(type(z), TensorMultiCategory)
test_eq(type(a), TensorImage)
test_eq(type(b), TensorBBox)
test_eq(type(c), LabeledBBox)
test_eq(y.get_meta('img_size'), (128,128))
```

```python
coco_tdl.show_batch();
```


![png](output_116_0.png)

