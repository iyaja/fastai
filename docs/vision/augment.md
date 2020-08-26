# Data augmentation in computer vision
> Transforms to apply data augmentation in Computer Vision


```python
img = PILImage(PILImage.create(TEST_IMAGE).resize((600,400)))
```


<h3 id="RandTransform" class="doc_header"><code>class</code> <code>RandTransform</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L21" class="source_link" style="float:right">[source]</a></h3>

> <code>RandTransform</code>(**`p`**=*`1.0`*, **`nm`**=*`None`*, **`before_call`**=*`None`*, **\*\*`kwargs`**) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

A transform that before_call its state at each `__call__`


As for all [`Transform`](https://fastcore.fast.ai/transform#Transform) you can pass <code>encodes</code> and <code>decodes</code> at init or subclass and implement them. You can do the same for the `before_call` method that is called at each `__call__`. Note that to have a consistent state for inputs and targets, a [`RandTransform`](/vision.augment.html#RandTransform) must be applied at the tuple level. 

By default the before_call behavior is to execute the transform with probability `p` (if subclassing and wanting to tweak that behavior, the attribute `self.do`, if it exists, is looked for to decide if the transform is executed or not).
{% include note.html content='A <code>RandTransform</code> is only applied to the training set by default, so you have to pass `split_idx=0` if you are calling it directly and not through a <code>Datasets</code>. That behavior can be changed by setting the attr `split_idx` of the transform to `None`.' %}


<h4 id="RandTransform.before_call" class="doc_header"><code>RandTransform.before_call</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L170" class="source_link" style="float:right">[source]</a></h4>

> <code>RandTransform.before_call</code>(**`b`**, **`split_idx`**)

Set `self.do` based on `self.p`


```python
def _add1(x): return x+1
dumb_tfm = RandTransform(enc=_add1, p=0.5)
start,d1,d2 = 2,False,False
for _ in range(40):
    t = dumb_tfm(start, split_idx=0)
    if dumb_tfm.do: test_eq(t, start+1); d1=True
    else:           test_eq(t, start)  ; d2=True
assert d1 and d2
dumb_tfm
```




    _add1 -- {'p': 0.5}:
    encodes: (object,object) -> _add1decodes: 



## Item transforms


<h4 id="Image.flip_lr" class="doc_header"><code>Image.flip_lr</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L45" class="source_link" style="float:right">[source]</a></h4>

> <code>Image.flip_lr</code>(**`x`**:`Image`)





<h4 id="TensorImageBase.flip_lr" class="doc_header"><code>TensorImageBase.flip_lr</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L47" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImageBase.flip_lr</code>(**`x`**:[`TensorImageBase`](/torch_core.html#TensorImageBase))





<h4 id="TensorPoint.flip_lr" class="doc_header"><code>TensorPoint.flip_lr</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L49" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorPoint.flip_lr</code>(**`x`**:[`TensorPoint`](/vision.core.html#TensorPoint))





<h4 id="TensorBBox.flip_lr" class="doc_header"><code>TensorBBox.flip_lr</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L51" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorBBox.flip_lr</code>(**`x`**:[`TensorBBox`](/vision.core.html#TensorBBox))




```python
_,axs = subplots(1,2)
show_image(img, ctx=axs[0], title='original')
show_image(img.flip_lr(), ctx=axs[1], title='flipped');
```


![png](output_16_0.png)



<h3 id="FlipItem" class="doc_header"><code>class</code> <code>FlipItem</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L55" class="source_link" style="float:right">[source]</a></h3>

> <code>FlipItem</code>(**`p`**=*`0.5`*) :: [`RandTransform`](/vision.augment.html#RandTransform)

Randomly flip with probability `p`


```python
tflip = FlipItem(p=1.)
test_eq(tflip(bbox,split_idx=0), tensor([[1.,0., 0.,1]]) -1)
```


<h4 id="PILImage.dihedral" class="doc_header"><code>PILImage.dihedral</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L61" class="source_link" style="float:right">[source]</a></h4>

> <code>PILImage.dihedral</code>(**`x`**:[`PILImage`](/vision.core.html#PILImage), **`k`**)





<h4 id="TensorImage.dihedral" class="doc_header"><code>TensorImage.dihedral</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L63" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.dihedral</code>(**`x`**:[`TensorImage`](/torch_core.html#TensorImage), **`k`**)





<h4 id="TensorPoint.dihedral" class="doc_header"><code>TensorPoint.dihedral</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L69" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorPoint.dihedral</code>(**`x`**:[`TensorPoint`](/vision.core.html#TensorPoint), **`k`**)





<h4 id="TensorBBox.dihedral" class="doc_header"><code>TensorBBox.dihedral</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L75" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorBBox.dihedral</code>(**`x`**:[`TensorBBox`](/vision.core.html#TensorBBox), **`k`**)




By default each of the 8 dihedral transformations (including noop) have the same probability of being picked when the transform is applied. You can customize this behavior by passing your own `draw` function. To force a specific flip, you can also pass an integer between 0 and 7. 


<h3 id="DihedralItem" class="doc_header"><code>class</code> <code>DihedralItem</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L82" class="source_link" style="float:right">[source]</a></h3>

> <code>DihedralItem</code>(**`p`**=*`1.0`*, **`nm`**=*`None`*, **`before_call`**=*`None`*, **\*\*`kwargs`**) :: [`RandTransform`](/vision.augment.html#RandTransform)

Randomly flip with probability `p`


```python
_,axs = subplots(2, 4)
for ax in axs.flatten():
    show_image(DihedralItem(p=1.)(img, split_idx=0), ctx=ax)
```


![png](output_28_0.png)


## Resize with crop, pad or squish


<h3 id="PadMode" class="doc_header"><code>class</code> <code>PadMode</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>PadMode</code>(**\*`args`**, **\*\*`kwargs`**)

All possible padding mode as attributes to get tab-completion and typo-proofing



<h4 id="TensorBBox.crop_pad" class="doc_header"><code>TensorBBox.crop_pad</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L129" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorBBox.crop_pad</code>(**`x`**:`Image'>)`, **`sz`**, **`tl`**=*`None`*, **`orig_sz`**=*`None`*, **`pad_mode`**=*`'zeros'`*, **`resize_mode`**=*`2`*, **`resize_to`**=*`None`*)





<h4 id="TensorPoint.crop_pad" class="doc_header"><code>TensorPoint.crop_pad</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L129" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorPoint.crop_pad</code>(**`x`**:`Image'>)`, **`sz`**, **`tl`**=*`None`*, **`orig_sz`**=*`None`*, **`pad_mode`**=*`'zeros'`*, **`resize_mode`**=*`2`*, **`resize_to`**=*`None`*)





<h4 id="Image.crop_pad" class="doc_header"><code>Image.crop_pad</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L129" class="source_link" style="float:right">[source]</a></h4>

> <code>Image.crop_pad</code>(**`x`**:`Image'>)`, **`sz`**, **`tl`**=*`None`*, **`orig_sz`**=*`None`*, **`pad_mode`**=*`'zeros'`*, **`resize_mode`**=*`2`*, **`resize_to`**=*`None`*)





<h3 id="CropPad" class="doc_header"><code>class</code> <code>CropPad</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L149" class="source_link" style="float:right">[source]</a></h3>

> <code>CropPad</code>(**`size`**, **`pad_mode`**=*`'zeros'`*, **\*\*`kwargs`**) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

Center crop or pad an image to `size`


```python
_,axs = plt.subplots(1,3,figsize=(12,4))
for ax,sz in zip(axs.flatten(), [300, 500, 700]):
    show_image(img.crop_pad(sz), ctx=ax, title=f'Size {sz}');
```


![png](output_41_0.png)


```python
_,axs = plt.subplots(1,3,figsize=(12,4))
for ax,mode in zip(axs.flatten(), [PadMode.Zeros, PadMode.Border, PadMode.Reflection]):
    show_image(img.crop_pad((600,700), pad_mode=mode), ctx=ax, title=mode);
```


![png](output_42_0.png)



<h3 id="RandTransform" class="doc_header"><code>class</code> <code>RandTransform</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L21" class="source_link" style="float:right">[source]</a></h3>

> <code>RandTransform</code>(**`p`**=*`1.0`*, **`nm`**=*`None`*, **`before_call`**=*`None`*, **\*\*`kwargs`**) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

A transform that before_call its state at each `__call__`



<h3 id="CropPad" class="doc_header"><code>class</code> <code>CropPad</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L149" class="source_link" style="float:right">[source]</a></h3>

> <code>CropPad</code>(**`size`**, **`pad_mode`**=*`'zeros'`*, **\*\*`kwargs`**) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

Center crop or pad an image to `size`



<h3 id="RandomCrop" class="doc_header"><code>class</code> <code>RandomCrop</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L194" class="source_link" style="float:right">[source]</a></h3>

> <code>RandomCrop</code>(**`size`**, **\*\*`kwargs`**) :: [`RandTransform`](/vision.augment.html#RandTransform)

Randomly crop an image to `size`



<h3 id="OldRandomCrop" class="doc_header"><code>class</code> <code>OldRandomCrop</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L210" class="source_link" style="float:right">[source]</a></h3>

> <code>OldRandomCrop</code>(**`size`**, **`pad_mode`**=*`'zeros'`*) :: [`CropPad`](/vision.augment.html#CropPad)

Randomly crop an image to `size`


```python
_,axs = plt.subplots(1,3,figsize=(12,4))
f = RandomCrop(200)
for ax in axs: show_image(f(img), ctx=ax);
```


![png](output_51_0.png)


On the validation set, we take a center crop.

```python
_,axs = plt.subplots(1,3,figsize=(12,4))
for ax in axs: show_image(f(img, split_idx=1), ctx=ax);
```


![png](output_53_0.png)



<h3 id="ResizeMethod" class="doc_header"><code>class</code> <code>ResizeMethod</code><a href="" class="source_link" style="float:right">[source]</a></h3>

> <code>ResizeMethod</code>(**\*`args`**, **\*\*`kwargs`**)

All possible resize method as attributes to get tab-completion and typo-proofing


```python
test_eq(ResizeMethod.Squish, 'squish')
```


<h3 id="Resize" class="doc_header"><code>class</code> <code>Resize</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L226" class="source_link" style="float:right">[source]</a></h3>

> <code>Resize</code>(**`size`**, **`method`**=*`'crop'`*, **`pad_mode`**=*`'reflection'`*, **`resamples`**=*`(2, 0)`*, **\*\*`kwargs`**) :: [`RandTransform`](/vision.augment.html#RandTransform)

A transform that before_call its state at each `__call__`


```python
Resize(224)
```




    Resize -- {'size': (224, 224), 'method': 'crop', 'pad_mode': 'reflection'}:
    encodes: (TensorBBox,object) -> encodes
    (TensorPoint,object) -> encodes
    (Image,object) -> encodes
    decodes: 



`size` can be an integer (in which case images will be resized to a square) or a tuple. Depending on the [`method`](https://fastcore.fast.ai/foundation#method):
- we squish any rectangle to `size`
- we resize so that the shorter dimension is a match an use padding with `pad_mode` 
- we resize so that the larger dimension is match and crop (randomly on the training set, center crop for the validation set)

When doing the resize, we use `resamples[0]` for images and `resamples[1]` for segmentation masks.

```python
_,axs = plt.subplots(1,3,figsize=(12,4))
for ax,method in zip(axs.flatten(), [ResizeMethod.Squish, ResizeMethod.Pad, ResizeMethod.Crop]):
    rsz = Resize(256, method=method)
    show_image(rsz(img, split_idx=0), ctx=ax, title=method);
```


![png](output_62_0.png)


On the validation set, the crop is always a center crop (on the dimension that's cropped).

```python
_,axs = plt.subplots(1,3,figsize=(12,4))
for ax,method in zip(axs.flatten(), [ResizeMethod.Squish, ResizeMethod.Pad, ResizeMethod.Crop]):
    rsz = Resize(256, method=method)
    show_image(rsz(img, split_idx=1), ctx=ax, title=method);
```


![png](output_64_0.png)



<h3 id="RandomResizedCrop" class="doc_header"><code>class</code> <code>RandomResizedCrop</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L255" class="source_link" style="float:right">[source]</a></h3>

> <code>RandomResizedCrop</code>(**`size`**, **`min_scale`**=*`0.08`*, **`ratio`**=*`(0.75, 1.3333333333333333)`*, **`resamples`**=*`(2, 0)`*, **`val_xtra`**=*`0.14`*, **\*\*`kwargs`**) :: [`RandTransform`](/vision.augment.html#RandTransform)

Picks a random scaled crop of an image and resize it to `size`


The crop picked as a random scale in range `(min_scale,1)` and `ratio` in the range passed, then the resize is done with `resamples[0]` for images and `resamples[1]` for segmentation masks. On the validation set, we center crop the image if it's ratio isn't in the range (to the minmum or maximum value) then resize.

```python
crop = RandomResizedCrop(256)
_,axs = plt.subplots(3,3,figsize=(9,9))
for ax in axs.flatten():
    cropped = crop(img)
    show_image(cropped, ctx=ax);
```


![png](output_68_0.png)


Squish is used on the validation set, removing `val_xtra` proportion of each side first.

```python
_,axs = subplots(1,3)
for ax in axs.flatten(): show_image(crop(img, split_idx=1), ctx=ax);
```


![png](output_70_0.png)


```python
test_eq(cropped.shape, [256,256])
```


<h3 id="RatioResize" class="doc_header"><code>class</code> <code>RatioResize</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L293" class="source_link" style="float:right">[source]</a></h3>

> <code>RatioResize</code>(**`max_sz`**, **`resamples`**=*`(2, 0)`*, **\*\*`kwargs`**) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

Resizes the biggest dimension of an image to `max_sz` maintaining the aspect ratio


```python
RatioResize(256)(img)
```




![png](output_74_0.png)



```python
test_eq(RatioResize(256)(img).size[0], 256)
test_eq(RatioResize(256)(img.dihedral(3)).size[1], 256)
```

## Affine and coord tfm on the GPU

```python
timg = TensorImage(array(img)).permute(2,0,1).float()/255.
def _batch_ex(bs): return TensorImage(timg[None].expand(bs, *timg.shape).clone())
```


<h4 id="TensorImage.affine_coord" class="doc_header"><code>TensorImage.affine_coord</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L329" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.affine_coord</code>(**`x`**:[`TensorImage`](/torch_core.html#TensorImage), **`mat`**=*`None`*, **`coord_tfm`**=*`None`*, **`sz`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*)





<h4 id="TensorMask.affine_coord" class="doc_header"><code>TensorMask.affine_coord</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L339" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorMask.affine_coord</code>(**`x`**:[`TensorMask`](/torch_core.html#TensorMask), **`mat`**=*`None`*, **`coord_tfm`**=*`None`*, **`sz`**=*`None`*, **`mode`**=*`'nearest'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*)





<h4 id="TensorPoint.affine_coord" class="doc_header"><code>TensorPoint.affine_coord</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L348" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorPoint.affine_coord</code>(**`x`**:[`TensorPoint`](/vision.core.html#TensorPoint), **`mat`**=*`None`*, **`coord_tfm`**=*`None`*, **`sz`**=*`None`*, **`mode`**=*`'nearest'`*, **`pad_mode`**=*`'zeros'`*, **`align_corners`**=*`True`*)





<h4 id="TensorBBox.affine_coord" class="doc_header"><code>TensorBBox.affine_coord</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L357" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorBBox.affine_coord</code>(**`x`**:[`TensorBBox`](/vision.core.html#TensorBBox), **`mat`**=*`None`*, **`coord_tfm`**=*`None`*, **`sz`**=*`None`*, **`mode`**=*`'nearest'`*, **`pad_mode`**=*`'zeros'`*, **`align_corners`**=*`True`*)





<h3 id="AffineCoordTfm" class="doc_header"><code>class</code> <code>AffineCoordTfm</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L380" class="source_link" style="float:right">[source]</a></h3>

> <code>AffineCoordTfm</code>(**`aff_fs`**=*`None`*, **`coord_fs`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`mode_mask`**=*`'nearest'`*, **`align_corners`**=*`None`*, **\*\*`kwargs`**) :: [`RandTransform`](/vision.augment.html#RandTransform)

Combine and apply affine and coord transforms


Multipliy all the matrices returned by `aff_fs` before doing the corresponding affine transformation on a basic grid corresponding to `size`, then applies all `coord_fs` on the resulting flow of coordinates before finally doing an interpolation with `mode` and `pad_mode`.


<h4 id="AffineCoordTfm.compose" class="doc_header"><code>AffineCoordTfm.compose</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L395" class="source_link" style="float:right">[source]</a></h4>

> <code>AffineCoordTfm.compose</code>(**`tfm`**)

Compose `self` with another [`AffineCoordTfm`](/vision.augment.html#AffineCoordTfm) to only do the interpolation step once



<h3 id="RandomResizedCropGPU" class="doc_header"><code>class</code> <code>RandomResizedCropGPU</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L419" class="source_link" style="float:right">[source]</a></h3>

> <code>RandomResizedCropGPU</code>(**`size`**, **`min_scale`**=*`0.08`*, **`ratio`**=*`(0.75, 1.3333333333333333)`*, **`mode`**=*`'bilinear'`*, **`valid_scale`**=*`1.0`*, **\*\*`kwargs`**) :: [`RandTransform`](/vision.augment.html#RandTransform)

Picks a random scaled crop of an image and resize it to `size`


```python
t = _batch_ex(8)
rrc = RandomResizedCropGPU(224, p=1.)
y = rrc(t)
_,axs = plt.subplots(2,4, figsize=(12,6))
for ax in axs.flatten():
    show_image(y[i], ctx=ax)
```


![png](output_92_0.png)


### Flip/Dihedral GPU helpers


<h4 id="affine_mat" class="doc_header"><code>affine_mat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L450" class="source_link" style="float:right">[source]</a></h4>

> <code>affine_mat</code>(**\*`ms`**)

Restructure length-6 vector `ms` into an affine matrix with 0,0,1 in the last line



<h4 id="mask_tensor" class="doc_header"><code>mask_tensor</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L457" class="source_link" style="float:right">[source]</a></h4>

> <code>mask_tensor</code>(**`x`**, **`p`**=*`0.5`*, **`neutral`**=*`0.0`*, **`batch`**=*`False`*)

Mask elements of `x` with `neutral` with probability `1-p`


```python
x = torch.zeros(5,2,3)
def_draw = lambda x: torch.randint(0,8, (x.size(0),))
t = _draw_mask(x, def_draw)
assert (0. <= t).all() and (t <= 7).all() 
t = _draw_mask(x, def_draw, 1)
assert (0. <= t).all() and (t <= 1).all() 
test_eq(_draw_mask(x, def_draw, 1, p=1), tensor([1.,1,1,1,1]))
test_eq(_draw_mask(x, def_draw, [0,1,2,3,4], p=1), tensor([0.,1,2,3,4]))
for i in range(5):
    t = _draw_mask(x, def_draw, 1, batch=True)
    assert (t==torch.zeros(5)).all() or (t==torch.ones(5)).all()
```


<h4 id="flip_mat" class="doc_header"><code>flip_mat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L477" class="source_link" style="float:right">[source]</a></h4>

> <code>flip_mat</code>(**`x`**, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`batch`**=*`False`*)

Return a random flip matrix


```python
x = flip_mat(torch.randn(100,4,3))
test_eq(set(x[:,0,0].numpy()), {-1,1}) #might fail with probability 2*2**(-100) (picked only 1s or -1s)
```


<h4 id="TensorImage.flip_batch" class="doc_header"><code>TensorImage.flip_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L493" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.flip_batch</code>(**`x`**:`TensorBBox'>)`, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`align_corners`**=*`True`*, **`batch`**=*`False`*)





<h4 id="TensorMask.flip_batch" class="doc_header"><code>TensorMask.flip_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L493" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorMask.flip_batch</code>(**`x`**:`TensorBBox'>)`, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`align_corners`**=*`True`*, **`batch`**=*`False`*)





<h4 id="TensorPoint.flip_batch" class="doc_header"><code>TensorPoint.flip_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L493" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorPoint.flip_batch</code>(**`x`**:`TensorBBox'>)`, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`align_corners`**=*`True`*, **`batch`**=*`False`*)





<h4 id="TensorBBox.flip_batch" class="doc_header"><code>TensorBBox.flip_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L493" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorBBox.flip_batch</code>(**`x`**:`TensorBBox'>)`, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`align_corners`**=*`True`*, **`batch`**=*`False`*)




```python
t = _pnt2tensor([[1,0], [2,1]], (3,3))
y = TensorImage(t[None,None]).flip_batch(p=1.)
test_eq(y, _pnt2tensor([[1,0], [0,1]], (3,3))[None,None])

pnts = TensorPoint((tensor([[1.,0.], [2,1]]) -1)[None])
test_eq(pnts.flip_batch(p=1.), tensor([[[1.,0.], [0,1]]]) -1)

bbox = TensorBBox(((tensor([[1.,0., 2.,1]]) -1)[None]))
test_eq(bbox.flip_batch(p=1.), tensor([[[0.,0., 1.,1.]]]) -1)
```


<h3 id="Flip" class="doc_header"><code>class</code> <code>Flip</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L501" class="source_link" style="float:right">[source]</a></h3>

> <code>Flip</code>(**`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **`batch`**=*`False`*) :: [`AffineCoordTfm`](/vision.augment.html#AffineCoordTfm)

Randomly flip a batch of images with a probability `p`


```python
Flip(0.3)
```




    Flip -- {'p': 0.3, 'size': None, 'mode': 'bilinear', 'pad_mode': 'reflection', 'mode_mask': 'nearest', 'align_corners': True}:
    encodes: (TensorBBox,object) -> encodes
    (TensorPoint,object) -> encodes
    (TensorImage,object) -> encodes
    (TensorMask,object) -> encodes
    decodes: 



```python
flip = Flip(p=1.)
t = _pnt2tensor([[1,0], [2,1]], (3,3))
y = flip(TensorImage(t[None,None]), split_idx=0)
test_eq(y, _pnt2tensor([[1,0], [0,1]], (3,3))[None,None])

pnts = TensorPoint((tensor([[1.,0.], [2,1]]) -1)[None])
test_eq(flip(pnts, split_idx=0), tensor([[[1.,0.], [0,1]]]) -1)

bbox = TensorBBox(((tensor([[1.,0., 2.,1]]) -1)[None]))
test_eq(flip(bbox, split_idx=0), tensor([[[0.,0., 1.,1.]]]) -1)
```


<h3 id="DeterministicDraw" class="doc_header"><code>class</code> <code>DeterministicDraw</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L508" class="source_link" style="float:right">[source]</a></h3>

> <code>DeterministicDraw</code>(**`vals`**)




```python
t =  _batch_ex(8)
draw = DeterministicDraw(list(range(8)))
for i in range(15): test_eq(draw(t), torch.zeros(8)+(i%8))
```


<h3 id="DeterministicFlip" class="doc_header"><code>class</code> <code>DeterministicFlip</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L516" class="source_link" style="float:right">[source]</a></h3>

> <code>DeterministicFlip</code>(**`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**) :: [`Flip`](/vision.augment.html#Flip)

Flip the batch every other call


```python
dih = DeterministicFlip({'p':.3})
```

```python
t = _batch_ex(8)
dih = DeterministicFlip()
_,axs = plt.subplots(2,4, figsize=(12,6))
for i,ax in enumerate(axs.flatten()):
    y = dih(t)
    show_image(y[0], ctx=ax, title=f'Call {i}')
```


![png](output_120_0.png)



<h4 id="dihedral_mat" class="doc_header"><code>dihedral_mat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L522" class="source_link" style="float:right">[source]</a></h4>

> <code>dihedral_mat</code>(**`x`**, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`batch`**=*`False`*)

Return a random dihedral matrix



<h4 id="TensorImage.dihedral_batch" class="doc_header"><code>TensorImage.dihedral_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L535" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.dihedral_batch</code>(**`x`**:`TensorBBox'>)`, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`batch`**=*`False`*, **`align_corners`**=*`True`*)





<h4 id="TensorMask.dihedral_batch" class="doc_header"><code>TensorMask.dihedral_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L535" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorMask.dihedral_batch</code>(**`x`**:`TensorBBox'>)`, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`batch`**=*`False`*, **`align_corners`**=*`True`*)





<h4 id="TensorPoint.dihedral_batch" class="doc_header"><code>TensorPoint.dihedral_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L535" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorPoint.dihedral_batch</code>(**`x`**:`TensorBBox'>)`, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`batch`**=*`False`*, **`align_corners`**=*`True`*)





<h4 id="TensorBBox.dihedral_batch" class="doc_header"><code>TensorBBox.dihedral_batch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L535" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorBBox.dihedral_batch</code>(**`x`**:`TensorBBox'>)`, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`batch`**=*`False`*, **`align_corners`**=*`True`*)





<h3 id="Dihedral" class="doc_header"><code>class</code> <code>Dihedral</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L543" class="source_link" style="float:right">[source]</a></h3>

> <code>Dihedral</code>(**`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`None`*, **`batch`**=*`False`*) :: [`AffineCoordTfm`](/vision.augment.html#AffineCoordTfm)

Apply a random dihedral transformation to a batch of images with a probability `p`


`draw` can be specified if you want to customize which flip is picked when the transform is applied (default is a random number between 0 and 7). It can be an integer between 0 and 7, a list of such integers (which then should have a length equal to the size of the batch) or a callable that returns an integer between 0 and 7.

```python
t = _batch_ex(8)
dih = Dihedral(p=1., draw=list(range(8)))
y = dih(t)
y = t.dihedral_batch(p=1., draw=list(range(8)))
_,axs = plt.subplots(2,4, figsize=(12,5))
for i,ax in enumerate(axs.flatten()): show_image(y[i], ctx=ax, title=f'Flip {i}')
```


![png](output_131_0.png)



<h3 id="DeterministicDihedral" class="doc_header"><code>class</code> <code>DeterministicDihedral</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L550" class="source_link" style="float:right">[source]</a></h3>

> <code>DeterministicDihedral</code>(**`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`None`*) :: [`Dihedral`](/vision.augment.html#Dihedral)

Apply a random dihedral transformation to a batch of images with a probability `p`


```python
t = _batch_ex(8)
dih = DeterministicDihedral()
_,axs = plt.subplots(2,4, figsize=(12,6))
for i,ax in enumerate(axs.flatten()):
    y = dih(t)
    show_image(y[0], ctx=ax, title=f'Call {i}')
```


![png](output_134_0.png)



<h4 id="rotate_mat" class="doc_header"><code>rotate_mat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L556" class="source_link" style="float:right">[source]</a></h4>

> <code>rotate_mat</code>(**`x`**, **`max_deg`**=*`10`*, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`batch`**=*`False`*)

Return a random rotation matrix with `max_deg` and `p`



<h4 id="TensorImage.rotate" class="doc_header"><code>TensorImage.rotate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L565" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.rotate</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h4 id="TensorMask.rotate" class="doc_header"><code>TensorMask.rotate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L565" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorMask.rotate</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h4 id="TensorPoint.rotate" class="doc_header"><code>TensorPoint.rotate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L565" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorPoint.rotate</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h4 id="TensorBBox.rotate" class="doc_header"><code>TensorBBox.rotate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L565" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorBBox.rotate</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`None`*, **`pad_mode`**=*`None`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h3 id="Rotate" class="doc_header"><code>class</code> <code>Rotate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L573" class="source_link" style="float:right">[source]</a></h3>

> <code>Rotate</code>(**`max_deg`**=*`10`*, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **`batch`**=*`False`*) :: [`AffineCoordTfm`](/vision.augment.html#AffineCoordTfm)

Apply a random rotation of at most `max_deg` with probability `p` to a batch of images


`draw` can be specified if you want to customize which angle is picked when the transform is applied (default is a random flaot between `-max_deg` and `max_deg`). It can be a float, a list of floats (which then should have a length equal to the size of the batch) or a callable that returns a float.

```python
thetas = [-30,-15,0,15,30]
y = _batch_ex(5).rotate(draw=thetas, p=1.)
_,axs = plt.subplots(1,5, figsize=(15,3))
for i,ax in enumerate(axs.flatten()):
    show_image(y[i], ctx=ax, title=f'{thetas[i]} degrees')
```


![png](output_145_0.png)



<h4 id="zoom_mat" class="doc_header"><code>zoom_mat</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L580" class="source_link" style="float:right">[source]</a></h4>

> <code>zoom_mat</code>(**`x`**, **`min_zoom`**=*`1.0`*, **`max_zoom`**=*`1.1`*, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`draw_x`**=*`None`*, **`draw_y`**=*`None`*, **`batch`**=*`False`*)

Return a random zoom matrix with `max_zoom` and `p`



<h4 id="TensorImage.zoom" class="doc_header"><code>TensorImage.zoom</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L597" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.zoom</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h4 id="TensorMask.zoom" class="doc_header"><code>TensorMask.zoom</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L597" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorMask.zoom</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h4 id="TensorPoint.zoom" class="doc_header"><code>TensorPoint.zoom</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L597" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorPoint.zoom</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h4 id="TensorBBox.zoom" class="doc_header"><code>TensorBBox.zoom</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L597" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorBBox.zoom</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h3 id="Zoom" class="doc_header"><code>class</code> <code>Zoom</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L605" class="source_link" style="float:right">[source]</a></h3>

> <code>Zoom</code>(**`min_zoom`**=*`1.0`*, **`max_zoom`**=*`1.1`*, **`p`**=*`0.5`*, **`draw`**=*`None`*, **`draw_x`**=*`None`*, **`draw_y`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`batch`**=*`False`*, **`align_corners`**=*`True`*) :: [`AffineCoordTfm`](/vision.augment.html#AffineCoordTfm)

Apply a random zoom of at most `max_zoom` with probability `p` to a batch of images


`draw`, `draw_x` and `draw_y` can be specified if you want to customize which scale and center are picked when the transform is applied (default is a random float between 1 and `max_zoom` for the first, between 0 and 1 for the last two). Each can be a float, a list of floats (which then should have a length equal to the size of the batch) or a callbale that returns a float.

`draw_x` and `draw_y` are expected to be the position of the center in pct, 0 meaning the most left/top possible and 1 meaning the most right/bottom possible.

```python
scales = [0.8, 1., 1.1, 1.25, 1.5]
n = len(scales)
y = _batch_ex(n).zoom(draw=scales, p=1., draw_x=0.5, draw_y=0.5)
fig,axs = plt.subplots(1, n, figsize=(12,3))
fig.suptitle('Center zoom with different scales')
for i,ax in enumerate(axs.flatten()):
    show_image(y[i], ctx=ax, title=f'scale {scales[i]}')
```


![png](output_156_0.png)


```python
y = _batch_ex(4).zoom(p=1., draw=1.5)
fig,axs = plt.subplots(1,4, figsize=(12,3))
fig.suptitle('Constant scale and different random centers')
for i,ax in enumerate(axs.flatten()):
    show_image(y[i], ctx=ax)
```


![png](output_157_0.png)


### Warping


<h4 id="find_coeffs" class="doc_header"><code>find_coeffs</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L613" class="source_link" style="float:right">[source]</a></h4>

> <code>find_coeffs</code>(**`p1`**, **`p2`**)

Find coefficients for warp tfm from `p1` to `p2`



<h4 id="apply_perspective" class="doc_header"><code>apply_perspective</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L627" class="source_link" style="float:right">[source]</a></h4>

> <code>apply_perspective</code>(**`coords`**, **`coeffs`**)

Apply perspective tranfom on `coords` with `coeffs`



<h4 id="TensorImage.warp" class="doc_header"><code>TensorImage.warp</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L662" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.warp</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h4 id="TensorMask.warp" class="doc_header"><code>TensorMask.warp</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L662" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorMask.warp</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h4 id="TensorPoint.warp" class="doc_header"><code>TensorPoint.warp</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L662" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorPoint.warp</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h4 id="TensorBBox.warp" class="doc_header"><code>TensorBBox.warp</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L662" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorBBox.warp</code>(**`x`**:`TensorBBox'>)`, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **\*\*`kwargs`**)





<h3 id="Warp" class="doc_header"><code>class</code> <code>Warp</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L672" class="source_link" style="float:right">[source]</a></h3>

> <code>Warp</code>(**`magnitude`**=*`0.2`*, **`p`**=*`0.5`*, **`draw_x`**=*`None`*, **`draw_y`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`batch`**=*`False`*, **`align_corners`**=*`True`*) :: [`AffineCoordTfm`](/vision.augment.html#AffineCoordTfm)

Apply perspective warping with `magnitude` and `p` on a batch of matrices


`draw_x` and `draw_y` can be specified if you want to customize the magnitudes that are picked when the transform is applied (default is a random float between `-magnitude` and `magnitude`. Each can be a float, a list of floats (which then should have a length equal to the size of the batch) or a callable that returns a float.

```python
scales = [-0.4, -0.2, 0., 0.2, 0.4]
warp = Warp(p=1., draw_y=scales, draw_x=0.)
y = warp(_batch_ex(5), split_idx=0)
fig,axs = plt.subplots(1,5, figsize=(15,3))
fig.suptitle('Vertical warping')
for i,ax in enumerate(axs.flatten()):
    show_image(y[i], ctx=ax, title=f'magnitude {scales[i]}')
```


![png](output_172_0.png)


```python
scales = [-0.4, -0.2, 0., 0.2, 0.4]
warp = Warp(p=1., draw_x=scales, draw_y=0.)
y = warp(_batch_ex(5), split_idx=0)
fig,axs = plt.subplots(1,5, figsize=(15,3))
fig.suptitle('Horizontal warping')
for i,ax in enumerate(axs.flatten()):
    show_image(y[i], ctx=ax, title=f'magnitude {scales[i]}')
```


![png](output_173_0.png)


## Lighting transforms

Lighting transforms are transforms that effect how light is represented in an image. These don't change the location of the object like previous transforms, but instead simulate how light could change in a scene. The [simclr paper](https://arxiv.org/abs/2002.05709) evaluates these transforms against other transforms for their use case of self-supurved image classification, note they use "color" and "color distortion" to refer to a combination of these transforms. 


<h4 id="TensorImage.lighting" class="doc_header"><code>TensorImage.lighting</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L681" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.lighting</code>(**`x`**:[`TensorImage`](/torch_core.html#TensorImage), **`func`**)




Most lighting transforms work better in "logit space", as we do not want to blowout the image by going over maximum or minimum brightness. Taking the sigmoid of the logit allows us to get back to "linear space." 

```python
x=TensorImage(torch.tensor([.01* i for i in range(0,101)]))
f_lin= lambda x:(2*(x-0.5)+0.5).clamp(0,1) #blue line
f_log= lambda x:2*x #red line
plt.plot(x,f_lin(x),'b',x,x.lighting(f_log),'r')
```




    [<matplotlib.lines.Line2D at 0x7f12cc15e4d0>,
     <matplotlib.lines.Line2D at 0x7f13250a1d10>]




![png](output_179_1.png)


The above graph shows the results of doing a contrast transformation in both linear and logit space. Notice how the blue linear plot has to be clamped, and we have lost information on how large 0.0 is by comparision to 0.2. While in the red plot the values curve, so we keep this relative information. 


<h3 id="LightingTfm" class="doc_header"><code>class</code> <code>LightingTfm</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L686" class="source_link" style="float:right">[source]</a></h3>

> <code>LightingTfm</code>(**`fs`**, **\*\*`kwargs`**) :: [`RandTransform`](/vision.augment.html#RandTransform)

Apply `fs` to the logits


Brightness refers to the amount of light on a scene. This can be zero in which the image is completely black or one where the image is completely white. This may be especially useful if you expect your dataset to have over or under exposed images. 


<h4 id="TensorImage.brightness" class="doc_header"><code>TensorImage.brightness</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L719" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.brightness</code>(**`x`**:[`TensorImage`](/torch_core.html#TensorImage), **\*\*`kwargs`**)





<h3 id="Brightness" class="doc_header"><code>class</code> <code>Brightness</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L727" class="source_link" style="float:right">[source]</a></h3>

> <code>Brightness</code>(**`max_lighting`**=*`0.2`*, **`p`**=*`0.75`*, **`draw`**=*`None`*, **`batch`**=*`False`*) :: [`LightingTfm`](/vision.augment.html#LightingTfm)

Apply `fs` to the logits


```python
Brightness(0.5, p=0.8)
```




    Brightness -- {'p': 1.0, 'max_lighting': 0.5}:
    encodes: (TensorImage,object) -> encodes
    decodes: 



`draw` can be specified if you want to customize the magnitude that is picked when the transform is applied (default is a random float between `-0.5*(1-max_lighting)` and `0.5*(1+max_lighting)`. Each can be a float, a list of floats (which then should have a length equal to the size of the batch) or a callable that returns a float.

```python
scales = [0.1, 0.3, 0.5, 0.7, 0.9]
y = _batch_ex(5).brightness(draw=scales, p=1.)
fig,axs = plt.subplots(1,5, figsize=(15,3))
for i,ax in enumerate(axs.flatten()):
    show_image(y[i], ctx=ax, title=f'scale {scales[i]}')
```


![png](output_191_0.png)


Contrast pushes pixels to either the maximum or minimum values. The minimum value for contrast is a solid gray image. As an example take a picture of a bright light source in a dark room. Your eyes should be able to see some detail in the room, but the photo taken should instead have much higher contrast, with all of the detail in the background missing to the darkness. This is one example of what this transform can help simulate. 


<h4 id="TensorImage.contrast" class="doc_header"><code>TensorImage.contrast</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L749" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.contrast</code>(**`x`**:[`TensorImage`](/torch_core.html#TensorImage), **\*\*`kwargs`**)





<h3 id="Contrast" class="doc_header"><code>class</code> <code>Contrast</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L757" class="source_link" style="float:right">[source]</a></h3>

> <code>Contrast</code>(**`max_lighting`**=*`0.2`*, **`p`**=*`0.75`*, **`draw`**=*`None`*, **`batch`**=*`False`*) :: [`LightingTfm`](/vision.augment.html#LightingTfm)

Apply change in contrast of `max_lighting` to batch of images with probability `p`.


`draw` can be specified if you want to customize the magnitude that is picked when the transform is applied (default is a random float taken with the log uniform distribution between `(1-max_lighting)` and `1/(1-max_lighting)`. Each can be a float, a list of floats (which then should have a length equal to the size of the batch) or a callable that returns a float.

```python
scales = [0.65, 0.8, 1., 1.25, 1.55]
y = _batch_ex(5).contrast(p=1., draw=scales)
fig,axs = plt.subplots(1,5, figsize=(15,3))
for i,ax in enumerate(axs.flatten()): show_image(y[i], ctx=ax, title=f'scale {scales[i]}')
```


![png](output_199_0.png)



<h4 id="grayscale" class="doc_header"><code>grayscale</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L765" class="source_link" style="float:right">[source]</a></h4>

> <code>grayscale</code>(**`x`**)

Tensor to grayscale tensor. Uses the ITU-R 601-2 luma transform. 


```python
'%.3f' % sum([0.2989,0.5870,0.1140])
```




    '1.000'



The above is just one way to convert to grayscale. We chose this one because it was fast. Notice that the sum of the weight of each channel is 1. 


<h4 id="TensorImage.saturation" class="doc_header"><code>TensorImage.saturation</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L790" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.saturation</code>(**`x`**:[`TensorImage`](/torch_core.html#TensorImage), **\*\*`kwargs`**)





<h3 id="Saturation" class="doc_header"><code>class</code> <code>Saturation</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L798" class="source_link" style="float:right">[source]</a></h3>

> <code>Saturation</code>(**`max_lighting`**=*`0.2`*, **`p`**=*`0.75`*, **`draw`**=*`None`*, **`batch`**=*`False`*) :: [`LightingTfm`](/vision.augment.html#LightingTfm)

Apply change in saturation of `max_lighting` to batch of images with probability `p`.
Ref: https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.functional.adjust_saturation


```python
scales = [0., 0.5, 1., 1.5, 2.0]
y = _batch_ex(5).saturation(p=1., draw=scales)
fig,axs = plt.subplots(1,5, figsize=(15,3))
for i,ax in enumerate(axs.flatten()): show_image(y[i], ctx=ax, title=f'scale {scales[i]}')
```


![png](output_209_0.png)


Saturation controls the amount of color in the image, but not the lightness or darkness of an image. If has no effect on neutral colors such as whites,grays and blacks. At zero saturation you actually get a grayscale image. Pushing saturation past one causes more neutral colors to take on any underlying chromatic color. 


<h4 id="rgb2hsv" class="doc_header"><code>rgb2hsv</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L808" class="source_link" style="float:right">[source]</a></h4>

> <code>rgb2hsv</code>(**`img`**)

Converts a RGB image to an HSV image.
Note: Will not work on logit space images.



<h4 id="hsv2rgb" class="doc_header"><code>hsv2rgb</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L833" class="source_link" style="float:right">[source]</a></h4>

> <code>hsv2rgb</code>(**`img`**)

Converts a HSV image to an RGB image.
    


```python
fig,axs=plt.subplots(figsize=(20, 4),ncols=5)
for ax in axs:
    ax.set_ylabel('Hue')
    ax.set_xlabel('Saturation')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

hsv=torch.stack([torch.arange(0,2.1,0.01)[:,None].repeat(1,210),torch.arange(0,1.05,0.005)[None].repeat(210,1),torch.ones([210,210])])[None]
for ax,i in zip(axs,range(0,5)):
    if i>0: hsv[:,2].mul_(0.80)
    ax.set_title('V='+'%.1f' %0.8**i)
    ax.imshow(hsv2rgb(hsv)[0].permute(1,2,0))
```


![png](output_215_0.png)


For the Hue transform we are using hsv space instead of logit space. HSV stands for hue,saturation and value. Hue in hsv space just cycles through colors of the rainbow. Notices how there is no maximum, because the colors just repeat. 

Above are some examples of Hue(H) and Saturation(S) at various Values(V). One property of note in HSV space is that V controls the color you get at minimum saturation when in HSV space. 


<h4 id="TensorImage.hue" class="doc_header"><code>TensorImage.hue</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L877" class="source_link" style="float:right">[source]</a></h4>

> <code>TensorImage.hue</code>(**`x`**:[`TensorImage`](/torch_core.html#TensorImage), **\*\*`kwargs`**)





<h3 id="Hue" class="doc_header"><code>class</code> <code>Hue</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L885" class="source_link" style="float:right">[source]</a></h3>

> <code>Hue</code>(**`max_hue`**=*`0.1`*, **`p`**=*`0.75`*, **`draw`**=*`None`*, **`batch`**=*`False`*) :: [`RandTransform`](/vision.augment.html#RandTransform)

Apply change in hue of `max_hue` to batch of images with probability `p`.
Ref: https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.functional.adjust_hue


```python
scales = [0.5, 0.75, 1., 1.5, 1.75]
y = _batch_ex(len(scales)).hue(p=1., draw=scales)
fig,axs = plt.subplots(1,len(scales), figsize=(15,3))
for i,ax in enumerate(axs.flatten()): show_image(y[i], ctx=ax, title=f'scale {scales[i]}')
```


![png](output_222_0.png)


## RandomErasing

[Random Erasing Data Augmentation](https://arxiv.org/pdf/1708.04896.pdf). This variant, designed by Ross Wightman, is applied to either a batch or single image tensor after it has been normalized.


<h4 id="cutout_gaussian" class="doc_header"><code>cutout_gaussian</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L895" class="source_link" style="float:right">[source]</a></h4>

> <code>cutout_gaussian</code>(**`x`**, **`areas`**)

Replace all `areas` in `x` with N(0,1) noise


Since this should be applied after normalization, we'll define a helper to apply a function inside normalization.


<h4 id="norm_apply_denorm" class="doc_header"><code>norm_apply_denorm</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L902" class="source_link" style="float:right">[source]</a></h4>

> <code>norm_apply_denorm</code>(**`x`**, **`f`**, **`nrm`**)

Normalize `x` with `nrm`, then apply `f`, then denormalize


```python
nrm = Normalize.from_stats(*imagenet_stats, cuda=False)
```

```python
f = partial(cutout_gaussian, areas=[(100,200,100,200),(200,300,200,300)])
show_image(norm_apply_denorm(timg, f, nrm)[0]);
```


![png](output_231_0.png)



<h3 id="RandomErasing" class="doc_header"><code>class</code> <code>RandomErasing</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L914" class="source_link" style="float:right">[source]</a></h3>

> <code>RandomErasing</code>(**`p`**=*`0.5`*, **`sl`**=*`0.0`*, **`sh`**=*`0.3`*, **`min_aspect`**=*`0.3`*, **`max_count`**=*`1`*) :: [`RandTransform`](/vision.augment.html#RandTransform)

Randomly selects a rectangle region in an image and randomizes its pixels.


**Args:**

- p: The probability that the Random Erasing operation will be performed
- sl: Minimum proportion of erased area
- sh: Maximum proportion of erased area
- min_aspect: Minimum aspect ratio of erased area
- max_count: maximum number of erasing blocks per image, area per box is scaled by count

```python
tfm = RandomErasing(p=1., max_count=6)

_,axs = subplots(2,3, figsize=(12,6))
f = partial(tfm, split_idx=0)
for i,ax in enumerate(axs.flatten()): show_image(norm_apply_denorm(timg, f, nrm)[0], ctx=ax)
```


![png](output_236_0.png)


```python
y = _batch_ex(6)
_,axs = plt.subplots(2,3, figsize=(12,6))
y = norm_apply_denorm(y, f, nrm)
for i,ax in enumerate(axs.flatten()): show_image(y[i], ctx=ax)
```


![png](output_237_0.png)


```python
tfm = RandomErasing(p=1., max_count=6)

_,axs = subplots(2,3, figsize=(12,6))
f = partial(tfm, split_idx=1)
for i,ax in enumerate(axs.flatten()): show_image(norm_apply_denorm(timg, f, nrm)[0], ctx=ax)
```


![png](output_238_0.png)


## All together


<h4 id="setup_aug_tfms" class="doc_header"><code>setup_aug_tfms</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L944" class="source_link" style="float:right">[source]</a></h4>

> <code>setup_aug_tfms</code>(**`tfms`**)

Go through `tfms` and combines together affine/coord or lighting transforms


```python
tfms = [Rotate(draw=10., p=1), Zoom(draw=1.1, draw_x=0.5, draw_y=0.5, p=1.)]
comp = setup_aug_tfms([Rotate(draw=10., p=1), Zoom(draw=1.1, draw_x=0.5, draw_y=0.5, p=1.)])
test_eq(len(comp), 1)
x = torch.randn(4,3,5,5)
test_close(comp[0]._get_affine_mat(x)[...,:2],tfms[0]._get_affine_mat(x)[...,:2] @ tfms[1]._get_affine_mat(x)[...,:2])
#We can't test that the ouput of comp or the composition of tfms on x is the same cause it's not (1 interpol vs 2 sp)
```

```python
tfms = [Rotate(), Zoom(), Warp(), Brightness(), Flip(), Contrast()]
comp = setup_aug_tfms(tfms)
```

```python
aff_tfm,lig_tfm = comp
test_eq(len(aff_tfm.aff_fs+aff_tfm.coord_fs+comp[1].fs), 6)
test_eq(len(aff_tfm.aff_fs), 3)
test_eq(len(aff_tfm.coord_fs), 1)
test_eq(len(lig_tfm.fs), 2)
```


<h4 id="aug_transforms" class="doc_header"><code>aug_transforms</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/augment.py#L956" class="source_link" style="float:right">[source]</a></h4>

> <code>aug_transforms</code>(**`mult`**=*`1.0`*, **`do_flip`**=*`True`*, **`flip_vert`**=*`False`*, **`max_rotate`**=*`10.0`*, **`min_zoom`**=*`1.0`*, **`max_zoom`**=*`1.1`*, **`max_lighting`**=*`0.2`*, **`max_warp`**=*`0.2`*, **`p_affine`**=*`0.75`*, **`p_lighting`**=*`0.75`*, **`xtra_tfms`**=*`None`*, **`size`**=*`None`*, **`mode`**=*`'bilinear'`*, **`pad_mode`**=*`'reflection'`*, **`align_corners`**=*`True`*, **`batch`**=*`False`*, **`min_scale`**=*`1.0`*)

Utility func to easily create a list of flip, rotate, zoom, warp, lighting transforms.


Random flip (or dihedral if `flip_vert=True`) with `p=0.5` is added when `do_flip=True`. With `p_affine` we apply a random rotation of `max_rotate` degrees, a random zoom between `min_zoom` and `max_zoom` and a perspective warping of `max_warp`. With `p_lighting` we apply a change in brightness and contrast of `max_lighting`. Custon `xtra_tfms` can be added. `size`, `mode` and `pad_mode` will be used for the interpolation. `max_rotate,max_lighting,max_warp` are multiplied by `mult` so you can more easily increase or decrease augmentation with a single parameter.

```python
tfms = aug_transforms(pad_mode='zeros', mult=2, min_scale=0.5)
y = _batch_ex(9)
for t in tfms: y = t(y, split_idx=0)
_,axs = plt.subplots(1,3, figsize=(12,3))
for i,ax in enumerate(axs.flatten()): show_image(y[i], ctx=ax)
```


![png](output_249_0.png)


```python
tfms = aug_transforms(pad_mode='zeros', mult=2, batch=True)
y = _batch_ex(9)
for t in tfms: y = t(y, split_idx=0)
_,axs = plt.subplots(1,3, figsize=(12,3))
for i,ax in enumerate(axs.flatten()): show_image(y[i], ctx=ax)
```


![png](output_250_0.png)


## Integration tests

### Segmentation

```python
camvid = untar_data(URLs.CAMVID_TINY)
fns = get_image_files(camvid/'images')
cam_fn = fns[0]
mask_fn = camvid/'labels'/f'{cam_fn.stem}_P{cam_fn.suffix}'
def _cam_lbl(fn): return mask_fn
```

```python
cam_dsrc = Datasets([cam_fn]*10, [PILImage.create, [_cam_lbl, PILMask.create]])
cam_tdl = TfmdDL(cam_dsrc.train, after_item=ToTensor(),
                 after_batch=[IntToFloatTensor(), *aug_transforms()], bs=9)
cam_tdl.show_batch(max_n=9, vmin=1, vmax=30)
```


![png](output_254_0.png)


### Point targets

```python
mnist = untar_data(URLs.MNIST_TINY)
mnist_fn = 'images/mnist3.png'
pnts = np.array([[0,0], [0,35], [28,0], [28,35], [9, 17]])
def _pnt_lbl(fn)->None: return TensorPoint.create(pnts)
```





```python
pnt_dsrc = Datasets([mnist_fn]*10, [[PILImage.create, Resize((35,28))], _pnt_lbl])
pnt_tdl = TfmdDL(pnt_dsrc.train, after_item=[PointScaler(), ToTensor()],
                 after_batch=[IntToFloatTensor(), *aug_transforms(max_warp=0)], bs=9)
pnt_tdl.show_batch(max_n=9)
```


![png](output_257_0.png)


### Bounding boxes

```python
coco = untar_data(URLs.COCO_TINY)
images, lbl_bbox = get_annotations(coco/'train.json')
idx=2
coco_fn,bbox = coco/'train'/images[idx],lbl_bbox[idx]

def _coco_bb(x):  return TensorBBox.create(bbox[0])
def _coco_lbl(x): return bbox[1]
```





```python
coco_dsrc = Datasets([coco_fn]*10, [PILImage.create, [_coco_bb], [_coco_lbl, MultiCategorize(add_na=True)]], n_inp=1)
coco_tdl = TfmdDL(coco_dsrc, bs=9, after_item=[BBoxLabeler(), PointScaler(), ToTensor()],
                  after_batch=[IntToFloatTensor(), *aug_transforms()])

coco_tdl.show_batch(max_n=9)
```


![png](output_260_0.png)


```python
coco_tdl.after_batch
```




    Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1} -> Flip -- {'p': 0.5, 'size': None, 'mode': 'bilinear', 'pad_mode': 'reflection', 'mode_mask': 'nearest', 'align_corners': True} -> Brightness -- {'p': 1.0, 'max_lighting': 0.2}


