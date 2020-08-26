# Medical Imaging
> Helpers for working with DICOM files


## Patching


<h4 id="get_dicom_files" class="doc_header"><code>get_dicom_files</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L27" class="source_link" style="float:right">[source]</a></h4>

> <code>get_dicom_files</code>(**`path`**, **`recurse`**=*`True`*, **`folders`**=*`None`*)

Get dicom files in `path` recursively, only in `folders`, if specified.



<h4 id="Path.dcmread" class="doc_header"><code>Path.dcmread</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L32" class="source_link" style="float:right">[source]</a></h4>

> <code>Path.dcmread</code>(**`fn`**:`Path`, **`force`**=*`False`*)

Open a `DICOM` file



<h2 id="TensorDicom" class="doc_header"><code>class</code> <code>TensorDicom</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L38" class="source_link" style="float:right">[source]</a></h2>

> <code>TensorDicom</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorImage`](/torch_core.html#TensorImage)





<h2 id="PILDicom" class="doc_header"><code>class</code> <code>PILDicom</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L41" class="source_link" style="float:right">[source]</a></h2>

> <code>PILDicom</code>() :: [`PILBase`](/vision.core.html#PILBase)

This class represents an image object.  To create
:py:class:`~PIL.Image.Image` objects, use the appropriate factory
functions.  There's hardly ever any reason to call the Image constructor
directly.

* :py:func:`~PIL.Image.open`
* :py:func:`~PIL.Image.new`
* :py:func:`~PIL.Image.frombytes`


```python
# @patch
# def png16read(self:Path): return array(Image.open(self), dtype=np.uint16)
```

```python
TEST_DCM = Path('images/sample.dcm')
dcm = TEST_DCM.dcmread()
```


<h4 id="pixels" class="doc_header"><code>pixels</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L55" class="source_link" style="float:right">[source]</a></h4>

> <code>pixels</code>()

`pixel_array` as a tensor



<h4 id="scaled_px" class="doc_header"><code>scaled_px</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L61" class="source_link" style="float:right">[source]</a></h4>

> <code>scaled_px</code>()

[`pixels`](/medical.imaging.html#pixels) scaled by `RescaleSlope` and `RescaleIntercept`



<h4 id="array_freqhist_bins" class="doc_header"><code>array_freqhist_bins</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L68" class="source_link" style="float:right">[source]</a></h4>

> <code>array_freqhist_bins</code>(**`n_bins`**=*`100`*)

A numpy based function to split the range of pixel values into groups, such that each group has around the same number of pixels



<h4 id="Tensor.freqhist_bins" class="doc_header"><code>Tensor.freqhist_bins</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L78" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.freqhist_bins</code>(**`n_bins`**=*`100`*)

A function to split the range of pixel values into groups, such that each group has around the same number of pixels



<h4 id="Tensor.hist_scaled_pt" class="doc_header"><code>Tensor.hist_scaled_pt</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L89" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.hist_scaled_pt</code>(**`brks`**=*`None`*)





<h4 id="Tensor.hist_scaled" class="doc_header"><code>Tensor.hist_scaled</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L98" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.hist_scaled</code>(**`brks`**=*`None`*)





<h4 id="Dataset.hist_scaled" class="doc_header"><code>Dataset.hist_scaled</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L108" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.hist_scaled</code>(**`brks`**=*`None`*, **`min_px`**=*`None`*, **`max_px`**=*`None`*)





<h4 id="Tensor.windowed" class="doc_header"><code>Tensor.windowed</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L116" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.windowed</code>(**`w`**, **`l`**)





<h4 id="Dataset.windowed" class="doc_header"><code>Dataset.windowed</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L126" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.windowed</code>(**`w`**, **`l`**)





<h2 id="TensorCTScan" class="doc_header"><code>class</code> <code>TensorCTScan</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L147" class="source_link" style="float:right">[source]</a></h2>

> <code>TensorCTScan</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorImageBW`](/torch_core.html#TensorImageBW)





<h2 id="PILCTScan" class="doc_header"><code>class</code> <code>PILCTScan</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L150" class="source_link" style="float:right">[source]</a></h2>

> <code>PILCTScan</code>() :: [`PILBase`](/vision.core.html#PILBase)

This class represents an image object.  To create
:py:class:`~PIL.Image.Image` objects, use the appropriate factory
functions.  There's hardly ever any reason to call the Image constructor
directly.

* :py:func:`~PIL.Image.open`
* :py:func:`~PIL.Image.new`
* :py:func:`~PIL.Image.frombytes`



<h4 id="Dataset.show" class="doc_header"><code>Dataset.show</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L153" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.show</code>(**`scale`**=*`True`*, **`cmap`**=*`<matplotlib.colors.LinearSegmentedColormap object at 0x7fdb87936670>`*, **`min_px`**=*`-1100`*, **`max_px`**=*`None`*, **`ax`**=*`None`*, **`figsize`**=*`None`*, **`title`**=*`None`*, **`ctx`**=*`None`*, **`norm`**=*`None`*, **`aspect`**=*`None`*, **`interpolation`**=*`None`*, **`alpha`**=*`None`*, **`vmin`**=*`None`*, **`vmax`**=*`None`*, **`origin`**=*`None`*, **`extent`**=*`None`*, **`filternorm`**=*`True`*, **`filterrad`**=*`4.0`*, **`resample`**=*`None`*, **`url`**=*`None`*, **`data`**=*`None`*)




```python
scales = False, True, dicom_windows.brain, dicom_windows.subdural
titles = 'raw','normalized','brain windowed','subdural windowed'
for s,a,t in zip(scales, subplots(2,2,imsize=4)[1].flat, titles):
    dcm.show(scale=s, ax=a, title=t)
```


![png](output_39_0.png)


```python
dcm.show(cmap=plt.cm.gist_ncar, figsize=(6,6))
```


![png](output_40_0.png)



<h4 id="Dataset.pct_in_window" class="doc_header"><code>Dataset.pct_in_window</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L163" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.pct_in_window</code>(**`dcm`**:`Dataset`, **`w`**, **`l`**)

% of pixels in the window `(w,l)`


```python
dcm.pct_in_window(*dicom_windows.brain)
```




    0.19049072265625




<h4 id="uniform_blur2d" class="doc_header"><code>uniform_blur2d</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L170" class="source_link" style="float:right">[source]</a></h4>

> <code>uniform_blur2d</code>(**`x`**, **`s`**)




```python
ims = dcm.hist_scaled(), uniform_blur2d(dcm.hist_scaled(),50)
show_images(ims, titles=('orig', 'blurred'))
```


![png](output_46_0.png)



<h4 id="gauss_blur2d" class="doc_header"><code>gauss_blur2d</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L179" class="source_link" style="float:right">[source]</a></h4>

> <code>gauss_blur2d</code>(**`x`**, **`s`**)





<h4 id="Tensor.mask_from_blur" class="doc_header"><code>Tensor.mask_from_blur</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L186" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.mask_from_blur</code>(**`x`**:`Tensor`, **`window`**, **`sigma`**=*`0.3`*, **`thresh`**=*`0.05`*, **`remove_max`**=*`True`*)





<h4 id="Dataset.mask_from_blur" class="doc_header"><code>Dataset.mask_from_blur</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L193" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.mask_from_blur</code>(**`x`**:`Dataset`, **`window`**, **`sigma`**=*`0.3`*, **`thresh`**=*`0.05`*, **`remove_max`**=*`True`*)




```python
mask = dcm.mask_from_blur(dicom_windows.brain)
wind = dcm.windowed(*dicom_windows.brain)

_,ax = subplots(1,1)
show_image(wind, ax=ax[0])
show_image(mask, alpha=0.5, cmap=plt.cm.Reds, ax=ax[0]);
```


![png](output_53_0.png)



<h4 id="mask2bbox" class="doc_header"><code>mask2bbox</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L209" class="source_link" style="float:right">[source]</a></h4>

> <code>mask2bbox</code>(**`mask`**)




```python
bbs = mask2bbox(mask)
lo,hi = bbs
show_image(wind[lo[0]:hi[0],lo[1]:hi[1]]);
```


![png](output_57_0.png)



<h4 id="crop_resize" class="doc_header"><code>crop_resize</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L228" class="source_link" style="float:right">[source]</a></h4>

> <code>crop_resize</code>(**`x`**, **`crops`**, **`new_sz`**)




```python
px256 = crop_resize(to_device(wind[None]), bbs[...,None], 128)[0]
show_image(px256)
px256.shape
```




    torch.Size([1, 128, 128])




![png](output_61_1.png)



<h4 id="Tensor.to_nchan" class="doc_header"><code>Tensor.to_nchan</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L241" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.to_nchan</code>(**`x`**:`Tensor`, **`wins`**, **`bins`**=*`None`*)





<h4 id="Dataset.to_nchan" class="doc_header"><code>Dataset.to_nchan</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L249" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.to_nchan</code>(**`x`**:`Dataset`, **`wins`**, **`bins`**=*`None`*)





<h4 id="Tensor.to_3chan" class="doc_header"><code>Tensor.to_3chan</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L254" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.to_3chan</code>(**`x`**:`Tensor`, **`win1`**, **`win2`**, **`bins`**=*`None`*)





<h4 id="Dataset.to_3chan" class="doc_header"><code>Dataset.to_3chan</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L259" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.to_3chan</code>(**`x`**:`Dataset`, **`win1`**, **`win2`**, **`bins`**=*`None`*)




```python
show_images(dcm.to_nchan([dicom_windows.brain,dicom_windows.subdural,dicom_windows.abdomen_soft]))
```


![png](output_70_0.png)



<h4 id="Tensor.save_jpg" class="doc_header"><code>Tensor.save_jpg</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L264" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.save_jpg</code>(**`x`**:`Dataset'>)`, **`path`**, **`wins`**, **`bins`**=*`None`*, **`quality`**=*`90`*)





<h4 id="Dataset.save_jpg" class="doc_header"><code>Dataset.save_jpg</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L264" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.save_jpg</code>(**`x`**:`Dataset'>)`, **`path`**, **`wins`**, **`bins`**=*`None`*, **`quality`**=*`90`*)





<h4 id="Tensor.to_uint16" class="doc_header"><code>Tensor.to_uint16</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L272" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.to_uint16</code>(**`x`**:`Dataset'>)`, **`bins`**=*`None`*)





<h4 id="Dataset.to_uint16" class="doc_header"><code>Dataset.to_uint16</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L272" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.to_uint16</code>(**`x`**:`Dataset'>)`, **`bins`**=*`None`*)





<h4 id="Tensor.save_tif16" class="doc_header"><code>Tensor.save_tif16</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L278" class="source_link" style="float:right">[source]</a></h4>

> <code>Tensor.save_tif16</code>(**`x`**:`Dataset'>)`, **`path`**, **`bins`**=*`None`*, **`compress`**=*`True`*)





<h4 id="Dataset.save_tif16" class="doc_header"><code>Dataset.save_tif16</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L278" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.save_tif16</code>(**`x`**:`Dataset'>)`, **`path`**, **`bins`**=*`None`*, **`compress`**=*`True`*)




```python
_,axs=subplots(1,2)
with tempfile.TemporaryDirectory() as f:
    f = Path(f)
    dcm.save_jpg(f/'test.jpg', [dicom_windows.brain,dicom_windows.subdural])
    show_image(Image.open(f/'test.jpg'), ax=axs[0])
    dcm.save_tif16(f/'test.tif')
    show_image(Image.open(str(f/'test.tif')), ax=axs[1]);
```


![png](output_80_0.png)



<h4 id="Dataset.set_pixels" class="doc_header"><code>Dataset.set_pixels</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L284" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.set_pixels</code>(**`px`**)





<h4 id="Dataset.zoom" class="doc_header"><code>Dataset.zoom</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L291" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.zoom</code>(**`ratio`**)





<h4 id="Dataset.zoom_to" class="doc_header"><code>Dataset.zoom_to</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L298" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.zoom_to</code>(**`sz`**)





<h4 id="shape" class="doc_header"><code>shape</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/core.py#L34" class="source_link" style="float:right">[source]</a></h4>

> <code>shape</code>(**`x`**:`Image`)




```python
dcm2 = TEST_DCM.dcmread()
dcm2.zoom_to(90)
test_eq(dcm2.shape, (90,90))
```

```python
dcm2 = TEST_DCM.dcmread()
dcm2.zoom(0.25)
dcm2.show()
```


![png](output_90_0.png)



<h4 id="Dataset.as_dict" class="doc_header"><code>Dataset.as_dict</code><a href="https://github.com/fastai/fastai/tree/master/fastai/medical/imaging.py#L321" class="source_link" style="float:right">[source]</a></h4>

> <code>Dataset.as_dict</code>(**`px_summ`**=*`True`*, **`window`**=*`(80, 40)`*)



