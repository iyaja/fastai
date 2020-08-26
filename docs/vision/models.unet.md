# Dynamic UNet
> Unet model using PixelShuffle ICNR upsampling that can be built on top of any pretrained architecture



<h2 id="UnetBlock" class="doc_header"><code>class</code> <code>UnetBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/unet.py#L18" class="source_link" style="float:right">[source]</a></h2>

> <code>UnetBlock</code>(**`up_in_c`**, **`x_in_c`**, **`hook`**, **`final_div`**=*`True`*, **`blur`**=*`False`*, **`act_cls`**=*`ReLU`*, **`self_attention`**=*`False`*, **`init`**=*`kaiming_normal_`*, **`norm_type`**=*`None`*, **`ks`**=*`3`*, **`stride`**=*`1`*, **`padding`**=*`None`*, **`bias`**=*`None`*, **`ndim`**=*`2`*, **`bn_1st`**=*`True`*, **`transpose`**=*`False`*, **`xtra`**=*`None`*, **`bias_std`**=*`0.01`*, **`dilation`**:`Union`\[`int`, `Tuple`\[`int`, `int`\]\]=*`1`*, **`groups`**:`int`=*`1`*, **`padding_mode`**:`str`=*`'zeros'`*) :: [`Module`](/torch_core.html#Module)

A quasi-UNet block, using `PixelShuffle_ICNR upsampling`.



<h2 id="ResizeToOrig" class="doc_header"><code>class</code> <code>ResizeToOrig</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/unet.py#L44" class="source_link" style="float:right">[source]</a></h2>

> <code>ResizeToOrig</code>(**`mode`**=*`'nearest'`*) :: [`Module`](/torch_core.html#Module)

Merge a shortcut with the result of the module by adding them or concatenating them if `dense=True`.



<h2 id="DynamicUnet" class="doc_header"><code>class</code> <code>DynamicUnet</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/unet.py#L53" class="source_link" style="float:right">[source]</a></h2>

> <code>DynamicUnet</code>(**`encoder`**, **`n_classes`**, **`img_size`**, **`blur`**=*`False`*, **`blur_final`**=*`True`*, **`self_attention`**=*`False`*, **`y_range`**=*`None`*, **`last_cross`**=*`True`*, **`bottle`**=*`False`*, **`act_cls`**=*`ReLU`*, **`init`**=*`kaiming_normal_`*, **`norm_type`**=*`None`*, **\*\*`kwargs`**) :: [`SequentialEx`](/layers.html#SequentialEx)

Create a U-Net from a given architecture.


```python
from fastai.vision.models import resnet34
```

```python
m = resnet34()
m = nn.Sequential(*list(m.children())[:-2])
tst = DynamicUnet(m, 5, (128,128), norm_type=None)
x = torch.randn(2, 3, 128, 128)
y = tst(x)
test_eq(y.shape, [2, 5, 128, 128])
```

```python
tst = DynamicUnet(m, 5, (128,128), norm_type=None)
x = torch.randn(2, 3, 127, 128)
y = tst(x)
```
