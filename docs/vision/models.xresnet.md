# XResnet
> Resnet from bags of tricks paper



<h4 id="init_cnn" class="doc_header"><code>init_cnn</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L16" class="source_link" style="float:right">[source]</a></h4>

> <code>init_cnn</code>(**`m`**)





<h2 id="XResNet" class="doc_header"><code>class</code> <code>XResNet</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L22" class="source_link" style="float:right">[source]</a></h2>

> <code>XResNet</code>(**`block`**, **`expansion`**, **`layers`**, **`p`**=*`0.0`*, **`c_in`**=*`3`*, **`n_out`**=*`1000`*, **`stem_szs`**=*`(32, 32, 64)`*, **`widen`**=*`1.0`*, **`sa`**=*`False`*, **`act_cls`**=*`ReLU`*, **`stride`**=*`1`*, **`groups`**=*`1`*, **`reduction`**=*`None`*, **`nh1`**=*`None`*, **`nh2`**=*`None`*, **`dw`**=*`False`*, **`g2`**=*`1`*, **`sym`**=*`False`*, **`norm_type`**=*`<NormType.Batch: 1>`*, **`ndim`**=*`2`*, **`ks`**=*`3`*, **`pool`**=*`AvgPool`*, **`pool_first`**=*`True`*, **`padding`**=*`None`*, **`bias`**=*`None`*, **`bn_1st`**=*`True`*, **`transpose`**=*`False`*, **`init`**=*`'auto'`*, **`xtra`**=*`None`*, **`bias_std`**=*`0.01`*, **`dilation`**:`Union`\[`int`, `Tuple`\[`int`, `int`\]\]=*`1`*, **`padding_mode`**:`str`=*`'zeros'`*) :: `Sequential`

A sequential container.
Modules will be added to it in the order they are passed in the constructor.
Alternatively, an ordered dict of modules can also be passed in.

To make it easier to understand, here is a small example::

    # Example of using Sequential
    model = nn.Sequential(
              nn.Conv2d(1,20,5),
              nn.ReLU(),
              nn.Conv2d(20,64,5),
              nn.ReLU()
            )

    # Example of using Sequential with OrderedDict
    model = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1,20,5)),
              ('relu1', nn.ReLU()),
              ('conv2', nn.Conv2d(20,64,5)),
              ('relu2', nn.ReLU())
            ]))



<h4 id="xresnet18" class="doc_header"><code>xresnet18</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L62" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet18</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnet34" class="doc_header"><code>xresnet34</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L63" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet34</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnet50" class="doc_header"><code>xresnet50</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L64" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet50</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnet101" class="doc_header"><code>xresnet101</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L65" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet101</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnet152" class="doc_header"><code>xresnet152</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L66" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet152</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnet18_deep" class="doc_header"><code>xresnet18_deep</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L67" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet18_deep</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnet34_deep" class="doc_header"><code>xresnet34_deep</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L68" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet34_deep</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnet50_deep" class="doc_header"><code>xresnet50_deep</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L69" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet50_deep</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnet18_deeper" class="doc_header"><code>xresnet18_deeper</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L70" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet18_deeper</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnet34_deeper" class="doc_header"><code>xresnet34_deeper</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L71" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet34_deeper</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnet50_deeper" class="doc_header"><code>xresnet50_deeper</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L72" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnet50_deeper</code>(**`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnet18" class="doc_header"><code>xse_resnet18</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L84" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnet18</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnext18" class="doc_header"><code>xse_resnext18</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L85" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnext18</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnext18" class="doc_header"><code>xresnext18</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L86" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnext18</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnet34" class="doc_header"><code>xse_resnet34</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L87" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnet34</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnext34" class="doc_header"><code>xse_resnext34</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L88" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnext34</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnext34" class="doc_header"><code>xresnext34</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L89" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnext34</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnet50" class="doc_header"><code>xse_resnet50</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L90" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnet50</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnext50" class="doc_header"><code>xse_resnext50</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L91" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnext50</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnext50" class="doc_header"><code>xresnext50</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L92" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnext50</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnet101" class="doc_header"><code>xse_resnet101</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L93" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnet101</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnext101" class="doc_header"><code>xse_resnext101</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L94" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnext101</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xresnext101" class="doc_header"><code>xresnext101</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L95" class="source_link" style="float:right">[source]</a></h4>

> <code>xresnext101</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnet152" class="doc_header"><code>xse_resnet152</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L96" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnet152</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xsenet154" class="doc_header"><code>xsenet154</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L97" class="source_link" style="float:right">[source]</a></h4>

> <code>xsenet154</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnext18_deep" class="doc_header"><code>xse_resnext18_deep</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L99" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnext18_deep</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnext34_deep" class="doc_header"><code>xse_resnext34_deep</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L100" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnext34_deep</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnext50_deep" class="doc_header"><code>xse_resnext50_deep</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L101" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnext50_deep</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnext18_deeper" class="doc_header"><code>xse_resnext18_deeper</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L102" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnext18_deeper</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnext34_deeper" class="doc_header"><code>xse_resnext34_deeper</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L103" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnext34_deeper</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)





<h4 id="xse_resnext50_deeper" class="doc_header"><code>xse_resnext50_deeper</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/models/xresnet.py#L104" class="source_link" style="float:right">[source]</a></h4>

> <code>xse_resnext50_deeper</code>(**`n_out`**=*`1000`*, **`pretrained`**=*`False`*, **\*\*`kwargs`**)




```python
tst = xse_resnext18()
x = torch.randn(64, 3, 128, 128)
y = tst(x)
```

```python
tst = xresnext18()
x = torch.randn(64, 3, 128, 128)
y = tst(x)
```

```python
tst = xse_resnet50()
x = torch.randn(8, 3, 64, 64)
y = tst(x)
```
