# Set up PyTorch links to appear in fastai docs



```python
from fastai.basics import *
```

Test links

```python
test_eq(_mod2page(Tensor), 'tensors.html')
test_eq(_mod2page(torch.nn), 'nn.html')
test_eq(_mod2page(inspect.getmodule(nn.Conv2d)), 'nn.html')
test_eq(_mod2page(F), 'nn.functional.html')
test_eq(_mod2page(torch.optim), 'optim.html')
test_eq(_mod2page(torch.utils.data), 'data.html')
```


<h4 id="pytorch_doc_link" class="doc_header"><code>pytorch_doc_link</code><a href="https://github.com/fastai/fastai/tree/master/fastai/_pytorch_doc.py#L20" class="source_link" style="float:right">[source]</a></h4>

> <code>pytorch_doc_link</code>(**`name`**)




```python
test_links = {
    'Tensor': 'https://pytorch.org/docs/stable/tensors.html',
    'Tensor.sqrt': 'https://pytorch.org/docs/stable/tensors.html#torch.Tensor.sqrt',
    'torch.zeros_like': 'https://pytorch.org/docs/stable/torch.html#torch.zeros_like',
    'nn.Module': 'https://pytorch.org/docs/stable/nn.html#torch.nn.Module',
    'nn.Linear': 'https://pytorch.org/docs/stable/nn.html#torch.nn.Linear',
    'F.cross_entropy': 'https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.cross_entropy'
}
for f,l in test_links.items(): test_eq(pytorch_doc_link(f), l)
```
