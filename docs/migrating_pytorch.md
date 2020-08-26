# Tutorial - Migrating from pure PyTorch
> Incrementally adding fastai goodness to your PyTorch models


```python
from fastai.vision.all import *
```

We're going to use the MNIST training code from the official PyTorch examples, slightly reformatted for space, updated from AdaDelta to AdamW, and converted from a script to a module. There's a lot of code, so we've put it into migrating_pytorch.py!

```python
from migrating_pytorch import *
```

We can entirely replace the custom training loop with fastai's. That means you can get rid of `train()`, `test()`, and the epoch loop in the original code, and replace it all with just this:

```python
data = DataLoaders(train_loader, test_loader)
learn = Learner(data, Net(), loss_func=F.nll_loss, opt_func=Adam, metrics=accuracy, cbs=CudaCallback)
```

We also added [`CudaCallback`](/callback.data.html#CudaCallback) to have the model and data moved to the GPU for us. Alternatively, you can use the fastai [`DataLoader`](/data.load.html#DataLoader), which provides a superset of the functionality of PyTorch's (with the same API), and can handle moving data to the GPU for us (see `migrating_ignite.ipynb` for an example of this approach). 

fastai supports many schedulers. We recommend fitting with 1cycle training:

```python
learn.fit_one_cycle(epochs, lr)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.129090</td>
      <td>0.052289</td>
      <td>0.982600</td>
      <td>00:17</td>
    </tr>
  </tbody>
</table>


As you can see, migrating from pure PyTorch allows you to remove a lot of code, and doesn't require you to change any of your existing data pipelines, optimizers, loss functions, models, etc.

Once you've made this change, you can then benefit from fastai's rich set of callbacks, transforms, visualizations, and so forth.

Note that fastai much more than just a training loop (although we're only using the training loop in this example) - it is a complete framework including GPU-accelerated transformations, end-to-end inference, integrated applications for vision, text, tabular, and collaborative filtering, and so forth. You can use any part of the framework on its own, or combine them together, as described in the [fastai paper](https://arxiv.org/abs/2002.04688).
