# Tutorial - Migrating from Ignite
> Incrementally adding fastai goodness to your Ignite training


We're going to use the MNIST training code from Ignite's examples directory (as at August 2020), converted to a module.

```python
from migrating_ignite import *

from fastai.vision.all import *
```

To use it in fastai, we first pull the DataLoaders from the module into a [`DataLoaders`](/data.core.html#DataLoaders) object:

```python
data = DataLoaders(*get_data_loaders(64, 128)).cuda()
```

We can now create a [`Learner`](/learner.html#Learner) and fit:

```python
opt_func = partial(SGD, momentum=0.5)
learn = Learner(data, Net(), loss_func=nn.NLLLoss(), opt_func=opt_func, metrics=accuracy)
learn.fit_one_cycle(1, 0.01)
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
      <td>0.999266</td>
      <td>0.597913</td>
      <td>0.856200</td>
      <td>00:22</td>
    </tr>
  </tbody>
</table>


As you can see, migrating from Ignite allowed us to replace 52 lines of code (in `run()`) with just 3 lines, and doesn't require you to change any of your existing data pipelines, optimizers, loss functions, models, etc. Once you've made this change, you can then benefit from fastai's rich set of callbacks, transforms, visualizations, and so forth.

Note that fastai is very different from Ignite, in that it is much more than just a training loop (although we're only using the training loop in this example) - it is a complete framework including GPU-accelerated transformations, end-to-end inference, integrated applications for vision, text, tabular, and collaborative filtering, and so forth. You can use any part of the framework on its own, or combine them together, as described in the [fastai paper](https://arxiv.org/abs/2002.04688).
