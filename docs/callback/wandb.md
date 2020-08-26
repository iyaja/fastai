# Wandb
> Integration with <a href='https://docs.wandb.com/library/integrations/fastai'>Weights & Biases</a> 


First thing first, you need to install wandb with
```
pip install wandb
```
Create a free account then run 
``` 
wandb login
```
in your terminal. Follow the link to get an API token that you will need to paste, then you're all set!


<h2 id="WandbCallback" class="doc_header"><code>class</code> <code>WandbCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/wandb.py#L17" class="source_link" style="float:right">[source]</a></h2>

> <code>WandbCallback</code>(**`log`**=*`'gradients'`*, **`log_preds`**=*`True`*, **`log_model`**=*`True`*, **`log_dataset`**=*`False`*, **`dataset_name`**=*`None`*, **`valid_dl`**=*`None`*, **`n_preds`**=*`36`*, **`seed`**=*`12345`*) :: [`Callback`](/callback.core.html#Callback)

Saves model topology, losses & metrics


Optionally logs weights and or gradients depending on `log` (can be "gradients", "parameters", "all" or None), sample predictions if ` log_preds=True` that will come from `valid_dl` or a random sample pf the validation set (determined by `seed`). `n_preds` are logged in this case.

If used in combination with [`SaveModelCallback`](/callback.tracker.html#SaveModelCallback), the best model is saved as well (can be deactivated with `log_model=False`).

Datasets can also be tracked:
* if [`log_dataset`](/callback.wandb.html#log_dataset) is `True`, tracked folder is retrieved from `learn.dls.path`
* [`log_dataset`](/callback.wandb.html#log_dataset) can explicitly be set to the folder to track
* the name of the dataset can explicitly be given through `dataset_name`, otherwise it is set to the folder name
* *Note: the subfolder "models" is always ignored*

For custom scenarios, you can also manually use functions [`log_dataset`](/callback.wandb.html#log_dataset) and [`log_model`](/callback.wandb.html#log_model) to respectively log your own datasets and models.


<h4 id="log_dataset" class="doc_header"><code>log_dataset</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/wandb.py#L151" class="source_link" style="float:right">[source]</a></h4>

> <code>log_dataset</code>(**`path`**, **`name`**=*`None`*, **`metadata`**=*`{}`*)

Log dataset folder



<h4 id="log_model" class="doc_header"><code>log_model</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/wandb.py#L170" class="source_link" style="float:right">[source]</a></h4>

> <code>log_model</code>(**`path`**, **`name`**=*`None`*, **`metadata`**=*`{}`*)

Log model file


## Example of use:

Once your have defined your [`Learner`](/learner.html#Learner), before you call to `fit` or `fit_one_cycle`, you need to initialize wandb:
```
import wandb
wandb.init()
```
To use Weights & Biases without an account, you can call `wandb.init(anonymous='allow')`.

Then you add the callback to your [`learner`](/learner.html) or call to `fit` methods, potentially with [`SaveModelCallback`](/callback.tracker.html#SaveModelCallback) if you want to save the best model:
```
from fastai.callback.wandb import *

# To log only during one training phase
learn.fit(..., cbs=WandbCallback())

# To log continuously for all training phases
learn = learner(..., cbs=WandbCallback())
```
Datasets and models can be tracked through the callback or directly through [`log_model`](/callback.wandb.html#log_model) and [`log_dataset`](/callback.wandb.html#log_dataset) functions.

For more details, refer to [W&B documentation](https://docs.wandb.com/library/integrations/fastai).
