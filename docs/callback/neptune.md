# Neptune.ai
> Integration with <a href='https://www.neptune.ai'>neptune.ai</a>.


## Registration
1. Create **free** account: [neptune.ai/register](https://neptune.ai/register).
2. Export API token to the environment variable (more help [here](https://docs.neptune.ai/python-api/tutorials/get-started.html#copy-api-token)). In your terminal run:

```
export NEPTUNE_API_TOKEN='YOUR_LONG_API_TOKEN'
```

or append the command above to your `~/.bashrc` or `~/.bash_profile` files (**recommended**). More help is [here](https://docs.neptune.ai/python-api/tutorials/get-started.html#copy-api-token).

## Installation
1. You need to install neptune-client. In your terminal run:

```
pip install neptune-client
```

or (alternative installation using conda). In your terminal run:

```
conda install neptune-client -c conda-forge
```
2. Install [psutil](https://psutil.readthedocs.io/en/latest/) to see hardware monitoring charts:

```
pip install psutil
```

## How to use?
Key is to call `neptune.init()` before you create `Learner()` and call `neptune_create_experiment()`, before you fit the model.

Use [`NeptuneCallback`](/callback.neptune.html#NeptuneCallback) in your [`Learner`](/learner.html#Learner), like this:

```
from fastai.callback.neptune import NeptuneCallback

neptune.init('USERNAME/PROJECT_NAME')  # specify project

learn = Learner(dls, model,
                cbs=NeptuneCallback()
                )

neptune.create_experiment()  # start experiment
learn.fit_one_cycle(1)
```


<h2 id="NeptuneCallback" class="doc_header"><code>class</code> <code>NeptuneCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/neptune.py#L14" class="source_link" style="float:right">[source]</a></h2>

> <code>NeptuneCallback</code>(**`log_model_weights`**=*`True`*, **`keep_experiment_running`**=*`False`*) :: [`Callback`](/callback.core.html#Callback)

Log losses, metrics, model weights, model architecture summary to neptune

