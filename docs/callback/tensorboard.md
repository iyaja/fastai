# Tensorboard
> Integration with <a href='https://www.tensorflow.org/tensorboard'>tensorboard</a> 


First thing first, you need to install tensorboard with
```
pip install tensoarboard
```
Then launch tensorboard with
``` 
tensorboard --logdir=runs
```
in your terminal. You can change the logdir as long as it matches the `log_dir` you pass to [`TensorBoardCallback`](/callback.tensorboard.html#TensorBoardCallback) (default is `runs` in the working directory).


<h2 id="TensorBoardCallback" class="doc_header"><code>class</code> <code>TensorBoardCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/tensorboard.py#L14" class="source_link" style="float:right">[source]</a></h2>

> <code>TensorBoardCallback</code>(**`log_dir`**=*`None`*, **`trace_model`**=*`True`*, **`log_preds`**=*`True`*, **`n_preds`**=*`9`*) :: [`Callback`](/callback.core.html#Callback)

Saves model topology, losses & metrics


## Test

```python
#from fastai.callback.all import *
```

```python
#                 get_items=get_image_files, 
#                 splitter=RandomSplitter(),
#                 get_y=RegexLabeller(pat = r'/([^/]+)_\d+.jpg$'))
```

```python
#                        batch_tfms=[*aug_transforms(size=299, max_warp=0), Normalize.from_stats(*imagenet_stats)])
```

```python

```

```python

```

```python

```
