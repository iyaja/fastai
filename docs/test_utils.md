# Synthetic Learner
> For quick testing of the training loop and Callbacks



<h4 id="synth_dbunch" class="doc_header"><code>synth_dbunch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/test_utils.py#L15" class="source_link" style="float:right">[source]</a></h4>

> <code>synth_dbunch</code>(**`a`**=*`2`*, **`b`**=*`3`*, **`bs`**=*`16`*, **`n_train`**=*`10`*, **`n_valid`**=*`2`*, **`cuda`**=*`False`*)





<h2 id="RegModel" class="doc_header"><code>class</code> <code>RegModel</code><a href="https://github.com/fastai/fastai/tree/master/fastai/test_utils.py#L27" class="source_link" style="float:right">[source]</a></h2>

> <code>RegModel</code>() :: [`Module`](/torch_core.html#Module)

Same as `nn.Module`, but no need for subclasses to call `super().__init__`



<h4 id="synth_learner" class="doc_header"><code>synth_learner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/test_utils.py#L32" class="source_link" style="float:right">[source]</a></h4>

> <code>synth_learner</code>(**`n_trn`**=*`10`*, **`n_val`**=*`2`*, **`cuda`**=*`False`*, **`lr`**=*`0.001`*, **`data`**=*`None`*, **`model`**=*`None`*, **`loss_func`**=*`None`*, **`opt_func`**=*`Adam`*, **`splitter`**=*`trainable_params`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*, **`moms`**=*`(0.95, 0.85, 0.95)`*)





<h2 id="VerboseCallback" class="doc_header"><code>class</code> <code>VerboseCallback</code><a href="https://github.com/fastai/fastai/tree/master/fastai/test_utils.py#L40" class="source_link" style="float:right">[source]</a></h2>

> <code>VerboseCallback</code>(**`before_fit`**=*`None`*, **`before_epoch`**=*`None`*, **`before_train`**=*`None`*, **`before_batch`**=*`None`*, **`after_pred`**=*`None`*, **`after_loss`**=*`None`*, **`before_backward`**=*`None`*, **`after_backward`**=*`None`*, **`after_step`**=*`None`*, **`after_cancel_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_cancel_train`**=*`None`*, **`after_train`**=*`None`*, **`before_validate`**=*`None`*, **`after_cancel_validate`**=*`None`*, **`after_validate`**=*`None`*, **`after_cancel_epoch`**=*`None`*, **`after_epoch`**=*`None`*, **`after_cancel_fit`**=*`None`*, **`after_fit`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

Callback that prints the name of each event called


## - Export
