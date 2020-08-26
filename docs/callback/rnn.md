# Callback for RNN training
> Callback that uses the outputs of language models to add AR and TAR regularization



<h2 id="ModelResetter" class="doc_header"><code>class</code> <code>ModelResetter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/rnn.py#L10" class="source_link" style="float:right">[source]</a></h2>

> <code>ModelResetter</code>(**`before_fit`**=*`None`*, **`before_epoch`**=*`None`*, **`before_train`**=*`None`*, **`before_batch`**=*`None`*, **`after_pred`**=*`None`*, **`after_loss`**=*`None`*, **`before_backward`**=*`None`*, **`after_backward`**=*`None`*, **`after_step`**=*`None`*, **`after_cancel_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_cancel_train`**=*`None`*, **`after_train`**=*`None`*, **`before_validate`**=*`None`*, **`after_cancel_validate`**=*`None`*, **`after_validate`**=*`None`*, **`after_cancel_epoch`**=*`None`*, **`after_epoch`**=*`None`*, **`after_cancel_fit`**=*`None`*, **`after_fit`**=*`None`*) :: [`Callback`](/callback.core.html#Callback)

[`Callback`](/callback.core.html#Callback) that resets the model at each validation/training step



<h2 id="RNNRegularizer" class="doc_header"><code>class</code> <code>RNNRegularizer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/callback/rnn.py#L23" class="source_link" style="float:right">[source]</a></h2>

> <code>RNNRegularizer</code>(**`alpha`**=*`0.0`*, **`beta`**=*`0.0`*) :: [`Callback`](/callback.core.html#Callback)

[`Callback`](/callback.core.html#Callback) that adds AR and TAR regularization in RNN training

