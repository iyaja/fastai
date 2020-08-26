# Metrics
> Definition of the metrics that can be used in training models


## Core metric

This is where the function that converts scikit-learn metrics to fastai metrics is defined. You should skip this section unless you want to know all about the internals of fastai.


<h4 id="flatten_check" class="doc_header"><code>flatten_check</code><a href="https://github.com/fastai/fastai/tree/master/fastai/torch_core.py#L747" class="source_link" style="float:right">[source]</a></h4>

> <code>flatten_check</code>(**`inp`**, **`targ`**)

Check that `out` and `targ` have the same number of elements and flatten them.


```python
x1,x2 = torch.randn(5,4),torch.randn(20)
x1,x2 = flatten_check(x1,x2)
test_eq(x1.shape, [20])
test_eq(x2.shape, [20])
x1,x2 = torch.randn(5,4),torch.randn(21)
test_fail(lambda: flatten_check(x1,x2))
```


<h3 id="AccumMetric" class="doc_header"><code>class</code> <code>AccumMetric</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L27" class="source_link" style="float:right">[source]</a></h3>

> <code>AccumMetric</code>(**`func`**, **`dim_argmax`**=*`None`*, **`activation`**=*`'no'`*, **`thresh`**=*`None`*, **`to_np`**=*`False`*, **`invert_arg`**=*`False`*, **`flatten`**=*`True`*, **\*\*`kwargs`**) :: [`Metric`](/learner.html#Metric)

Stores predictions and targets on CPU in accumulate to perform final calculations with `func`.


`func` is only applied to the accumulated predictions/targets when the `value` attribute is asked for (so at the end of a validation/training phase, in use with [`Learner`](/learner.html#Learner) and its [`Recorder`](/learner.html#Recorder)).The signature of `func` should be `inp,targ` (where `inp` are the predictions of the model and `targ` the corresponding labels).

For classification problems with single label, predictions need to be transformed with a softmax then an argmax before being compared to the targets. Since a softmax doesn't change the order of the numbers, we can just apply the argmax. Pass along `dim_argmax` to have this done by [`AccumMetric`](/metrics.html#AccumMetric) (usually -1 will work pretty well). If you need to pass to your metrics the probabilities and not the predictions, use `softmax=True`.

For classification problems with multiple labels, or if your targets are one-hot encoded, predictions may need to pass through a sigmoid (if it wasn't included in your model) then be compared to a given threshold (to decide between 0 and 1), this is done by [`AccumMetric`](/metrics.html#AccumMetric) if you pass `sigmoid=True` and/or a value for `thresh`.

If you want to use a metric function sklearn.metrics, you will need to convert predictions and labels to numpy arrays with `to_np=True`. Also, scikit-learn metrics adopt the convention `y_true`, `y_preds` which is the opposite from us, so you will need to pass `invert_arg=True` to make [`AccumMetric`](/metrics.html#AccumMetric) do the inversion for you.

```python
class TstLearner():
    def __init__(self): self.pred,self.y = None,None
```

```python
def _l2_mean(x,y): return torch.sqrt((x.float()-y.float()).pow(2).mean())

#Go through a fake cycle with various batch sizes and computes the value of met
def compute_val(met, x1, x2):
    met.reset()
    vals = [0,6,15,20]
    learn = TstLearner()
    for i in range(3): 
        learn.pred,learn.y = x1[vals[i]:vals[i+1]],x2[vals[i]:vals[i+1]]
        met.accumulate(learn)
    return met.value
```

```python
x1,x2 = torch.randn(20,5),torch.randn(20,5)
tst = AccumMetric(_l2_mean)
test_close(compute_val(tst, x1, x2), _l2_mean(x1, x2))
test_eq(torch.cat(tst.preds), x1.view(-1))
test_eq(torch.cat(tst.targs), x2.view(-1))

#test argmax
x1,x2 = torch.randn(20,5),torch.randint(0, 5, (20,))
tst = AccumMetric(_l2_mean, dim_argmax=-1)
test_close(compute_val(tst, x1, x2), _l2_mean(x1.argmax(dim=-1), x2))

#test thresh
x1,x2 = torch.randn(20,5),torch.randint(0, 2, (20,5)).bool()
tst = AccumMetric(_l2_mean, thresh=0.5)
test_close(compute_val(tst, x1, x2), _l2_mean((x1 >= 0.5), x2))

#test sigmoid
x1,x2 = torch.randn(20,5),torch.randn(20,5)
tst = AccumMetric(_l2_mean, activation=ActivationType.Sigmoid)
test_close(compute_val(tst, x1, x2), _l2_mean(torch.sigmoid(x1), x2))

#test to_np
x1,x2 = torch.randn(20,5),torch.randn(20,5)
tst = AccumMetric(lambda x,y: isinstance(x, np.ndarray) and isinstance(y, np.ndarray), to_np=True)
assert compute_val(tst, x1, x2)

#test invert_arg
x1,x2 = torch.randn(20,5),torch.randn(20,5)
tst = AccumMetric(lambda x,y: torch.sqrt(x.pow(2).mean()))
test_close(compute_val(tst, x1, x2), torch.sqrt(x1.pow(2).mean()))
tst = AccumMetric(lambda x,y: torch.sqrt(x.pow(2).mean()), invert_arg=True)
test_close(compute_val(tst, x1, x2), torch.sqrt(x2.pow(2).mean()))
```


<h4 id="skm_to_fastai" class="doc_header"><code>skm_to_fastai</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L74" class="source_link" style="float:right">[source]</a></h4>

> <code>skm_to_fastai</code>(**`func`**, **`is_class`**=*`True`*, **`thresh`**=*`None`*, **`axis`**=*`-1`*, **`activation`**=*`None`*, **\*\*`kwargs`**)

Convert `func` from sklearn.metrics to a fastai metric


This is the quickest way to use a scikit-learn metric in a fastai training loop. `is_class` indicates if you are in a classification problem or not. In this case:
- leaving `thresh` to `None` indicates it's a single-label classification problem and predictions will pass through an argmax over `axis` before being compared to the targets
- setting a value for `thresh` indicates it's a multi-label classification problem and predictions will pass through a sigmoid (can be deactivated with `sigmoid=False`) and be compared to `thresh` before being compared to the targets

If `is_class=False`, it indicates you are in a regression problem, and predictions are compared to the targets without being modified. In all cases, `kwargs` are extra keyword arguments passed to `func`.

```python
tst_single = skm_to_fastai(skm.precision_score)
x1,x2 = torch.randn(20,2),torch.randint(0, 2, (20,))
test_close(compute_val(tst_single, x1, x2), skm.precision_score(x2, x1.argmax(dim=-1)))
```

```python
tst_multi = skm_to_fastai(skm.precision_score, thresh=0.2)
x1,x2 = torch.randn(20),torch.randint(0, 2, (20,))
test_close(compute_val(tst_multi, x1, x2), skm.precision_score(x2, torch.sigmoid(x1) >= 0.2))

tst_multi = skm_to_fastai(skm.precision_score, thresh=0.2, activation=ActivationType.No)
x1,x2 = torch.randn(20),torch.randint(0, 2, (20,))
test_close(compute_val(tst_multi, x1, x2), skm.precision_score(x2, x1 >= 0.2))
```

```python
tst_reg = skm_to_fastai(skm.r2_score, is_class=False)
x1,x2 = torch.randn(20,5),torch.randn(20,5)
test_close(compute_val(tst_reg, x1, x2), skm.r2_score(x2.view(-1), x1.view(-1)))
```

```python
test_close(tst_reg(x1, x2), skm.r2_score(x2.view(-1), x1.view(-1)))
```


<h4 id="optim_metric" class="doc_header"><code>optim_metric</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L83" class="source_link" style="float:right">[source]</a></h4>

> <code>optim_metric</code>(**`f`**, **`argname`**, **`bounds`**, **`tol`**=*`0.01`*, **`do_neg`**=*`True`*, **`get_x`**=*`False`*)

Replace metric `f` with a version that optimizes argument `argname`


## Single-label classification

{% include warning.html content='All functions defined in this section are intended for single-label classification and targets that are not one-hot encoded. For multi-label problems or one-hot encoded targets, use the version suffixed with multi.' %}

{% include warning.html content='Many metrics in fastai are thin wrappers around sklearn functionality. However, sklearn metrics can handle python list strings, amongst other things, whereas fastai metrics work with PyTorch, and thus require tensors. The arguments that are passed to metrics are after all transformations, such as categories being converted to indices, have occurred. This means that when you pass a label of a metric, for instance, that you must pass indices, not strings. This can be converted with `vocab.map_obj`.' %}


<h4 id="accuracy" class="doc_header"><code>accuracy</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L98" class="source_link" style="float:right">[source]</a></h4>

> <code>accuracy</code>(**`inp`**, **`targ`**, **`axis`**=*`-1`*)

Compute accuracy with `targ` when `pred` is bs * n_classes


```python
def change_targ(targ, n, c):
    idx = torch.randperm(len(targ))[:n]
    res = targ.clone()
    for i in idx: res[i] = (res[i]+random.randint(1,c-1))%c
    return res
```

```python
x = torch.randn(4,5)
y = x.argmax(dim=1)
test_eq(accuracy(x,y), 1)
y1 = change_targ(y, 2, 5)
test_eq(accuracy(x,y1), 0.5)
test_eq(accuracy(x.unsqueeze(1).expand(4,2,5), torch.stack([y,y1], dim=1)), 0.75)
```


<h4 id="error_rate" class="doc_header"><code>error_rate</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L104" class="source_link" style="float:right">[source]</a></h4>

> <code>error_rate</code>(**`inp`**, **`targ`**, **`axis`**=*`-1`*)

1 - [`accuracy`](/metrics.html#accuracy)


```python
x = torch.randn(4,5)
y = x.argmax(dim=1)
test_eq(error_rate(x,y), 0)
y1 = change_targ(y, 2, 5)
test_eq(error_rate(x,y1), 0.5)
test_eq(error_rate(x.unsqueeze(1).expand(4,2,5), torch.stack([y,y1], dim=1)), 0.25)
```


<h4 id="top_k_accuracy" class="doc_header"><code>top_k_accuracy</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L109" class="source_link" style="float:right">[source]</a></h4>

> <code>top_k_accuracy</code>(**`inp`**, **`targ`**, **`k`**=*`5`*, **`axis`**=*`-1`*)

Computes the Top-k accuracy (`targ` is in the top `k` predictions of `inp`)


```python
x = torch.randn(6,5)
y = torch.arange(0,6)
test_eq(top_k_accuracy(x[:5],y[:5]), 1)
test_eq(top_k_accuracy(x, y), 5/6)
```


<h4 id="APScoreBinary" class="doc_header"><code>APScoreBinary</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L116" class="source_link" style="float:right">[source]</a></h4>

> <code>APScoreBinary</code>(**`axis`**=*`-1`*, **`average`**=*`'macro'`*, **`pos_label`**=*`1`*, **`sample_weight`**=*`None`*)

Average Precision for single-label binary classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) for more details.


<h4 id="BalancedAccuracy" class="doc_header"><code>BalancedAccuracy</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L122" class="source_link" style="float:right">[source]</a></h4>

> <code>BalancedAccuracy</code>(**`axis`**=*`-1`*, **`sample_weight`**=*`None`*, **`adjusted`**=*`False`*)

Balanced Accuracy for single-label binary classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score) for more details.


<h4 id="BrierScore" class="doc_header"><code>BrierScore</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L128" class="source_link" style="float:right">[source]</a></h4>

> <code>BrierScore</code>(**`axis`**=*`-1`*, **`sample_weight`**=*`None`*, **`pos_label`**=*`None`*)

Brier score for single-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss) for more details.


<h4 id="CohenKappa" class="doc_header"><code>CohenKappa</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L134" class="source_link" style="float:right">[source]</a></h4>

> <code>CohenKappa</code>(**`axis`**=*`-1`*, **`labels`**=*`None`*, **`weights`**=*`None`*, **`sample_weight`**=*`None`*)

Cohen kappa for single-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score) for more details.


<h4 id="F1Score" class="doc_header"><code>F1Score</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L140" class="source_link" style="float:right">[source]</a></h4>

> <code>F1Score</code>(**`axis`**=*`-1`*, **`labels`**=*`None`*, **`pos_label`**=*`1`*, **`average`**=*`'binary'`*, **`sample_weight`**=*`None`*)

F1 score for single-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) for more details.


<h4 id="FBeta" class="doc_header"><code>FBeta</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L146" class="source_link" style="float:right">[source]</a></h4>

> <code>FBeta</code>(**`beta`**, **`axis`**=*`-1`*, **`labels`**=*`None`*, **`pos_label`**=*`1`*, **`average`**=*`'binary'`*, **`sample_weight`**=*`None`*)

FBeta score with `beta` for single-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score) for more details.


<h4 id="HammingLoss" class="doc_header"><code>HammingLoss</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L152" class="source_link" style="float:right">[source]</a></h4>

> <code>HammingLoss</code>(**`axis`**=*`-1`*, **`sample_weight`**=*`None`*)

Hamming loss for single-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss) for more details.


<h4 id="Jaccard" class="doc_header"><code>Jaccard</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L158" class="source_link" style="float:right">[source]</a></h4>

> <code>Jaccard</code>(**`axis`**=*`-1`*, **`labels`**=*`None`*, **`pos_label`**=*`1`*, **`average`**=*`'binary'`*, **`sample_weight`**=*`None`*)

Jaccard score for single-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score) for more details.


<h4 id="Precision" class="doc_header"><code>Precision</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L164" class="source_link" style="float:right">[source]</a></h4>

> <code>Precision</code>(**`axis`**=*`-1`*, **`labels`**=*`None`*, **`pos_label`**=*`1`*, **`average`**=*`'binary'`*, **`sample_weight`**=*`None`*)

Precision for single-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) for more details.


<h4 id="Recall" class="doc_header"><code>Recall</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L170" class="source_link" style="float:right">[source]</a></h4>

> <code>Recall</code>(**`axis`**=*`-1`*, **`labels`**=*`None`*, **`pos_label`**=*`1`*, **`average`**=*`'binary'`*, **`sample_weight`**=*`None`*)

Recall for single-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) for more details.


<h4 id="RocAuc" class="doc_header"><code>RocAuc</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L176" class="source_link" style="float:right">[source]</a></h4>

> <code>RocAuc</code>(**`axis`**=*`-1`*, **`average`**=*`'macro'`*, **`sample_weight`**=*`None`*, **`max_fpr`**=*`None`*, **`multi_class`**=*`'ovr'`*)

Area Under the Receiver Operating Characteristic Curve for single-label multiclass classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) for more details.


<h4 id="RocAucBinary" class="doc_header"><code>RocAucBinary</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L183" class="source_link" style="float:right">[source]</a></h4>

> <code>RocAucBinary</code>(**`axis`**=*`-1`*, **`average`**=*`'macro'`*, **`sample_weight`**=*`None`*, **`max_fpr`**=*`None`*, **`multi_class`**=*`'raise'`*)

Area Under the Receiver Operating Characteristic Curve for single-label binary classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) for more details.


<h4 id="MatthewsCorrCoef" class="doc_header"><code>MatthewsCorrCoef</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L189" class="source_link" style="float:right">[source]</a></h4>

> <code>MatthewsCorrCoef</code>(**`sample_weight`**=*`None`*, **\*\*`kwargs`**)

Matthews correlation coefficient for single-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef) for more details.


<h3 id="Perplexity" class="doc_header"><code>class</code> <code>Perplexity</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L194" class="source_link" style="float:right">[source]</a></h3>

> <code>Perplexity</code>() :: [`AvgLoss`](/learner.html#AvgLoss)

Perplexity (exponential of cross-entropy loss) for Language Models


```python
x1,x2 = torch.randn(20,5),torch.randint(0, 5, (20,))
tst = perplexity
tst.reset()
vals = [0,6,15,20]
learn = TstLearner()
for i in range(3): 
    learn.y,learn.yb = x2[vals[i]:vals[i+1]],(x2[vals[i]:vals[i+1]],)
    learn.loss = F.cross_entropy(x1[vals[i]:vals[i+1]],x2[vals[i]:vals[i+1]])
    tst.accumulate(learn)
test_close(tst.value, torch.exp(F.cross_entropy(x1,x2)))
```

## Multi-label classification


<h4 id="accuracy_multi" class="doc_header"><code>accuracy_multi</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L204" class="source_link" style="float:right">[source]</a></h4>

> <code>accuracy_multi</code>(**`inp`**, **`targ`**, **`thresh`**=*`0.5`*, **`sigmoid`**=*`True`*)

Compute accuracy when `inp` and `targ` are the same size.


```python
def change_1h_targ(targ, n):
    idx = torch.randperm(targ.numel())[:n]
    res = targ.clone().view(-1)
    for i in idx: res[i] = 1-res[i]
    return res.view(targ.shape)
```

```python
x = torch.randn(4,5)
y = (torch.sigmoid(x) >= 0.5).byte()
test_eq(accuracy_multi(x,y), 1)
test_eq(accuracy_multi(x,1-y), 0)
y1 = change_1h_targ(y, 5)
test_eq(accuracy_multi(x,y1), 0.75)

#Different thresh
y = (torch.sigmoid(x) >= 0.2).byte()
test_eq(accuracy_multi(x,y, thresh=0.2), 1)
test_eq(accuracy_multi(x,1-y, thresh=0.2), 0)
y1 = change_1h_targ(y, 5)
test_eq(accuracy_multi(x,y1, thresh=0.2), 0.75)

#No sigmoid
y = (x >= 0.5).byte()
test_eq(accuracy_multi(x,y, sigmoid=False), 1)
test_eq(accuracy_multi(x,1-y, sigmoid=False), 0)
y1 = change_1h_targ(y, 5)
test_eq(accuracy_multi(x,y1, sigmoid=False), 0.75)
```


<h4 id="APScoreMulti" class="doc_header"><code>APScoreMulti</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L211" class="source_link" style="float:right">[source]</a></h4>

> <code>APScoreMulti</code>(**`sigmoid`**=*`True`*, **`average`**=*`'macro'`*, **`pos_label`**=*`1`*, **`sample_weight`**=*`None`*)

Average Precision for multi-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) for more details.


<h4 id="BrierScoreMulti" class="doc_header"><code>BrierScoreMulti</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L218" class="source_link" style="float:right">[source]</a></h4>

> <code>BrierScoreMulti</code>(**`thresh`**=*`0.5`*, **`sigmoid`**=*`True`*, **`sample_weight`**=*`None`*, **`pos_label`**=*`None`*)

Brier score for multi-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss) for more details.


<h4 id="F1ScoreMulti" class="doc_header"><code>F1ScoreMulti</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L225" class="source_link" style="float:right">[source]</a></h4>

> <code>F1ScoreMulti</code>(**`thresh`**=*`0.5`*, **`sigmoid`**=*`True`*, **`labels`**=*`None`*, **`pos_label`**=*`1`*, **`average`**=*`'macro'`*, **`sample_weight`**=*`None`*)

F1 score for multi-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) for more details.


<h4 id="FBetaMulti" class="doc_header"><code>FBetaMulti</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L232" class="source_link" style="float:right">[source]</a></h4>

> <code>FBetaMulti</code>(**`beta`**, **`thresh`**=*`0.5`*, **`sigmoid`**=*`True`*, **`labels`**=*`None`*, **`pos_label`**=*`1`*, **`average`**=*`'macro'`*, **`sample_weight`**=*`None`*)

FBeta score with `beta` for multi-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score) for more details.


<h4 id="HammingLossMulti" class="doc_header"><code>HammingLossMulti</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L239" class="source_link" style="float:right">[source]</a></h4>

> <code>HammingLossMulti</code>(**`thresh`**=*`0.5`*, **`sigmoid`**=*`True`*, **`labels`**=*`None`*, **`sample_weight`**=*`None`*)

Hamming loss for multi-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss) for more details.


<h4 id="JaccardMulti" class="doc_header"><code>JaccardMulti</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L246" class="source_link" style="float:right">[source]</a></h4>

> <code>JaccardMulti</code>(**`thresh`**=*`0.5`*, **`sigmoid`**=*`True`*, **`labels`**=*`None`*, **`pos_label`**=*`1`*, **`average`**=*`'macro'`*, **`sample_weight`**=*`None`*)

Jaccard score for multi-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score) for more details.


<h4 id="MatthewsCorrCoefMulti" class="doc_header"><code>MatthewsCorrCoefMulti</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L253" class="source_link" style="float:right">[source]</a></h4>

> <code>MatthewsCorrCoefMulti</code>(**`thresh`**=*`0.5`*, **`sigmoid`**=*`True`*, **`sample_weight`**=*`None`*)

Matthews correlation coefficient for multi-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef) for more details.


<h4 id="PrecisionMulti" class="doc_header"><code>PrecisionMulti</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L259" class="source_link" style="float:right">[source]</a></h4>

> <code>PrecisionMulti</code>(**`thresh`**=*`0.5`*, **`sigmoid`**=*`True`*, **`labels`**=*`None`*, **`pos_label`**=*`1`*, **`average`**=*`'macro'`*, **`sample_weight`**=*`None`*)

Precision for multi-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) for more details.


<h4 id="RecallMulti" class="doc_header"><code>RecallMulti</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L266" class="source_link" style="float:right">[source]</a></h4>

> <code>RecallMulti</code>(**`thresh`**=*`0.5`*, **`sigmoid`**=*`True`*, **`labels`**=*`None`*, **`pos_label`**=*`1`*, **`average`**=*`'macro'`*, **`sample_weight`**=*`None`*)

Recall for multi-label classification problems


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) for more details.


<h4 id="RocAucMulti" class="doc_header"><code>RocAucMulti</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L273" class="source_link" style="float:right">[source]</a></h4>

> <code>RocAucMulti</code>(**`sigmoid`**=*`True`*, **`average`**=*`'macro'`*, **`sample_weight`**=*`None`*, **`max_fpr`**=*`None`*)

Area Under the Receiver Operating Characteristic Curve for multi-label binary classification problems


```python
roc_auc_metric = RocAucMulti(sigmoid=False)
x,y = torch.tensor([np.arange(start=0, stop=0.2, step=0.04)]*20), torch.tensor([0, 0, 1, 1]).repeat(5)
assert compute_val(roc_auc_metric, x, y) == 0.5
```

See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) for more details.

## Regression


<h4 id="mse" class="doc_header"><code>mse</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L280" class="source_link" style="float:right">[source]</a></h4>

> <code>mse</code>(**`inp`**, **`targ`**)

Mean squared error between `inp` and `targ`.


```python
x1,x2 = torch.randn(4,5),torch.randn(4,5)
test_close(mse(x1,x2), (x1-x2).pow(2).mean())
```


<h4 id="rmse" class="doc_header"><code>rmse</code><a href="" class="source_link" style="float:right">[source]</a></h4>

> <code>rmse</code>(**`preds`**, **`targs`**)

Root mean squared error


```python
x1,x2 = torch.randn(20,5),torch.randn(20,5)
test_eq(compute_val(rmse, x1, x2), torch.sqrt(F.mse_loss(x1,x2)))
```


<h4 id="mae" class="doc_header"><code>mae</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L290" class="source_link" style="float:right">[source]</a></h4>

> <code>mae</code>(**`inp`**, **`targ`**)

Mean absolute error between `inp` and `targ`.


```python
x1,x2 = torch.randn(4,5),torch.randn(4,5)
test_eq(mae(x1,x2), torch.abs(x1-x2).mean())
```


<h4 id="msle" class="doc_header"><code>msle</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L296" class="source_link" style="float:right">[source]</a></h4>

> <code>msle</code>(**`inp`**, **`targ`**)

Mean squared logarithmic error between `inp` and `targ`.


```python
x1,x2 = torch.randn(4,5),torch.randn(4,5)
x1,x2 = torch.relu(x1),torch.relu(x2)
test_close(msle(x1,x2), (torch.log(x1+1)-torch.log(x2+1)).pow(2).mean())
```


<h4 id="exp_rmspe" class="doc_header"><code>exp_rmspe</code><a href="" class="source_link" style="float:right">[source]</a></h4>

> <code>exp_rmspe</code>(**`preds`**, **`targs`**)

Root mean square percentage error of the exponential of  predictions and targets


```python
x1,x2 = torch.randn(20,5),torch.randn(20,5)
test_eq(compute_val(exp_rmspe, x1, x2), torch.sqrt((((torch.exp(x2) - torch.exp(x1))/torch.exp(x2))**2).mean()))
```


<h4 id="ExplainedVariance" class="doc_header"><code>ExplainedVariance</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L309" class="source_link" style="float:right">[source]</a></h4>

> <code>ExplainedVariance</code>(**`sample_weight`**=*`None`*)

Explained variance betzeen predictions and targets


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score) for more details.


<h4 id="R2Score" class="doc_header"><code>R2Score</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L314" class="source_link" style="float:right">[source]</a></h4>

> <code>R2Score</code>(**`sample_weight`**=*`None`*)

R2 score betzeen predictions and targets


See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score) for more details.


<h4 id="PearsonCorrCoef" class="doc_header"><code>PearsonCorrCoef</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L319" class="source_link" style="float:right">[source]</a></h4>

> <code>PearsonCorrCoef</code>(**`dim_argmax`**=*`None`*, **`activation`**=*`'no'`*, **`thresh`**=*`None`*, **`to_np`**=*`False`*, **`invert_arg`**=*`False`*, **`flatten`**=*`True`*)

Pearson correlation coefficient for regression problem


See the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html?highlight=pearson#scipy.stats.pearsonr) for more details.

```python
x = torch.randint(-999, 999,(20,))
y = torch.randint(-999, 999,(20,))
test_eq(compute_val(PearsonCorrCoef(), x, y), scs.pearsonr(x.view(-1), y.view(-1))[0])
```


<h4 id="SpearmanCorrCoef" class="doc_header"><code>SpearmanCorrCoef</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L326" class="source_link" style="float:right">[source]</a></h4>

> <code>SpearmanCorrCoef</code>(**`dim_argmax`**=*`None`*, **`axis`**=*`0`*, **`nan_policy`**=*`'propagate'`*, **`activation`**=*`'no'`*, **`thresh`**=*`None`*, **`to_np`**=*`False`*, **`invert_arg`**=*`False`*, **`flatten`**=*`True`*)

Spearman correlation coefficient for regression problem


See the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html?highlight=spearman#scipy.stats.spearmanr) for more details.

```python
x = torch.randint(-999, 999,(20,))
y = torch.randint(-999, 999,(20,))
test_eq(compute_val(SpearmanCorrCoef(), x, y), scs.spearmanr(x.view(-1), y.view(-1))[0])
```

## Segmentation


<h4 id="foreground_acc" class="doc_header"><code>foreground_acc</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L334" class="source_link" style="float:right">[source]</a></h4>

> <code>foreground_acc</code>(**`inp`**, **`targ`**, **`bkg_idx`**=*`0`*, **`axis`**=*`1`*)

Computes non-background accuracy for multiclass segmentation


```python
x = torch.randn(4,5,3,3)
y = x.argmax(dim=1)[:,None]
test_eq(foreground_acc(x,y), 1)
y[0] = 0 #the 0s are ignored so we get the same value
test_eq(foreground_acc(x,y), 1)
```


<h3 id="Dice" class="doc_header"><code>class</code> <code>Dice</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L341" class="source_link" style="float:right">[source]</a></h3>

> <code>Dice</code>(**`axis`**=*`1`*) :: [`Metric`](/learner.html#Metric)

Dice coefficient metric for binary target in segmentation


```python
x1 = torch.randn(20,2,3,3)
x2 = torch.randint(0, 2, (20, 3, 3))
pred = x1.argmax(1)
inter = (pred*x2).float().sum().item()
union = (pred+x2).float().sum().item()
test_eq(compute_val(Dice(), x1, x2), 2*inter/union)
```


<h3 id="JaccardCoeff" class="doc_header"><code>class</code> <code>JaccardCoeff</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L354" class="source_link" style="float:right">[source]</a></h3>

> <code>JaccardCoeff</code>(**`axis`**=*`1`*) :: [`Dice`](/metrics.html#Dice)

Implemetation of the jaccard coefficient that is lighter in RAM


```python
x1 = torch.randn(20,2,3,3)
x2 = torch.randint(0, 2, (20, 3, 3))
pred = x1.argmax(1)
inter = (pred*x2).float().sum().item()
union = (pred+x2).float().sum().item()
test_eq(compute_val(JaccardCoeff(), x1, x2), inter/(union-inter))
```

## NLP


<h3 id="CorpusBLEUMetric" class="doc_header"><code>class</code> <code>CorpusBLEUMetric</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L360" class="source_link" style="float:right">[source]</a></h3>

> <code>CorpusBLEUMetric</code>(**`vocab_sz`**=*`5000`*, **`axis`**=*`-1`*) :: [`Metric`](/learner.html#Metric)

Blueprint for defining a metric


```python
def create_vcb_emb(pred, targ):
    # create vocab "embedding" for predictions
    vcb_sz = max(torch.unique(torch.cat([pred, targ])))+1
    pred_emb=torch.zeros(pred.size()[0], pred.size()[1] ,vcb_sz)
    for i,v in enumerate(pred):
        pred_emb[i].scatter_(1, v.view(len(v),1),1)
    return pred_emb

def compute_bleu_val(met, x1, x2):
    met.reset()
    learn = TstLearner()
    learn.training=False    
    for i in range(len(x1)): 
        learn.pred,learn.y = x1, x2
        met.accumulate(learn)
    return met.value

targ = torch.tensor([[1,2,3,4,5,6,1,7,8]]) 
pred = torch.tensor([[1,9,3,4,5,6,1,10,8]])
pred_emb = create_vcb_emb(pred, targ)
test_close(compute_bleu_val(CorpusBLEUMetric(), pred_emb, targ), 0.48549)

targ = torch.tensor([[1,2,3,4,5,6,1,7,8],[1,2,3,4,5,6,1,7,8]]) 
pred = torch.tensor([[1,9,3,4,5,6,1,10,8],[1,9,3,4,5,6,1,10,8]])
pred_emb = create_vcb_emb(pred, targ)
test_close(compute_bleu_val(CorpusBLEUMetric(), pred_emb, targ), 0.48549)
```

The BLEU metric was introduced in [this article](https://www.aclweb.org/anthology/P02-1040) to come up with a way to evaluate the performance of translation models. It's based on the precision of n-grams in your prediction compared to your target. See the [fastai NLP course BLEU notebook](https://github.com/fastai/course-nlp/blob/master/bleu_metric.ipynb) for a more detailed description of BLEU.

The smoothing used in the precision calculation is the same as in [SacreBLEU](https://github.com/mjpost/sacrebleu/blob/32c54cdd0dfd6a9fadd5805f2ea189ac0df63907/sacrebleu/sacrebleu.py#L540-L542), which in turn is "method 3" from the [Chen & Cherry, 2014](http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf) paper.


<h3 id="LossMetric" class="doc_header"><code>class</code> <code>LossMetric</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L412" class="source_link" style="float:right">[source]</a></h3>

> <code>LossMetric</code>(**`attr`**, **`nm`**=*`None`*) :: [`AvgMetric`](/learner.html#AvgMetric)

Create a metric from `loss_func.attr` named `nm`



<h4 id="LossMetrics" class="doc_header"><code>LossMetrics</code><a href="https://github.com/fastai/fastai/tree/master/fastai/metrics.py#L424" class="source_link" style="float:right">[source]</a></h4>

> <code>LossMetrics</code>(**`attrs`**, **`nms`**=*`None`*)

List of [`LossMetric`](/metrics.html#LossMetric) for each of `attrs` and `nms`


```python
class CombineL1L2(Module):
    def forward(self, out, targ):
        self.l1 = F.l1_loss(out, targ)
        self.l2 = F.mse_loss(out, targ)
        return self.l1+self.l2
```

```python
learn = synth_learner(metrics=LossMetrics('l1,l2'))
learn.loss_func = CombineL1L2()
learn.fit(2)
```
