# Interpretation
> Classes to build objects to better interpret predictions of a model



<h2 id="Interpretation" class="doc_header"><code>class</code> <code>Interpretation</code><a href="https://github.com/fastai/fastai/tree/master/fastai/interpret.py#L20" class="source_link" style="float:right">[source]</a></h2>

> <code>Interpretation</code>(**`dl`**, **`inputs`**, **`preds`**, **`targs`**, **`decoded`**, **`losses`**)

Interpretation base class, can be inherited for task specific Interpretation classes


```python
learn = synth_learner()
interp = Interpretation.from_learner(learn)
x,y = learn.dls.valid_ds.tensors
test_eq(interp.inputs, x)
test_eq(interp.targs, y)
out = learn.model.a * x + learn.model.b
test_eq(interp.preds, out)
test_eq(interp.losses, (out-y)[:,0]**2)
```


<h2 id="ClassificationInterpretation" class="doc_header"><code>class</code> <code>ClassificationInterpretation</code><a href="https://github.com/fastai/fastai/tree/master/fastai/interpret.py#L51" class="source_link" style="float:right">[source]</a></h2>

> <code>ClassificationInterpretation</code>(**`dl`**, **`inputs`**, **`preds`**, **`targs`**, **`decoded`**, **`losses`**) :: [`Interpretation`](/interpret.html#Interpretation)

Interpretation methods for classification models.

