# Tabular model
> A basic model that can be used on tabular data


## Embeddings


<h4 id="emb_sz_rule" class="doc_header"><code>emb_sz_rule</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/model.py#L10" class="source_link" style="float:right">[source]</a></h4>

> <code>emb_sz_rule</code>(**`n_cat`**)

Rule of thumb to pick embedding size corresponding to `n_cat`


Through trial and error, this general rule takes the lower of two values:
* A dimension space of 600
* A dimension space equal to 1.6 times the cardinality of the variable to 0.56.

This provides a good starter for a good embedding space for your variables. For more advanced users who wish to lean into this practice, you can tweak these values to your discretion. It is not uncommon for slight adjustments to this general formula to provide more success.


<h4 id="get_emb_sz" class="doc_header"><code>get_emb_sz</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/model.py#L23" class="source_link" style="float:right">[source]</a></h4>

> <code>get_emb_sz</code>(**`to`**, **`sz_dict`**=*`None`*)

Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`



<h2 id="TabularModel" class="doc_header"><code>class</code> <code>TabularModel</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/model.py#L28" class="source_link" style="float:right">[source]</a></h2>

> <code>TabularModel</code>(**`emb_szs`**, **`n_cont`**, **`out_sz`**, **`layers`**, **`ps`**=*`None`*, **`embed_p`**=*`0.0`*, **`y_range`**=*`None`*, **`use_bn`**=*`True`*, **`bn_final`**=*`False`*, **`bn_cont`**=*`True`*) :: [`Module`](/torch_core.html#Module)

Basic model for tabular data.


This model expects your `cat` and `cont` variables seperated. `cat` is passed through an [`Embedding`](/layers.html#Embedding) layer and potential `Dropout`, while `cont` is passed though potential `BatchNorm1d`. Afterwards both are concatenated and passed through a series of [`LinBnDrop`](/layers.html#LinBnDrop), before a final `Linear` layer corresponding to the expected outputs. 

```python
emb_szs = [(4,2), (17,8)]
m = TabularModel(emb_szs, n_cont=2, out_sz=2, layers=[200,100]).eval()
x_cat = torch.tensor([[2,12]]).long()
x_cont = torch.tensor([[0.7633, -0.1887]]).float()
out = m(x_cat, x_cont)
```


<h4 id="tabular_config" class="doc_header"><code>tabular_config</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/model.py#L57" class="source_link" style="float:right">[source]</a></h4>

> <code>tabular_config</code>(**`ps`**=*`None`*, **`embed_p`**=*`0.0`*, **`y_range`**=*`None`*, **`use_bn`**=*`True`*, **`bn_final`**=*`False`*, **`bn_cont`**=*`True`*)

Convenience function to easily create a config for `tabular_model`


Any direct setup of [`TabularModel`](/tabular.model.html#TabularModel)'s internals should be passed through here:

```python
config = tabular_config(embed_p=0.6, use_bn=False); config
```




    {'embed_p': 0.6, 'use_bn': False}


