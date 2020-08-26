# Tabular learner
> The function to immediately get a `Learner` ready to train for tabular data


```python
from fastai.tabular.data import *
```

The main function you probably want to use in this module is [`tabular_learner`](/tabular.learner.html#tabular_learner). It will automatically create a [`TabularModel`](/tabular.model.html#TabularModel) suitable for your data and infer the right loss function. See the [tabular tutorial](http://docs.fast.ai/tutorial.tabular) for an example of use in context.

## Main functions


<h3 id="TabularLearner" class="doc_header"><code>class</code> <code>TabularLearner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/learner.py#L12" class="source_link" style="float:right">[source]</a></h3>

> <code>TabularLearner</code>(**`dls`**, **`model`**, **`loss_func`**=*`None`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`splitter`**=*`trainable_params`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*, **`moms`**=*`(0.95, 0.85, 0.95)`*) :: [`Learner`](/learner.html#Learner)

[`Learner`](/learner.html#Learner) for tabular data


It works exactly as a normal [`Learner`](/learner.html#Learner), the only difference is that it implements a `predict` method specific to work on a row of data.


<h4 id="tabular_learner" class="doc_header"><code>tabular_learner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/learner.py#L26" class="source_link" style="float:right">[source]</a></h4>

> <code>tabular_learner</code>(**`dls`**, **`layers`**=*`None`*, **`emb_szs`**=*`None`*, **`config`**=*`None`*, **`n_out`**=*`None`*, **`y_range`**=*`None`*, **`loss_func`**=*`None`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`splitter`**=*`trainable_params`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*, **`moms`**=*`(0.95, 0.85, 0.95)`*)

Get a [`Learner`](/learner.html#Learner) using `dls`, with `metrics`, including a [`TabularModel`](/tabular.model.html#TabularModel) created using the remaining params.


If your data was built with fastai, you probably won't need to pass anything to `emb_szs` unless you want to change the default of the library (produced by [`get_emb_sz`](/tabular.model.html#get_emb_sz)), same for `n_out` which should be automatically inferred. [`layers`](/layers.html) will default to `[200,100]` and is passed to [`TabularModel`](/tabular.model.html#TabularModel) along with the `config`.

Use [`tabular_config`](/tabular.model.html#tabular_config) to create a `config` and customize the model used. There is just easy access to `y_range` because this argument is often used.

All the other arguments are passed to [`Learner`](/learner.html#Learner).

```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
dls = TabularDataLoaders.from_df(df, path, procs=procs, cat_names=cat_names, cont_names=cont_names, 
                                 y_names="salary", valid_idx=list(range(800,1000)), bs=64)
learn = tabular_learner(dls)
```


<h4 id="TabularLearner.predict" class="doc_header"><code>TabularLearner.predict</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/learner.py#L14" class="source_link" style="float:right">[source]</a></h4>

> <code>TabularLearner.predict</code>(**`row`**)

Prediction on `item`, fully decoded, loss function decoded and probabilities


We can pass in an individual row of data into our [`TabularLearner`](/tabular.learner.html#TabularLearner)'s `predict` method. It's output is slightly different from the other `predict` methods, as this one will always return the input as well:

```python
row, clas, probs = learn.predict(df.iloc[0])
```

```python
row.show()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Private</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>#na#</td>
      <td>Wife</td>
      <td>White</td>
      <td>False</td>
      <td>49.0</td>
      <td>101320.001685</td>
      <td>12.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>


```python
clas, probs
```




    (tensor(0), tensor([0.5264, 0.4736]))


