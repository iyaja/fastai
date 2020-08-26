# Collaborative filtering
> Tools to quickly get the data and train models suitable for collaborative filtering


This module contains all the high-level functions you need in a collaborative filtering application to assemble your data, get a model and train it with a [`Learner`](/learner.html#Learner). We will go other those in order but you can also check the [collaborative filtering tutorial](http://docs.fast.ai/tutorial.collab).

## Gather the data


<h2 id="TabularCollab" class="doc_header"><code>class</code> <code>TabularCollab</code><a href="https://github.com/fastai/fastai/tree/master/fastai/collab.py#L9" class="source_link" style="float:right">[source]</a></h2>

> <code>TabularCollab</code>(**`df`**, **`procs`**=*`None`*, **`cat_names`**=*`None`*, **`cont_names`**=*`None`*, **`y_names`**=*`None`*, **`y_block`**=*`None`*, **`splits`**=*`None`*, **`do_setup`**=*`True`*, **`device`**=*`None`*, **`inplace`**=*`False`*, **`reduce_memory`**=*`True`*) :: [`TabularPandas`](/tabular.core.html#TabularPandas)

Instance of [`TabularPandas`](/tabular.core.html#TabularPandas) suitable for collaborative filtering (with no continuous variable)


This is just to use the internal of the tabular application, don't worry about it.


<h2 id="CollabDataLoaders" class="doc_header"><code>class</code> <code>CollabDataLoaders</code><a href="https://github.com/fastai/fastai/tree/master/fastai/collab.py#L14" class="source_link" style="float:right">[source]</a></h2>

> <code>CollabDataLoaders</code>(**\*`loaders`**, **`path`**=*`'.'`*, **`device`**=*`None`*) :: [`DataLoaders`](/data.core.html#DataLoaders)

Base [`DataLoaders`](/data.core.html#DataLoaders) for collaborative filtering.


This class should not be used directly, one of the factory methods should be preferred instead. All those factory methods accept as arguments:

- `valid_pct`: the random percentage of the dataset to set aside for validation (with an optional `seed`)
- `user_name`: the name of the column containing the user (defaults to the first column)
- `item_name`: the name of the column containing the item (defaults to the second column)
- `rating_name`: the name of the column containing the rating (defaults to the third column)
- `path`: the folder where to work
- `bs`: the batch size
- `val_bs`: the batch size for the validation [`DataLoader`](/data.load.html#DataLoader) (defaults to `bs`)
- `shuffle_train`: if we shuffle the training [`DataLoader`](/data.load.html#DataLoader) or not
- `device`: the PyTorch device to use (defaults to `default_device()`)


<h4 id="CollabDataLoaders.from_df" class="doc_header"><code>CollabDataLoaders.from_df</code><a href="https://github.com/fastai/fastai/tree/master/fastai/collab.py#L16" class="source_link" style="float:right">[source]</a></h4>

> <code>CollabDataLoaders.from_df</code>(**`ratings`**, **`valid_pct`**=*`0.2`*, **`user_name`**=*`None`*, **`item_name`**=*`None`*, **`rating_name`**=*`None`*, **`seed`**=*`None`*, **`path`**=*`'.'`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create a [`DataLoaders`](/data.core.html#DataLoaders) suitable for collaborative filtering from `ratings`.


Let's see how this works on an example:

```python
path = untar_data(URLs.ML_SAMPLE)
ratings = pd.read_csv(path/'ratings.csv')
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>73</td>
      <td>1097</td>
      <td>4.0</td>
      <td>1255504951</td>
    </tr>
    <tr>
      <th>1</th>
      <td>561</td>
      <td>924</td>
      <td>3.5</td>
      <td>1172695223</td>
    </tr>
    <tr>
      <th>2</th>
      <td>157</td>
      <td>260</td>
      <td>3.5</td>
      <td>1291598691</td>
    </tr>
    <tr>
      <th>3</th>
      <td>358</td>
      <td>1210</td>
      <td>5.0</td>
      <td>957481884</td>
    </tr>
    <tr>
      <th>4</th>
      <td>130</td>
      <td>316</td>
      <td>2.0</td>
      <td>1138999234</td>
    </tr>
  </tbody>
</table>
</div>



```python
dls = CollabDataLoaders.from_df(ratings, bs=64)
dls.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>157</td>
      <td>1265</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>130</td>
      <td>2858</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>481</td>
      <td>1196</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>105</td>
      <td>597</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>128</td>
      <td>597</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>587</td>
      <td>5952</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>111</td>
      <td>2571</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>105</td>
      <td>356</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>77</td>
      <td>5349</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>119</td>
      <td>1240</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>



<h4 id="CollabDataLoaders.from_csv" class="doc_header"><code>CollabDataLoaders.from_csv</code><a href="https://github.com/fastai/fastai/tree/master/fastai/collab.py#L28" class="source_link" style="float:right">[source]</a></h4>

> <code>CollabDataLoaders.from_csv</code>(**`csv`**, **`valid_pct`**=*`0.2`*, **`user_name`**=*`None`*, **`item_name`**=*`None`*, **`rating_name`**=*`None`*, **`seed`**=*`None`*, **`path`**=*`'.'`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create a [`DataLoaders`](/data.core.html#DataLoaders) suitable for collaborative filtering from `csv`.


```python
dls = CollabDataLoaders.from_csv(path/'ratings.csv', bs=64)
```

## Models

fastai provides two kinds of models for collaborative filtering: a dot-product model and a neural net. 


<h2 id="EmbeddingDotBias" class="doc_header"><code>class</code> <code>EmbeddingDotBias</code><a href="https://github.com/fastai/fastai/tree/master/fastai/collab.py#L36" class="source_link" style="float:right">[source]</a></h2>

> <code>EmbeddingDotBias</code>(**`n_factors`**, **`n_users`**, **`n_items`**, **`y_range`**=*`None`*) :: [`Module`](/torch_core.html#Module)

Base dot model for collaborative filtering.


The model is built with `n_factors` (the length of the internal vectors), `n_users` and `n_items`. For a given user and item, it grabs the corresponding weights and bias and returns
``` python
torch.dot(user_w, item_w) + user_b + item_b
```
Optionally, if `y_range` is passed, it applies a [`SigmoidRange`](/layers.html#SigmoidRange) to that result.

```python
x,y = dls.one_batch()
model = EmbeddingDotBias(50, len(dls.classes['userId']), len(dls.classes['movieId']), y_range=(0,5)
                        ).to(x.device)
out = model(x)
assert (0 <= out).all() and (out <= 5).all()
```


<h4 id="EmbeddingDotBias.from_classes" class="doc_header"><code>EmbeddingDotBias.from_classes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/collab.py#L51" class="source_link" style="float:right">[source]</a></h4>

> <code>EmbeddingDotBias.from_classes</code>(**`n_factors`**, **`classes`**, **`user`**=*`None`*, **`item`**=*`None`*, **`y_range`**=*`None`*)

Build a model with `n_factors` by inferring `n_users` and  `n_items` from `classes`


`y_range` is passed to the main init. `user` and `item` are the names of the keys for users and items in `classes` (default to the first and second key respectively). `classes` is expected to be a dictionary key to list of categories like the result of `dls.classes` in a [`CollabDataLoaders`](/collab.html#CollabDataLoaders):

```python
dls.classes
```




    {'userId': (#101) ['#na#',15,17,19,23,30,48,56,73,77...],
     'movieId': (#101) ['#na#',1,10,32,34,39,47,50,110,150...]}



Let's see how it can be used in practice:

```python
model = EmbeddingDotBias.from_classes(50, dls.classes,  y_range=(0,5)
                                     ).to(x.device)
out = model(x)
assert (0 <= out).all() and (out <= 5).all()
```

Two convenience methods are added to easily access the weights and bias when a model is created with [`EmbeddingDotBias.from_classes`](/collab.html#EmbeddingDotBias.from_classes):


<h4 id="EmbeddingDotBias.weight" class="doc_header"><code>EmbeddingDotBias.weight</code><a href="https://github.com/fastai/fastai/tree/master/fastai/collab.py#L76" class="source_link" style="float:right">[source]</a></h4>

> <code>EmbeddingDotBias.weight</code>(**`arr`**, **`is_item`**=*`True`*)

Weight for item or user (based on `is_item`) for all in `arr`


The elements of `arr` are expected to be class names (which is why the model needs to be created with [`EmbeddingDotBias.from_classes`](/collab.html#EmbeddingDotBias.from_classes))

```python
mov = dls.classes['movieId'][42] 
w = model.weight([mov])
test_eq(w, model.i_weight(tensor([42])))
```


<h4 id="EmbeddingDotBias.bias" class="doc_header"><code>EmbeddingDotBias.bias</code><a href="https://github.com/fastai/fastai/tree/master/fastai/collab.py#L70" class="source_link" style="float:right">[source]</a></h4>

> <code>EmbeddingDotBias.bias</code>(**`arr`**, **`is_item`**=*`True`*)

Bias for item or user (based on `is_item`) for all in `arr`


The elements of `arr` are expected to be class names (which is why the model needs to be created with [`EmbeddingDotBias.from_classes`](/collab.html#EmbeddingDotBias.from_classes))

```python
mov = dls.classes['movieId'][42] 
b = model.bias([mov])
test_eq(b, model.i_bias(tensor([42])))
```


<h2 id="EmbeddingNN" class="doc_header"><code>class</code> <code>EmbeddingNN</code><a href="https://github.com/fastai/fastai/tree/master/fastai/collab.py#L83" class="source_link" style="float:right">[source]</a></h2>

> <code>EmbeddingNN</code>(**`emb_szs`**, **`layers`**, **`ps`**=*`None`*, **`embed_p`**=*`0.0`*, **`y_range`**=*`None`*, **`use_bn`**=*`True`*, **`bn_final`**=*`False`*, **`bn_cont`**=*`True`*) :: [`TabularModel`](/tabular.model.html#TabularModel)

Subclass [`TabularModel`](/tabular.model.html#TabularModel) to create a NN suitable for collaborative filtering.


`emb_szs` should be a list of two tuples, one for the users, one for the items, each tuple containing the number of users/items and the corresponding embedding size (the function [`get_emb_sz`](/tabular.model.html#get_emb_sz) can give a good default). All the other arguments are passed to [`TabularModel`](/tabular.model.html#TabularModel).

```python
emb_szs = get_emb_sz(dls.train_ds, {})
model = EmbeddingNN(emb_szs, [50], y_range=(0,5)
                   ).to(x.device)
out = model(x)
assert (0 <= out).all() and (out <= 5).all()
```

## Create a [`Learner`](/learner.html#Learner)

The following function lets us quickly create a [`Learner`](/learner.html#Learner) for collaborative filtering from the data.


<h4 id="collab_learner" class="doc_header"><code>collab_learner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/collab.py#L90" class="source_link" style="float:right">[source]</a></h4>

> <code>collab_learner</code>(**`dls`**, **`n_factors`**=*`50`*, **`use_nn`**=*`False`*, **`emb_szs`**=*`None`*, **`layers`**=*`None`*, **`config`**=*`None`*, **`y_range`**=*`None`*, **`loss_func`**=*`None`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`splitter`**=*`trainable_params`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*, **`moms`**=*`(0.95, 0.85, 0.95)`*)

Create a Learner for collaborative filtering on `dls`.


If `use_nn=False`, the model used is an [`EmbeddingDotBias`](/collab.html#EmbeddingDotBias) with `n_factors` and `y_range`. Otherwise, it's a [`EmbeddingNN`](/collab.html#EmbeddingNN) for which you can pass `emb_szs` (will be inferred from the `dls` with [`get_emb_sz`](/tabular.model.html#get_emb_sz) if you don't provide any), [`layers`](/layers.html) (defaults to `[n_factors]`) `y_range`, and a `config` that you can create with [`tabular_config`](/tabular.model.html#tabular_config) to customize your model. 

`loss_func` will default to [`MSELossFlat`](/layers.html#MSELossFlat) and all the other arguments are passed to [`Learner`](/learner.html#Learner).

```python
learn = collab_learner(dls, y_range=(0,5))
```

```python
learn.fit_one_cycle(1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.521979</td>
      <td>2.541627</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>

