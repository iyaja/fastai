# Tabular data
> Helper functions to get data in a `DataLoaders` in the tabular application and higher class `TabularDataLoaders`


The main class to get your data ready for model training is [`TabularDataLoaders`](/tabular.data.html#TabularDataLoaders) and its factory methods. Checkout the [tabular tutorial](http://docs.fast.ai/tutorial.tabular) for examples of use.


<h2 id="TabularDataLoaders" class="doc_header"><code>class</code> <code>TabularDataLoaders</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/data.py#L11" class="source_link" style="float:right">[source]</a></h2>

> <code>TabularDataLoaders</code>(**\*`loaders`**, **`path`**=*`'.'`*, **`device`**=*`None`*) :: [`DataLoaders`](/data.core.html#DataLoaders)

Basic wrapper around several [`DataLoader`](/data.load.html#DataLoader)s with factory methods for tabular data


This class should not be used directly, one of the factory methods should be preferred instead. All those factory methods accept as arguments:

- `cat_names`: the names of the categorical variables
- `cont_names`: the names of the continuous variables
- `y_names`: the names of the dependent variables
- `y_block`: the [`TransformBlock`](/data.block.html#TransformBlock) to use for the target
- `valid_idx`: the indices to use for the validation set (defaults to a random split otherwise)
- `bs`: the batch size
- `val_bs`: the batch size for the validation [`DataLoader`](/data.load.html#DataLoader) (defaults to `bs`)
- `shuffle_train`: if we shuffle the training [`DataLoader`](/data.load.html#DataLoader) or not
- `n`: overrides the numbers of elements in the dataset
- `device`: the PyTorch device to use (defaults to `default_device()`)


<h4 id="TabularDataLoaders.from_df" class="doc_header"><code>TabularDataLoaders.from_df</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/data.py#L13" class="source_link" style="float:right">[source]</a></h4>

> <code>TabularDataLoaders.from_df</code>(**`df`**, **`path`**=*`'.'`*, **`procs`**=*`None`*, **`cat_names`**=*`None`*, **`cont_names`**=*`None`*, **`y_names`**=*`None`*, **`y_block`**=*`None`*, **`valid_idx`**=*`None`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`n`**=*`None`*, **`device`**=*`None`*)

Create from `df` in `path` using `procs`


Let's have a look on an example with the adult dataset:

```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>Private</td>
      <td>101320</td>
      <td>Assoc-acdm</td>
      <td>12.0</td>
      <td>Married-civ-spouse</td>
      <td>NaN</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>1902</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>Private</td>
      <td>236746</td>
      <td>Masters</td>
      <td>14.0</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>10520</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>96185</td>
      <td>HS-grad</td>
      <td>NaN</td>
      <td>Divorced</td>
      <td>NaN</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>Self-emp-inc</td>
      <td>112847</td>
      <td>Prof-school</td>
      <td>15.0</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>Self-emp-not-inc</td>
      <td>82297</td>
      <td>7th-8th</td>
      <td>NaN</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>
</div>



```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
```

```python
dls = TabularDataLoaders.from_df(df, path, procs=procs, cat_names=cat_names, cont_names=cont_names, 
                                 y_names="salary", valid_idx=list(range(800,1000)), bs=64)
```

```python
dls.show_batch()
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
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>Sales</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>40.0</td>
      <td>116632.001407</td>
      <td>9.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>State-gov</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Protective-serv</td>
      <td>Own-child</td>
      <td>Black</td>
      <td>False</td>
      <td>22.0</td>
      <td>293363.998886</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>Craft-repair</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>35.0</td>
      <td>126568.998886</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Private</td>
      <td>Masters</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>False</td>
      <td>39.0</td>
      <td>150061.001071</td>
      <td>14.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>21.0</td>
      <td>283756.998474</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Private</td>
      <td>Masters</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>29.0</td>
      <td>134565.997603</td>
      <td>14.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Self-emp-not-inc</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>39.0</td>
      <td>148442.999504</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Adm-clerical</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>49.0</td>
      <td>280524.999991</td>
      <td>10.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Local-gov</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>39.0</td>
      <td>166497.000063</td>
      <td>9.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>9</th>
      <td>?</td>
      <td>11th</td>
      <td>Never-married</td>
      <td>?</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>17.0</td>
      <td>47407.001911</td>
      <td>7.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>



<h4 id="TabularDataLoaders.from_csv" class="doc_header"><code>TabularDataLoaders.from_csv</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/data.py#L24" class="source_link" style="float:right">[source]</a></h4>

> <code>TabularDataLoaders.from_csv</code>(**`csv`**, **`path`**=*`'.'`*, **`procs`**=*`None`*, **`cat_names`**=*`None`*, **`cont_names`**=*`None`*, **`y_names`**=*`None`*, **`y_block`**=*`None`*, **`valid_idx`**=*`None`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`n`**=*`None`*, **`device`**=*`None`*)

Create from `csv` file in `path` using `procs`


```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, procs=procs, cat_names=cat_names, cont_names=cont_names, 
                                  y_names="salary", valid_idx=list(range(800,1000)), bs=64)
```
