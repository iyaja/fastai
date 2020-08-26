# Tabular core
> Basic function to preprocess tabular data before assembling it in a `DataLoaders`.


## Initial preprocessing


<h4 id="make_date" class="doc_header"><code>make_date</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L16" class="source_link" style="float:right">[source]</a></h4>

> <code>make_date</code>(**`df`**, **`date_field`**)

Make sure `df[date_field]` is of the right date type.


```python
df = pd.DataFrame({'date': ['2019-12-04', '2019-11-29', '2019-11-15', '2019-10-24']})
make_date(df, 'date')
test_eq(df['date'].dtype, np.dtype('datetime64[ns]'))
```


<h4 id="add_datepart" class="doc_header"><code>add_datepart</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L25" class="source_link" style="float:right">[source]</a></h4>

> <code>add_datepart</code>(**`df`**, **`field_name`**, **`prefix`**=*`None`*, **`drop`**=*`True`*, **`time`**=*`False`*)

Helper function that adds columns relevant to a date in the column `field_name` of `df`.


```python
df = pd.DataFrame({'date': ['2019-12-04', None, '2019-11-15', '2019-10-24']})
df = add_datepart(df, 'date')
test_eq(df.columns, ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Elapsed'])
test_eq(df[df.Elapsed.isna()].shape,(1, 13))
df.head()
```

    <ipython-input-7-02bfdad141e1>:10: FutureWarning: Series.dt.weekofyear and Series.dt.week have been deprecated.  Please use Series.dt.isocalendar().week instead.
      for n in attr: df[prefix + n] = getattr(field.dt, n.lower())





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
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019.0</td>
      <td>12.0</td>
      <td>49.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>338.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1575417600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019.0</td>
      <td>11.0</td>
      <td>46.0</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>319.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1573776000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019.0</td>
      <td>10.0</td>
      <td>43.0</td>
      <td>24.0</td>
      <td>3.0</td>
      <td>297.0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1571875200</td>
    </tr>
  </tbody>
</table>
</div>




<h4 id="add_elapsed_times" class="doc_header"><code>add_elapsed_times</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L53" class="source_link" style="float:right">[source]</a></h4>

> <code>add_elapsed_times</code>(**`df`**, **`field_names`**, **`date_field`**, **`base_field`**)

Add in `df` for each event in `field_names` the elapsed time according to `date_field` grouped by `base_field`


```python
df = pd.DataFrame({'date': ['2019-12-04', '2019-11-29', '2019-11-15', '2019-10-24'],
                   'event': [False, True, False, True], 'base': [1,1,2,2]})
df = add_elapsed_times(df, ['event'], 'date', 'base')
df
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
      <th>date</th>
      <th>event</th>
      <th>base</th>
      <th>Afterevent</th>
      <th>Beforeevent</th>
      <th>event_bw</th>
      <th>event_fw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-04</td>
      <td>False</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-11-29</td>
      <td>True</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-11-15</td>
      <td>False</td>
      <td>2</td>
      <td>22</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-10-24</td>
      <td>True</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




<h4 id="cont_cat_split" class="doc_header"><code>cont_cat_split</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L81" class="source_link" style="float:right">[source]</a></h4>

> <code>cont_cat_split</code>(**`df`**, **`max_card`**=*`20`*, **`dep_var`**=*`None`*)

Helper function that returns column names of cont and cat variables from given `df`.



<h3 id="df_shrink_dtypes" class="doc_header"><code>df_shrink_dtypes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L92" class="source_link" style="float:right">[source]</a></h3>

> <code>df_shrink_dtypes</code>(**`df`**, **`skip`**=*`[]`*, **`obj2cat`**=*`True`*, **`int2uint`**=*`False`*)

Return any possible smaller data types for DataFrame columns. Allows `object`->`category`, `int`->`uint`, and exclusion.


```python
df = pd.DataFrame({'i': [-100, 0, 100], 'f': [-100.0, 0.0, 100.0], 'e': [True, False, True],
                   'date':['2019-12-04','2019-11-29','2019-11-15',]})
dt = df_shrink_dtypes(df)
test_eq(df['i'].dtype, 'int64')
test_eq(dt['i'], 'int8')

test_eq(df['f'].dtype, 'float64')
test_eq(dt['f'], 'float32')

# Default ignore 'object' and 'boolean' columns
test_eq(df['date'].dtype, 'object')
test_eq(dt['date'], 'category')

# Test categorifying 'object' type
dt2 = df_shrink_dtypes(df, obj2cat=False)
test_eq('date' not in dt2, True)
```


<h3 id="df_shrink" class="doc_header"><code>df_shrink</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L121" class="source_link" style="float:right">[source]</a></h3>

> <code>df_shrink</code>(**`df`**, **`skip`**=*`[]`*, **`obj2cat`**=*`True`*, **`int2uint`**=*`False`*)

Reduce DataFrame memory usage, by casting to smaller types returned by `df_shrink_dtypes()`.


`df_shrink(df)` attempts to make a DataFrame uses less memory, by fit numeric columns into smallest datatypes.  In addition:

 * `boolean`, `category`, `datetime64[ns]` dtype columns are ignored.
 * 'object' type columns are categorified, which can save a lot of memory in large dataset.  It can be turned off by `obj2cat=False`.
 * `int2uint=True`, to fit `int` types to `uint` types, if all data in the column is >= 0.
 * columns can be excluded by name using `excl_cols=['col1','col2']`.

To get only new column data types without actually casting a DataFrame,
use `df_shrink_dtypes()` with all the same parameters for `df_shrink()`.

```python
df = pd.DataFrame({'i': [-100, 0, 100], 'f': [-100.0, 0.0, 100.0], 'u':[0, 10,254],
                  'date':['2019-12-04','2019-11-29','2019-11-15']})
df2 = df_shrink(df, skip=['date'])

test_eq(df['i'].dtype=='int64' and df2['i'].dtype=='int8', True)
test_eq(df['f'].dtype=='float64' and df2['f'].dtype=='float32', True)
test_eq(df['u'].dtype=='int64' and df2['u'].dtype=='int16', True)
test_eq(df2['date'].dtype, 'object')

test_eq(df2.memory_usage().sum() < df.memory_usage().sum(), True)

# Test int => uint (when col.min() >= 0)
df3 = df_shrink(df, int2uint=True)
test_eq(df3['u'].dtype, 'uint8')  # int64 -> uint8 instead of int16

# Test excluding columns
df4 = df_shrink(df, skip=['i','u'])
test_eq(df['i'].dtype, df4['i'].dtype)
test_eq(df4['u'].dtype, 'int64')
```

Here's an example using the `ADULT_SAMPLE` dataset:

```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
new_df = df_shrink(df, int2uint=True)
print(f"Memory usage: {df.memory_usage().sum()} --> {new_df.memory_usage().sum()}")
```

    Memory usage: 3907448 --> 818665



<h2 id="Tabular" class="doc_header"><code>class</code> <code>Tabular</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L139" class="source_link" style="float:right">[source]</a></h2>

> <code>Tabular</code>(**`df`**, **`procs`**=*`None`*, **`cat_names`**=*`None`*, **`cont_names`**=*`None`*, **`y_names`**=*`None`*, **`y_block`**=*`None`*, **`splits`**=*`None`*, **`do_setup`**=*`True`*, **`device`**=*`None`*, **`inplace`**=*`False`*, **`reduce_memory`**=*`True`*) :: [`CollBase`](https://fastcore.fast.ai/foundation#CollBase)

A `DataFrame` wrapper that knows which cols are cont/cat/y, and returns rows in `__getitem__`


* `df`: A `DataFrame` of your data
* `cat_names`: Your categorical `x` variables
* `cont_names`: Your continuous `x` variables
* `y_names`: Your dependent `y` variables
  * Note: Mixed y's such as Regression and Classification is not currently supported, however multiple regression or classification outputs is
* `y_block`: How to sub-categorize the type of `y_names` ([`CategoryBlock`](/data.block.html#CategoryBlock) or [`RegressionBlock`](/data.block.html#RegressionBlock))
* `splits`: How to split your data
* `do_setup`: A parameter for if [`Tabular`](/tabular.core.html#Tabular) will run the data through the `procs` upon initialization
* `device`: `cuda` or `cpu`
* `inplace`: If `True`, [`Tabular`](/tabular.core.html#Tabular) will not keep a separate copy of your original `DataFrame` in memory. You should ensure `pd.options.mode.chained_assignment` is `None` before setting this
* `reduce_memory`: `fastai` will attempt to reduce the overall memory usage by the inputted `DataFrame` with [`df_shrink`](/tabular.core.html#df_shrink)


<h2 id="TabularPandas" class="doc_header"><code>class</code> <code>TabularPandas</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L194" class="source_link" style="float:right">[source]</a></h2>

> <code>TabularPandas</code>(**`df`**, **`procs`**=*`None`*, **`cat_names`**=*`None`*, **`cont_names`**=*`None`*, **`y_names`**=*`None`*, **`y_block`**=*`None`*, **`splits`**=*`None`*, **`do_setup`**=*`True`*, **`device`**=*`None`*, **`inplace`**=*`False`*, **`reduce_memory`**=*`True`*) :: [`Tabular`](/tabular.core.html#Tabular)

A [`Tabular`](/tabular.core.html#Tabular) object with transforms


```python
df = pd.DataFrame({'a':[0,1,2,0,2], 'b':[0,0,0,0,1]})
to = TabularPandas(df, cat_names='a')
t = pickle.loads(pickle.dumps(to))
test_eq(t.items,to.items)
test_eq(to.all_cols,to[['a']])
```


<h2 id="TabularProc" class="doc_header"><code>class</code> <code>TabularProc</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L216" class="source_link" style="float:right">[source]</a></h2>

> <code>TabularProc</code>(**`enc`**=*`None`*, **`dec`**=*`None`*, **`split_idx`**=*`None`*, **`order`**=*`None`*) :: [`InplaceTransform`](https://fastcore.fast.ai/transform#InplaceTransform)

Base class to write a non-lazy tabular processor for dataframes



<h4 id="setups" class="doc_header"><code>setups</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L373" class="source_link" style="float:right">[source]</a></h4>

> <code>setups</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))





<h4 id="encodes" class="doc_header"><code>encodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L379" class="source_link" style="float:right">[source]</a></h4>

> <code>encodes</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))





<h4 id="decodes" class="doc_header"><code>decodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L382" class="source_link" style="float:right">[source]</a></h4>

> <code>decodes</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))





<h3 id="Categorify" class="doc_header"><code>class</code> <code>Categorify</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L238" class="source_link" style="float:right">[source]</a></h3>

> <code>Categorify</code>(**`enc`**=*`None`*, **`dec`**=*`None`*, **`split_idx`**=*`None`*, **`order`**=*`None`*) :: [`TabularProc`](/tabular.core.html#TabularProc)

Transform the categorical variables to something similar to `pd.Categorical`


```python
df = pd.DataFrame({'a':[0,1,2,0,2]})
to = TabularPandas(df, Categorify, 'a')
cat = to.procs.categorify
test_eq(cat['a'], ['#na#',0,1,2])
test_eq(to['a'], [1,2,3,1,3])
to.show()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
    </tr>
  </tbody>
</table>


```python
df1 = pd.DataFrame({'a':[1,0,3,-1,2]})
to1 = to.new(df1)
to1.process()
#Values that weren't in the training df are sent to 0 (na)
test_eq(to1['a'], [2,1,0,0,3])
to2 = cat.decode(to1)
test_eq(to2['a'], [1,0,'#na#','#na#',2])
```

```python
cat = Categorify()
df = pd.DataFrame({'a':[0,1,2,3,2]})
to = TabularPandas(df, cat, 'a', splits=[[0,1,2],[3,4]])
test_eq(cat['a'], ['#na#',0,1,2])
test_eq(to['a'], [1,2,3,0,3])
```

```python
df = pd.DataFrame({'a':pd.Categorical(['M','H','L','M'], categories=['H','M','L'], ordered=True)})
to = TabularPandas(df, Categorify, 'a')
cat = to.procs.categorify
test_eq(cat['a'], ['#na#','H','M','L'])
test_eq(to.items.a, [2,1,3,2])
to2 = cat.decode(to)
test_eq(to2['a'], ['M','H','L','M'])
```

```python
cat = Categorify()
df = pd.DataFrame({'a':[0,1,2,3,2], 'b': ['a', 'b', 'a', 'b', 'b']})
to = TabularPandas(df, cat, 'a', splits=[[0,1,2],[3,4]], y_names='b')
test_eq(to.vocab, ['a', 'b'])
test_eq(to['b'], [0,1,0,1,1])
to2 = to.procs.decode(to)
test_eq(to2['b'], ['a', 'b', 'a', 'b', 'b'])
```

```python
cat = Categorify()
df = pd.DataFrame({'a':[0,1,2,3,2], 'b': ['a', 'b', 'a', 'b', 'b']})
to = TabularPandas(df, cat, 'a', splits=[[0,1,2],[3,4]], y_names='b')
test_eq(to.vocab, ['a', 'b'])
test_eq(to['b'], [0,1,0,1,1])
to2 = to.procs.decode(to)
test_eq(to2['b'], ['a', 'b', 'a', 'b', 'b'])
```

```python
cat = Categorify()
df = pd.DataFrame({'a':[0,1,2,3,2], 'b': ['a', 'b', 'a', 'c', 'b']})
to = TabularPandas(df, cat, 'a', splits=[[0,1,2],[3,4]], y_names='b')
test_eq(to.vocab, ['a', 'b'])
```


<h4 id="setups" class="doc_header"><code>setups</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L373" class="source_link" style="float:right">[source]</a></h4>

> <code>setups</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))





<h4 id="encodes" class="doc_header"><code>encodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L379" class="source_link" style="float:right">[source]</a></h4>

> <code>encodes</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))





<h4 id="decodes" class="doc_header"><code>decodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L382" class="source_link" style="float:right">[source]</a></h4>

> <code>decodes</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))




```python
norm = Normalize()
df = pd.DataFrame({'a':[0,1,2,3,4]})
to = TabularPandas(df, norm, cont_names='a')
x = np.array([0,1,2,3,4])
m,s = x.mean(),x.std()
test_eq(norm.means['a'], m)
test_close(norm.stds['a'], s)
test_close(to['a'].values, (x-m)/s)
```

```python
df1 = pd.DataFrame({'a':[5,6,7]})
to1 = to.new(df1)
to1.process()
test_close(to1['a'].values, (np.array([5,6,7])-m)/s)
to2 = norm.decode(to1)
test_close(to2['a'].values, [5,6,7])
```

```python
norm = Normalize()
df = pd.DataFrame({'a':[0,1,2,3,4]})
to = TabularPandas(df, norm, cont_names='a', splits=[[0,1,2],[3,4]])
x = np.array([0,1,2])
m,s = x.mean(),x.std()
test_eq(norm.means['a'], m)
test_close(norm.stds['a'], s)
test_close(to['a'].values, (np.array([0,1,2,3,4])-m)/s)
```


<h2 id="FillStrategy" class="doc_header"><code>class</code> <code>FillStrategy</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L287" class="source_link" style="float:right">[source]</a></h2>

> <code>FillStrategy</code>()

Namespace containing the various filling strategies.


Currently, filling with the `median`, a `constant`, and the `mode` are supported.


<h3 id="FillMissing" class="doc_header"><code>class</code> <code>FillMissing</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L294" class="source_link" style="float:right">[source]</a></h3>

> <code>FillMissing</code>(**`fill_strategy`**=*`median`*, **`add_col`**=*`True`*, **`fill_vals`**=*`None`*) :: [`TabularProc`](/tabular.core.html#TabularProc)

Fill the missing values in continuous columns.


```python
fill1,fill2,fill3 = (FillMissing(fill_strategy=s) 
                     for s in [FillStrategy.median, FillStrategy.constant, FillStrategy.mode])
df = pd.DataFrame({'a':[0,1,np.nan,1,2,3,4]})
df1 = df.copy(); df2 = df.copy()
tos = (TabularPandas(df, fill1, cont_names='a'),
       TabularPandas(df1, fill2, cont_names='a'),
       TabularPandas(df2, fill3, cont_names='a'))
test_eq(fill1.na_dict, {'a': 1.5})
test_eq(fill2.na_dict, {'a': 0})
test_eq(fill3.na_dict, {'a': 1.0})

for t in tos: test_eq(t.cat_names, ['a_na'])

for to_,v in zip(tos, [1.5, 0., 1.]):
    test_eq(to_['a'].values, np.array([0, 1, v, 1, 2, 3, 4]))
    test_eq(to_['a_na'].values, np.array([0, 0, 1, 0, 0, 0, 0]))
```

```python
fill = FillMissing() 
df = pd.DataFrame({'a':[0,1,np.nan,1,2,3,4], 'b': [0,1,2,3,4,5,6]})
to = TabularPandas(df, fill, cont_names=['a', 'b'])
test_eq(fill.na_dict, {'a': 1.5})
test_eq(to.cat_names, ['a_na'])
test_eq(to['a'].values, np.array([0, 1, 1.5, 1, 2, 3, 4]))
test_eq(to['a_na'].values, np.array([0, 0, 1, 0, 0, 0, 0]))
test_eq(to['b'].values, np.array([0,1,2,3,4,5,6]))
```

```python
procs = [Normalize, Categorify, FillMissing, noop]
df = pd.DataFrame({'a':[0,1,2,1,1,2,0], 'b':[0,1,np.nan,1,2,3,4]})
to = TabularPandas(df, procs, cat_names='a', cont_names='b')

#Test setup and apply on df_main
test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to['a'], [1,2,3,2,2,3,1])
test_eq(to['b_na'], [1,1,2,1,1,1,1])
x = np.array([0,1,1.5,1,2,3,4])
m,s = x.mean(),x.std()
test_close(to['b'].values, (x-m)/s)
test_eq(to.classes, {'a': ['#na#',0,1,2], 'b_na': ['#na#',False,True]})
```

```python
df = pd.DataFrame({'a':[0,1,2,1,1,2,0], 'b':[0,1,np.nan,1,2,3,4], 'c': ['b','a','b','a','a','b','a']})
to = TabularPandas(df, procs, 'a', 'b', y_names='c')

test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to['a'], [1,2,3,2,2,3,1])
test_eq(to['b_na'], [1,1,2,1,1,1,1])
test_eq(to['c'], [1,0,1,0,0,1,0])
x = np.array([0,1,1.5,1,2,3,4])
m,s = x.mean(),x.std()
test_close(to['b'].values, (x-m)/s)
test_eq(to.classes, {'a': ['#na#',0,1,2], 'b_na': ['#na#',False,True]})
test_eq(to.vocab, ['a','b'])
```

```python
df = pd.DataFrame({'a':[0,1,2,1,1,2,0], 'b':[0,1,np.nan,1,2,3,4], 'c': ['b','a','b','a','a','b','a']})
to = TabularPandas(df, procs, 'a', 'b', y_names='c')

test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to['a'], [1,2,3,2,2,3,1])
test_eq(df.a.dtype,int)
test_eq(to['b_na'], [1,1,2,1,1,1,1])
test_eq(to['c'], [1,0,1,0,0,1,0])
```

```python
df = pd.DataFrame({'a':[0,1,2,1,1,2,0], 'b':[0,np.nan,1,1,2,3,4], 'c': ['b','a','b','a','a','b','a']})
to = TabularPandas(df, procs, cat_names='a', cont_names='b', y_names='c', splits=[[0,1,4,6], [2,3,5]])

test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to['a'], [1,2,2,1,0,2,0])
test_eq(df.a.dtype,int)
test_eq(to['b_na'], [1,2,1,1,1,1,1])
test_eq(to['c'], [1,0,0,0,1,0,1])
```


<h2 id="ReadTabBatch" class="doc_header"><code>class</code> <code>ReadTabBatch</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L321" class="source_link" style="float:right">[source]</a></h2>

> <code>ReadTabBatch</code>(**`to`**) :: [`ItemTransform`](https://fastcore.fast.ai/transform#ItemTransform)

A transform that always take tuples as items


```python
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter,_DatasetKind
```

```python
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)
```


<h2 id="TabDataLoader" class="doc_header"><code>class</code> <code>TabDataLoader</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L347" class="source_link" style="float:right">[source]</a></h2>

> <code>TabDataLoader</code>(**`dataset`**, **`bs`**=*`16`*, **`shuffle`**=*`False`*, **`after_batch`**=*`None`*, **`num_workers`**=*`0`*, **`verbose`**=*`False`*, **`do_setup`**=*`True`*, **`pin_memory`**=*`False`*, **`timeout`**=*`0`*, **`batch_size`**=*`None`*, **`drop_last`**=*`False`*, **`indexed`**=*`None`*, **`n`**=*`None`*, **`device`**=*`None`*, **`wif`**=*`None`*, **`before_iter`**=*`None`*, **`after_item`**=*`None`*, **`before_batch`**=*`None`*, **`after_iter`**=*`None`*, **`create_batches`**=*`None`*, **`create_item`**=*`None`*, **`create_batch`**=*`None`*, **`retain`**=*`None`*, **`get_idxs`**=*`None`*, **`sample`**=*`None`*, **`shuffle_fn`**=*`None`*, **`do_batch`**=*`None`*) :: [`TfmdDL`](/data.core.html#TfmdDL)

A transformed [`DataLoader`](/data.load.html#DataLoader) for Tabular data


## Integration example

For a more in-depth explanation, see the [tabular tutorial](http://docs.fast.ai/tutorial.tabular)

```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df_main,df_test = df.iloc[:10000].copy(),df.iloc[10000:].copy()
df_test.drop('salary', axis=1, inplace=True)
df_main.head()
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
splits = RandomSplitter()(range_of(df_main))
```

```python
to = TabularPandas(df_main, procs, cat_names, cont_names, y_names="salary", splits=splits)
```

```python
dls = to.dataloaders()
dls.valid.show_batch()
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
      <td>Some-college</td>
      <td>Married-spouse-absent</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>22.999999</td>
      <td>54472.005407</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Other-relative</td>
      <td>Black</td>
      <td>False</td>
      <td>21.000001</td>
      <td>236683.999905</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>18.000001</td>
      <td>163786.998406</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Local-gov</td>
      <td>Masters</td>
      <td>Divorced</td>
      <td>#na#</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>False</td>
      <td>44.000000</td>
      <td>135055.998622</td>
      <td>14.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Self-emp-inc</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Adm-clerical</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>40.000000</td>
      <td>207577.999886</td>
      <td>9.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>State-gov</td>
      <td>Masters</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>37.000000</td>
      <td>210451.999548</td>
      <td>14.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>?</td>
      <td>Bachelors</td>
      <td>Never-married</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>32.000000</td>
      <td>169885.999453</td>
      <td>13.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>20.000000</td>
      <td>236804.000495</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>31.000000</td>
      <td>137680.998667</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Self-emp-inc</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>46.000000</td>
      <td>284798.997462</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>


```python
to.show()
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
      <th>3380</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>33.0</td>
      <td>248584.0</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3158</th>
      <td>Local-gov</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>51.0</td>
      <td>110327.0</td>
      <td>13.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>8904</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>27.0</td>
      <td>133937.0</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>5912</th>
      <td>Self-emp-not-inc</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>48.0</td>
      <td>164582.0</td>
      <td>10.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>3583</th>
      <td>Private</td>
      <td>Masters</td>
      <td>Never-married</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>39.0</td>
      <td>49020.0</td>
      <td>14.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>2945</th>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>26.0</td>
      <td>166051.0</td>
      <td>13.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>204</th>
      <td>?</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>#na#</td>
      <td>Husband</td>
      <td>White</td>
      <td>True</td>
      <td>60.0</td>
      <td>174073.0</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3196</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>21.0</td>
      <td>241367.0</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>1183</th>
      <td>?</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>?</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>65.0</td>
      <td>52728.0</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>2829</th>
      <td>Private</td>
      <td>Masters</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>46.0</td>
      <td>261059.0</td>
      <td>14.0</td>
      <td>&gt;=50k</td>
    </tr>
  </tbody>
</table>


We can decode any set of transformed data by calling `to.decode_row` with our raw data:

```python
row = to.items.iloc[0]
to.decode_row(row)
```




    age                                  33
    workclass                       Private
    fnlwgt                           248584
    education                  Some-college
    education-num                        10
    marital-status       Married-civ-spouse
    occupation                Other-service
    relationship                    Husband
    race                              White
    sex                                Male
    capital-gain                          0
    capital-loss                          0
    hours-per-week                       50
    native-country            United-States
    salary                             <50k
    education-num_na                  False
    Name: 3380, dtype: object



```python
to_tst = to.new(df_test)
to_tst.process()
to_tst.items.head()
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
      <th>education-num_na</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10000</th>
      <td>0.466910</td>
      <td>5</td>
      <td>1.359596</td>
      <td>10</td>
      <td>1.170520</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Philippines</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10001</th>
      <td>-0.932292</td>
      <td>5</td>
      <td>1.271990</td>
      <td>12</td>
      <td>-0.425893</td>
      <td>3</td>
      <td>15</td>
      <td>1</td>
      <td>4</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10002</th>
      <td>1.056047</td>
      <td>5</td>
      <td>0.161911</td>
      <td>2</td>
      <td>-1.224099</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>5</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>United-States</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10003</th>
      <td>0.540552</td>
      <td>5</td>
      <td>-0.274100</td>
      <td>12</td>
      <td>-0.425893</td>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>43</td>
      <td>United-States</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10004</th>
      <td>0.761479</td>
      <td>6</td>
      <td>1.462819</td>
      <td>9</td>
      <td>0.372313</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>United-States</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
tst_dl = dls.valid.new(to_tst)
tst_dl.show_batch()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Adm-clerical</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>False</td>
      <td>45.000000</td>
      <td>338105.000172</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Transport-moving</td>
      <td>Husband</td>
      <td>Other</td>
      <td>False</td>
      <td>26.000000</td>
      <td>328662.996625</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>11th</td>
      <td>Divorced</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>52.999999</td>
      <td>209021.999484</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>False</td>
      <td>46.000000</td>
      <td>162030.001554</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Self-emp-inc</td>
      <td>Assoc-voc</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>49.000000</td>
      <td>349230.005561</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Local-gov</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>34.000000</td>
      <td>124827.001916</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Self-emp-inc</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>52.999999</td>
      <td>290640.000454</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>19.000000</td>
      <td>106273.002866</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Protective-serv</td>
      <td>Husband</td>
      <td>Black</td>
      <td>False</td>
      <td>71.999999</td>
      <td>53683.997254</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>20.000000</td>
      <td>505980.004555</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>


## Other target types

### Multi-label categories

#### one-hot encoded label

```python
def _mock_multi_label(df):
    sal,sex,white = [],[],[]
    for row in df.itertuples():
        sal.append(row.salary == '>=50k')
        sex.append(row.sex == ' Male')
        white.append(row.race == ' White')
    df['salary'] = np.array(sal)
    df['male']   = np.array(sex)
    df['white']  = np.array(white)
    return df
```

```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df_main,df_test = df.iloc[:10000].copy(),df.iloc[10000:].copy()
df_main = _mock_multi_label(df_main)
```

```python
df_main.head()
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
      <th>male</th>
      <th>white</th>
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
      <td>True</td>
      <td>False</td>
      <td>True</td>
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
      <td>True</td>
      <td>True</td>
      <td>True</td>
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
      <td>False</td>
      <td>False</td>
      <td>False</td>
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
      <td>True</td>
      <td>True</td>
      <td>False</td>
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
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




<h4 id="setups" class="doc_header"><code>setups</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L373" class="source_link" style="float:right">[source]</a></h4>

> <code>setups</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))





<h4 id="encodes" class="doc_header"><code>encodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L379" class="source_link" style="float:right">[source]</a></h4>

> <code>encodes</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))





<h4 id="decodes" class="doc_header"><code>decodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L382" class="source_link" style="float:right">[source]</a></h4>

> <code>decodes</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))




```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter()(range_of(df_main))
y_names=["salary", "male", "white"]
```

```python
%time to = TabularPandas(df_main, procs, cat_names, cont_names, y_names=y_names, y_block=MultiCategoryBlock(encoded=True, vocab=y_names), splits=splits)
```

    CPU times: user 77.2 ms, sys: 238 µs, total: 77.4 ms
    Wall time: 76.7 ms


```python
dls = to.dataloaders()
dls.valid.show_batch()
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
      <th>male</th>
      <th>white</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>#na#</td>
      <td>Husband</td>
      <td>Amer-Indian-Eskimo</td>
      <td>True</td>
      <td>30.000000</td>
      <td>216811.000739</td>
      <td>10.000000</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>#na#</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>53.000000</td>
      <td>96061.998009</td>
      <td>13.000000</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Adm-clerical</td>
      <td>Wife</td>
      <td>White</td>
      <td>False</td>
      <td>31.000000</td>
      <td>196787.999901</td>
      <td>9.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>?</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>?</td>
      <td>Husband</td>
      <td>White</td>
      <td>True</td>
      <td>65.999999</td>
      <td>177351.000226</td>
      <td>10.000000</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Private</td>
      <td>10th</td>
      <td>Separated</td>
      <td>Sales</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>False</td>
      <td>21.000000</td>
      <td>353628.005662</td>
      <td>5.999999</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>40.000000</td>
      <td>143045.999229</td>
      <td>13.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Private</td>
      <td>Masters</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>37.000000</td>
      <td>117381.002561</td>
      <td>14.000000</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>29.000000</td>
      <td>183854.000291</td>
      <td>9.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>Priv-house-serv</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>54.999999</td>
      <td>175942.000053</td>
      <td>9.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Widowed</td>
      <td>Tech-support</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>False</td>
      <td>64.000000</td>
      <td>91342.999448</td>
      <td>10.000000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>


#### Not one-hot encoded

```python
def _mock_multi_label(df):
    targ = []
    for row in df.itertuples():
        labels = []
        if row.salary == '>=50k': labels.append('>50k')
        if row.sex == ' Male':   labels.append('male')
        if row.race == ' White': labels.append('white')
        targ.append(' '.join(labels))
    df['target'] = np.array(targ)
    return df
```

```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df_main,df_test = df.iloc[:10000].copy(),df.iloc[10000:].copy()
df_main = _mock_multi_label(df_main)
```

```python
df_main.head()
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
      <th>target</th>
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
      <td>&gt;50k white</td>
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
      <td>&gt;50k male white</td>
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
      <td></td>
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
      <td>&gt;50k male</td>
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
      <td></td>
    </tr>
  </tbody>
</table>
</div>



```python
@MultiCategorize
def encodes(self, to:Tabular): 
    #to.transform(to.y_names, partial(_apply_cats, {n: self.vocab for n in to.y_names}, 0))
    return to
  
@MultiCategorize
def decodes(self, to:Tabular): 
    #to.transform(to.y_names, partial(_decode_cats, {n: self.vocab for n in to.y_names}))
    return to
```

```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter()(range_of(df_main))
```

```python
%time to = TabularPandas(df_main, procs, cat_names, cont_names, y_names="target", y_block=MultiCategoryBlock(), splits=splits)
```

    CPU times: user 81 ms, sys: 178 µs, total: 81.2 ms
    Wall time: 80.1 ms


```python
to.procs[2].vocab
```




    (#24) ['-','_','a','c','d','e','f','g','h','i'...]



### Regression


<h4 id="setups" class="doc_header"><code>setups</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L373" class="source_link" style="float:right">[source]</a></h4>

> <code>setups</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))





<h4 id="encodes" class="doc_header"><code>encodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L379" class="source_link" style="float:right">[source]</a></h4>

> <code>encodes</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))





<h4 id="decodes" class="doc_header"><code>decodes</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L382" class="source_link" style="float:right">[source]</a></h4>

> <code>decodes</code>(**`to`**:[`Tabular`](/tabular.core.html#Tabular))




```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df_main,df_test = df.iloc[:10000].copy(),df.iloc[10000:].copy()
df_main = _mock_multi_label(df_main)
```

```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter()(range_of(df_main))
```

```python
%time to = TabularPandas(df_main, procs, cat_names, cont_names, y_names='age', splits=splits)
```

    CPU times: user 82.2 ms, sys: 508 µs, total: 82.7 ms
    Wall time: 81.8 ms


```python
to.procs[-1].means
```




    {'fnlwgt': 193046.84475, 'education-num': 10.08025}



```python
dls = to.dataloaders()
dls.valid.show_batch()
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
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>State-gov</td>
      <td>Masters</td>
      <td>Never-married</td>
      <td>#na#</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>47569.994748</td>
      <td>14.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Federal-gov</td>
      <td>11th</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>False</td>
      <td>166418.999287</td>
      <td>7.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>9th</td>
      <td>Divorced</td>
      <td>Farming-fishing</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>False</td>
      <td>225603.000537</td>
      <td>5.0</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Local-gov</td>
      <td>12th</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>48055.004282</td>
      <td>8.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Federal-gov</td>
      <td>Prof-school</td>
      <td>Divorced</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>66504.003988</td>
      <td>15.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>Asian-Pac-Islander</td>
      <td>False</td>
      <td>91274.998927</td>
      <td>10.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>391584.996528</td>
      <td>13.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Self-emp-not-inc</td>
      <td>1st-4th</td>
      <td>Divorced</td>
      <td>Craft-repair</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>130435.999390</td>
      <td>2.0</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>62507.003940</td>
      <td>13.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>236696.000903</td>
      <td>9.0</td>
      <td>24.0</td>
    </tr>
  </tbody>
</table>


## Not being used now - for multi-modal

```python
class TensorTabular(fastuple):
    def get_ctxs(self, max_n=10, **kwargs):
        n_samples = min(self[0].shape[0], max_n)
        df = pd.DataFrame(index = range(n_samples))
        return [df.iloc[i] for i in range(n_samples)]

    def display(self, ctxs): display_df(pd.DataFrame(ctxs))

class TabularLine(pd.Series):
    "A line of a dataframe that knows how to show itself"
    def show(self, ctx=None, **kwargs): return self if ctx is None else ctx.append(self)

class ReadTabLine(ItemTransform):
    def __init__(self, proc): self.proc = proc

    def encodes(self, row):
        cats,conts = (o.map(row.__getitem__) for o in (self.proc.cat_names,self.proc.cont_names))
        return TensorTabular(tensor(cats).long(),tensor(conts).float())

    def decodes(self, o):
        to = TabularPandas(o, self.proc.cat_names, self.proc.cont_names, self.proc.y_names)
        to = self.proc.decode(to)
        return TabularLine(pd.Series({c: v for v,c in zip(to.items[0]+to.items[1], self.proc.cat_names+self.proc.cont_names)}))

class ReadTabTarget(ItemTransform):
    def __init__(self, proc): self.proc = proc
    def encodes(self, row): return row[self.proc.y_names].astype(np.int64)
    def decodes(self, o): return Category(self.proc.classes[self.proc.y_names][o])
```

```python
# enc = tds[1]
# test_eq(enc[0][0], tensor([2,1]))
# test_close(enc[0][1], tensor([-0.628828]))
# test_eq(enc[1], 1)

# dec = tds.decode(enc)
# assert isinstance(dec[0], TabularLine)
# test_close(dec[0], pd.Series({'a': 1, 'b_na': False, 'b': 1}))
# test_eq(dec[1], 'a')

# test_stdout(lambda: print(show_at(tds, 1)), """a               1
# b_na        False
# b               1
# category        a
# dtype: object""")
```
