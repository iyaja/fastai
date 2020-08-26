# Tabular training
> How to use the tabular application in fastai


To illustrate the tabular application, we will use the example of the [Adult dataset](https://archive.ics.uci.edu/ml/datasets/Adult) where we have to predict if a person is earning more or less than $50k per year using some general data.

```python
from fastai.tabular.all import *
```

We can download a sample of this dataset with the usual [`untar_data`](/data.external.html#untar_data) command:

```python
path = untar_data(URLs.ADULT_SAMPLE)
path.ls()
```








    (#3) [Path('/root/.fastai/data/adult_sample/adult.csv'),Path('/root/.fastai/data/adult_sample/models'),Path('/root/.fastai/data/adult_sample/export.pkl')]



Then we can have a look at how the data is structured:

```python
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



Some of the columns are continuous (like age) and we will treat them as float numbers we can feed our model directly. Others are categorical (like workclass or education) and we will convert them to a unique index that we will feed to embedding layers. We can specify our categorical and continuous column names, as well as the name of the dependent variable in [`TabularDataLoaders`](/tabular.data.html#TabularDataLoaders) factory methods:

```python
dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])
```

The last part is the list of pre-processors we apply to our data:

- [`Categorify`](/tabular.core.html#Categorify) is going to take every categorical variable and make a map from integer to unique categories, then replace the values by the corresponding index.
- [`FillMissing`](/tabular.core.html#FillMissing) will fill the missing values in the continuous variables by the median of existing values (you can choose a specific value if you prefer)
- [`Normalize`](/data.transforms.html#Normalize) will normalize the continuous variables (substract the mean and divide by the std)



To further expose what's going on below the surface, let's rewrite this utilizing `fastai`'s [`TabularPandas`](/tabular.core.html#TabularPandas) class. We will need to make one adjustment, which is defining how we want to split our data. By default the factory method above used a random 80/20 split, so we will do the same:

```python
splits = RandomSplitter(valid_pct=0.2)(range_of(df))
```

```python
to = TabularPandas(df, procs=[Categorify, FillMissing,Normalize],
                   cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
                   cont_names = ['age', 'fnlwgt', 'education-num'],
                   y_names='salary',
                   splits=splits)
```

Before finally building our [`DataLoaders`](/data.core.html#DataLoaders) again:

```python
dls = to.dataloaders(bs=64)
```

> Later we will explore why using [`TabularPandas`](/tabular.core.html#TabularPandas) to preprocess will be valuable.

The `show_batch` method works like for every other application:

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
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>White</td>
      <td>False</td>
      <td>41.000000</td>
      <td>75409.001182</td>
      <td>13.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Craft-repair</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>24.000000</td>
      <td>38455.005013</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>48.000000</td>
      <td>101299.003093</td>
      <td>12.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Other-relative</td>
      <td>Black</td>
      <td>False</td>
      <td>42.000000</td>
      <td>227465.999281</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>State-gov</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>20.999999</td>
      <td>258489.997130</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Local-gov</td>
      <td>12th</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>39.000000</td>
      <td>207853.000067</td>
      <td>8.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Private</td>
      <td>Assoc-voc</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>36.000000</td>
      <td>238414.998930</td>
      <td>11.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Craft-repair</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>19.000000</td>
      <td>445727.998937</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Local-gov</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>#na#</td>
      <td>Husband</td>
      <td>White</td>
      <td>True</td>
      <td>59.000000</td>
      <td>196013.000174</td>
      <td>10.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>False</td>
      <td>39.000000</td>
      <td>147500.000403</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>


We can define a model using the [`tabular_learner`](/tabular.learner.html#tabular_learner) method. When we define our model, `fastai` will try to infer the loss function based on our `y_names` earlier. 

**Note**: Sometimes with tabular data, your `y`'s may be encoded (such as 0 and 1). In such a case you should explicitly pass `y_block = CategoryBlock` in your constructor so `fastai` won't presume you are doing regression.

```python
learn = tabular_learner(dls, metrics=accuracy)
```

And we can train that model with the `fit_one_cycle` method (the `fine_tune` method won't be useful here since we don't have a pretrained model).

```python
learn.fit_one_cycle(1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.369360</td>
      <td>0.348096</td>
      <td>0.840756</td>
      <td>00:05</td>
    </tr>
  </tbody>
</table>


We can then have a look at some predictions:

```python
learn.show_results()
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
      <th>salary_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.324868</td>
      <td>-1.138177</td>
      <td>-0.424022</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>-0.482055</td>
      <td>-1.351911</td>
      <td>1.148438</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>-0.775482</td>
      <td>0.138709</td>
      <td>-0.424022</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>16.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>-1.362335</td>
      <td>-0.227515</td>
      <td>-0.030907</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>-1.509048</td>
      <td>-0.191191</td>
      <td>-1.210252</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>16.0</td>
      <td>3.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.498575</td>
      <td>-0.051096</td>
      <td>-0.030907</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>15.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>-0.555412</td>
      <td>0.039167</td>
      <td>-0.424022</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>-1.582405</td>
      <td>-1.396391</td>
      <td>-1.603367</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>-1.362335</td>
      <td>0.158354</td>
      <td>-0.817137</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>


Or use the predict method on a row:

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
      <td>101319.99788</td>
      <td>12.0</td>
      <td>&gt;=50k</td>
    </tr>
  </tbody>
</table>


```python
clas, probs
```




    (tensor(1), tensor([0.4995, 0.5005]))



To get prediction on a new dataframe, you can use the `test_dl` method of the [`DataLoaders`](/data.core.html#DataLoaders). That dataframe does not need to have the dependent variable in its column.

```python
test_df = df.copy()
test_df.drop(['salary'], axis=1, inplace=True)
dl = learn.dls.test_dl(test_df)
```

Then [`Learner.get_preds`](/learner.html#Learner.get_preds) will give you the predictions:

```python
learn.get_preds(dl=dl)
```








    (tensor([[0.4995, 0.5005],
             [0.4882, 0.5118],
             [0.9824, 0.0176],
             ...,
             [0.5324, 0.4676],
             [0.7628, 0.2372],
             [0.5934, 0.4066]]), None)



## `fastai` with Other Libraries

As mentioned earlier, [`TabularPandas`](/tabular.core.html#TabularPandas) is a powerful and easy preprocessing tool for tabular data. Integration with libraries such as Random Forests and XGBoost requires only one extra step, that the `.dataloaders` call did for us. Let's look at our `to` again. It's values are stored in a `DataFrame` like object, where we can extract the `cats`, `conts,` `xs` and `ys` if we want to:

```python
to.xs[:3]
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
      <th>25387</th>
      <td>5</td>
      <td>16</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0.471582</td>
      <td>-1.467756</td>
      <td>-0.030907</td>
    </tr>
    <tr>
      <th>16872</th>
      <td>1</td>
      <td>16</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>-1.215622</td>
      <td>-0.649792</td>
      <td>-0.030907</td>
    </tr>
    <tr>
      <th>25852</th>
      <td>5</td>
      <td>16</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1.865358</td>
      <td>-0.218915</td>
      <td>-0.030907</td>
    </tr>
  </tbody>
</table>
</div>



To then preprocess our data, all we need to do is call `process` to apply all of our `procs` inplace:

```python
to.process()
to.xs[:3]
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
      <th>25387</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-3.034491</td>
      <td>-1.792679</td>
      <td>-5.524377</td>
    </tr>
    <tr>
      <th>16872</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-3.043570</td>
      <td>-1.792679</td>
      <td>-5.524377</td>
    </tr>
    <tr>
      <th>25852</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-3.026991</td>
      <td>-1.792679</td>
      <td>-5.524377</td>
    </tr>
  </tbody>
</table>
</div>



Now that everything is encoded, you can then send this off to XGBoost or Random Forests by extracting the train and validation sets and their values:

```python
X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()
```

And now we can directly send this in!
