# fastai applications - quick start



fastai's applications all use the same basic steps and code:

- Create appropriate [`DataLoaders`](/data.core.html#DataLoaders)
- Create a [`Learner`](/learner.html#Learner)
- Call a *fit* method
- Make predictions or view results.

In this quick start, we'll show these steps for a wide range of difference applications and datasets. As you'll see, the code in each case is extremely similar, despite the very different models and data being used.

## Computer vision classification

The code below does the following things:

1. A dataset called the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) that contains 7,349 images of cats and dogs from 37 different breeds will be downloaded from the fast.ai datasets collection to the GPU server you are using, and will then be extracted.
2. A *pretrained model* that has already been trained on 1.3 million images, using a competition-winning model will be downloaded from the internet.
3. The pretrained model will be *fine-tuned* using the latest advances in transfer learning, to create a model that is specially customized for recognizing dogs and cats.

The first two steps only need to be run once. If you run it again, it will use the dataset and model that have already been downloaded, rather than downloading them again.

```python
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.173790</td>
      <td>0.018827</td>
      <td>0.005413</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.064295</td>
      <td>0.013404</td>
      <td>0.005413</td>
      <td>00:14</td>
    </tr>
  </tbody>
</table>


You can do inference with your model with the `predict` method:

```python
img = PILImage.create('images/cat.jpg')
img
```




![png](output_6_0.png)



```python
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
```





    Is this a cat?: True.
    Probability it's a cat: 0.999722


### Computer vision segmentation

Here is how we can train a segmentation model with fastai, using a subset of the [*Camvid* dataset](http://www0.cs.ucl.ac.uk/staff/G.Brostow/papers/Brostow_2009-PRL.pdf):

```python
path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str)
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)
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
      <td>2.882460</td>
      <td>2.096923</td>
      <td>00:03</td>
    </tr>
  </tbody>
</table>



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
      <td>1.602270</td>
      <td>1.543582</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.417732</td>
      <td>1.225782</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.307454</td>
      <td>1.071090</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.170338</td>
      <td>0.884501</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.047036</td>
      <td>0.799820</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.947965</td>
      <td>0.754801</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.868178</td>
      <td>0.728161</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.804939</td>
      <td>0.720942</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>


We can visualize how well it achieved its task, by asking the model to color-code each pixel of an image.

```python
learn.show_results(max_n=6, figsize=(7,8))
```






![png](output_12_1.png)


## Natural language processing

Here is all of the code necessary to train a model that can classify the sentiment of a movie review better than anything that existed in the world just five years ago:

```python
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(2, 1e-2)
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
      <td>0.594912</td>
      <td>0.407416</td>
      <td>0.823640</td>
      <td>01:35</td>
    </tr>
  </tbody>
</table>



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
      <td>0.268259</td>
      <td>0.316242</td>
      <td>0.876000</td>
      <td>03:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.184861</td>
      <td>0.246242</td>
      <td>0.898080</td>
      <td>03:10</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.136392</td>
      <td>0.220086</td>
      <td>0.918200</td>
      <td>03:16</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.106423</td>
      <td>0.191092</td>
      <td>0.931360</td>
      <td>03:15</td>
    </tr>
  </tbody>
</table>


Predictions are done with `predict`, as for computer vision:

```python
learn.predict("I really liked that movie!")
```








    ('pos', tensor(1), tensor([0.0041, 0.9959]))



## Tabular

Building models from plain *tabular* data is done using the same basic steps as the previous models. Here is the code necessary to train a model that will predict whether a person is a high-income earner, based on their socioeconomic background:

```python
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(2)
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
      <td>0.372298</td>
      <td>0.359698</td>
      <td>0.829392</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.357530</td>
      <td>0.349440</td>
      <td>0.837377</td>
      <td>00:06</td>
    </tr>
  </tbody>
</table>


## Recommendation systems

Recommendation systems are very important, particularly in e-commerce. Companies like Amazon and Netflix try hard to recommend products or movies that users might like. Here's how to train a model that will predict movies people might like, based on their previous viewing habits, using the [MovieLens dataset](https://doi.org/10.1145/2827872):

```python
path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
learn = collab_learner(dls, y_range=(0.5,5.5))
learn.fine_tune(6)
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
      <td>1.497551</td>
      <td>1.435720</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



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
      <td>1.332337</td>
      <td>1.351769</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.180177</td>
      <td>1.046801</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.913091</td>
      <td>0.799319</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.749806</td>
      <td>0.731218</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.686577</td>
      <td>0.715372</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.665683</td>
      <td>0.713309</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


We can use the same `show_results` call we saw earlier to view a few examples of user and movie IDs, actual ratings, and predictions:

```python
learn.show_results()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>rating_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.985477</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>62.0</td>
      <td>4.0</td>
      <td>3.629225</td>
    </tr>
    <tr>
      <th>2</th>
      <td>91.0</td>
      <td>81.0</td>
      <td>1.0</td>
      <td>3.476280</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48.0</td>
      <td>26.0</td>
      <td>2.0</td>
      <td>4.043919</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75.0</td>
      <td>54.0</td>
      <td>3.0</td>
      <td>4.023057</td>
    </tr>
    <tr>
      <th>5</th>
      <td>42.0</td>
      <td>22.0</td>
      <td>3.0</td>
      <td>3.509050</td>
    </tr>
    <tr>
      <th>6</th>
      <td>40.0</td>
      <td>59.0</td>
      <td>4.0</td>
      <td>3.686552</td>
    </tr>
    <tr>
      <th>7</th>
      <td>63.0</td>
      <td>77.0</td>
      <td>3.0</td>
      <td>2.862713</td>
    </tr>
    <tr>
      <th>8</th>
      <td>32.0</td>
      <td>61.0</td>
      <td>4.0</td>
      <td>4.356578</td>
    </tr>
  </tbody>
</table>

