# Computer vision
> Using the fastai library in computer vision.


```python
from fastai.vision.all import *
```

This tutorial highlights on how to quickly build a [`Learner`](/learner.html#Learner) and fine tune a pretrained model on most computer vision tasks. 

## Single-label classification

For this task, we will use the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) that contains images of cats and dogs of 37 different breeds. We will first show how to build a simple cat-vs-dog classifier, then a little bit more advanced model that can classify all breeds.

The dataset can be downloaded and decompressed with this line of code:

```python
path = untar_data(URLs.PETS)
```

It will only do this download once, and return the location of the decompressed archive. We can check what is inside with the `.ls()` method.

```python
path.ls()
```




    (#3) [Path('/home/ashwin/.fastai/data/oxford-iiit-pet/images'),Path('/home/ashwin/.fastai/data/oxford-iiit-pet/models'),Path('/home/ashwin/.fastai/data/oxford-iiit-pet/annotations')]



We will ignore the annotations folder for now, and focus on the images one. [`get_image_files`](/data.transforms.html#get_image_files) is a fastai function that helps us grab all the image files (recursively) in one folder.

```python
files = get_image_files(path/"images")
len(files)
```




    7390



### Cats vs dogs

To label our data for the cats vs dogs problem, we need to know which filenames are of dog pictures and which ones are of cat pictures. There is an easy way to distinguish: the name of the file begins with a capital for cats, and a lowercased letter for dogs:

```python
files[0],files[6]
```




    (Path('/home/ashwin/.fastai/data/oxford-iiit-pet/images/yorkshire_terrier_102.jpg'),
     Path('/home/ashwin/.fastai/data/oxford-iiit-pet/images/great_pyrenees_102.jpg'))



We can then define an easy label function:

```python
def label_func(f): return f[0].isupper()
```

To get our data ready for a model, we need to put it in a [`DataLoaders`](/data.core.html#DataLoaders) object. Here we have a function that labels using the file names, so we will use [`ImageDataLoaders.from_name_func`](/vision.data.html#ImageDataLoaders.from_name_func). There are other factory methods of [`ImageDataLoaders`](/vision.data.html#ImageDataLoaders) that could be more suitable for your problem, so make sure to check them all in [`vision.data`](/vision.data.html). 

```python
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
```

We have passed to this function the directory we're working in, the `files` we grabbed, our `label_func` and one last piece as `item_tfms`: this is a [`Transform`](https://fastcore.fast.ai/transform#Transform) applied on all items of our dataset that will resize each imge to 224 by 224, by using a random crop on the largest dimension to make it a square, then resizing to 224 by 224. If we didn't pass this, we would get an error later as it would be impossible to batch the items together.

We can then check if everything looks okay with the `show_batch` method (`True` is for cat, `False` is for dog):

```python
dls.show_batch()
```


![png](output_18_0.png)


Then we can create a [`Learner`](/learner.html#Learner), which is a fastai object that combines the data and a model for training, and uses transfer learning to fine tune a pretrained model in just two lines of code:

```python
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
      <td>0.138543</td>
      <td>0.023240</td>
      <td>0.008119</td>
      <td>00:18</td>
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
      <td>0.055319</td>
      <td>0.017367</td>
      <td>0.004736</td>
      <td>00:24</td>
    </tr>
  </tbody>
</table>


The first line downloaded a model called ResNet34, pretrained on [ImageNet](http://www.image-net.org/), and adapted it to our specific problem. It then fine tuned that model and in a relatively short time, we get a model with an error rate of 0.3%... amazing!

If you want to make a prediction on a new image, you can use `learn.predict`:

```python
learn.predict(files[0])
```








    ('False', tensor(0), tensor([9.9970e-01, 2.9784e-04]))



The predict method returns three things: the decoded prediction (here `False` for dog), the index of the predicted class and the tensor of probabilities that our image is one of a dog (here the model is quite confident!) This method accepts a filename, a PIL image or a tensor directly in this case.

We can also have a look at some predictions with the `show_results` method:

```python
learn.show_results()
```






![png](output_24_1.png)


Check out the other applications like text or tabular, or the other problems covered in this tutorial, and you will see they all share a consistent API for gathering the data and look at it, create a [`Learner`](/learner.html#Learner), train the model and look at some predictions.

### Classifying breeds

To label our data with the breed name, we will use a regular expression to extract it from the filename. Looking back at a filename, we have:

```python
files[0].name
```




    'yorkshire_terrier_187.jpg'



so the class is everything before the last `_` followed by some digits. A regular expression that will catch the name is thus:

```python
pat = r'^(.*)_\d+.jpg'
```

Since it's pretty common to use regular expressions to label the data (often, labels are hidden in the file names), there is a factory method to do just that:

```python
dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(224))
```

Like before, we can then use `show_batch` to have a look at our data:

```python
dls.show_batch()
```


![png](output_34_0.png)


Since classifying the exact breed of cats or dogs amongst 37 different breeds is a harder problem, we will slightly change the definition of our [`DataLoaders`](/data.core.html#DataLoaders) to use data augmentation:

```python
dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(460),
                                    batch_tfms=aug_transforms(size=224))
```

This time we resized to a larger size before batching, and we add `batch_tfms`. [`aug_transforms`](/vision.augment.html#aug_transforms) is a function that provides a collection of data augmentation transforms with defaults we found worked very well on most datasets (you can customize each one by passing the right arguments).

```python
dls.show_batch()
```


![png](output_38_0.png)


We can then create our [`Learner`](/learner.html#Learner) exactly as before and train our model.

```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
```

We used the default learning rate before, but we might want to find the best one possible. For this, we can use the learning rate finder:

```python
learn.lr_find()
```








    (0.010000000149011612, 0.00363078061491251)




![png](output_42_2.png)


It plots the graph of the learning rate finder and gives us two suggestions (minimum divided by 10 and steepest gradient). Let's use `3e-3` here. We will also do a bit more epochs:

```python
learn.fine_tune(4, 3e-3)
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
      <td>1.282734</td>
      <td>0.274779</td>
      <td>0.085250</td>
      <td>00:14</td>
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
      <td>0.464303</td>
      <td>0.334058</td>
      <td>0.102165</td>
      <td>00:17</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.433266</td>
      <td>0.319324</td>
      <td>0.094723</td>
      <td>00:17</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.259503</td>
      <td>0.202348</td>
      <td>0.066306</td>
      <td>00:17</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.145520</td>
      <td>0.183488</td>
      <td>0.060217</td>
      <td>00:17</td>
    </tr>
  </tbody>
</table>


Again, we can have a look at some predictions with `show_results`:

```python
learn.show_results()
```






![png](output_46_1.png)


Another thing that is useful is an interpretation object, it can show us where the model made the worse predictions:

```python
interp = Interpretation.from_learner(learn)
```





```python
interp.plot_top_losses(9, figsize=(15,10))
```


![png](output_49_0.png)


### With the data block API

We can also use the data block API to get our data in a [`DataLoaders`](/data.core.html#DataLoaders). This is a bit more advanced, so fell free to skip this part if you are not comfortable with learning new API's just yet.

A datablock is built by giving the fastai library a bunch of informations:

- the types used, through an argument called `blocks`: here we have images and categories, so we pass [`ImageBlock`](/vision.data.html#ImageBlock) and [`CategoryBlock`](/data.block.html#CategoryBlock).
- how to get the raw items, here our function [`get_image_files`](/data.transforms.html#get_image_files).
- how to label those items, here with the same regular expression as before.
- how to split those items, here with a random splitter.
- the `item_tfms` and `batch_tfms` like before.

```python
pets = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224))
```

The pets object by itself is empty: it only containes the functions that will help us gather the data. We have to call `dataloaders` method to get a [`DataLoaders`](/data.core.html#DataLoaders). We pass it the source of the data:

```python
dls = pets.dataloaders(untar_data(URLs.PETS)/"images")
```

Then we can look at some of our pictures with `dls.show_batch()`

```python
dls.show_batch(max_n=9)
```


![png](output_56_0.png)


## Multi-label classification

For this task, we will use the [Pascal Dataset](http://host.robots.ox.ac.uk/pascal/VOC/) that contains images with different kinds of objects/persons. It's orginally a dataset for object detection, meaning the task is not only to detect if there is an instance of one class of an image, but to also draw a bounding box around it. Here we will just try to predict all the classes in one given image.

Multi-label classification defers from before in the sense each image does not belong to one category. An image could have a person *and* a horse inside it for instance. Or have none of the categories we study.

As before, we can download the dataset pretty easily:

```python
path = untar_data(URLs.PASCAL_2007)
path.ls()
```




    (#9) [Path('/home/ashwin/.fastai/data/pascal_2007/valid.json'),Path('/home/ashwin/.fastai/data/pascal_2007/segmentation'),Path('/home/ashwin/.fastai/data/pascal_2007/train.csv'),Path('/home/ashwin/.fastai/data/pascal_2007/test.csv'),Path('/home/ashwin/.fastai/data/pascal_2007/models'),Path('/home/ashwin/.fastai/data/pascal_2007/test'),Path('/home/ashwin/.fastai/data/pascal_2007/train.json'),Path('/home/ashwin/.fastai/data/pascal_2007/train'),Path('/home/ashwin/.fastai/data/pascal_2007/test.json')]



The information about the labels of each image is in the file named `train.csv`. We load it using pandas:

```python
df = pd.read_csv(path/'train.csv')
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
      <th>fname</th>
      <th>labels</th>
      <th>is_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000005.jpg</td>
      <td>chair</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000007.jpg</td>
      <td>car</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000009.jpg</td>
      <td>horse person</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000012.jpg</td>
      <td>car</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>000016.jpg</td>
      <td>bicycle</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### Using the high-level API

That's pretty straightforward: for each filename, we get the different labels (separated by space) and the last column tells if it's in the validation set or not. To get this in [`DataLoaders`](/data.core.html#DataLoaders) quickly, we have a factory method, `from_df`. We can specify the underlying path where all the images are, an additional folder to add between the base path and the filenames (here `train`), the `valid_col` to consider for the validation set (if we don't specify this, we take a random subset), a `label_delim` to split the labels and, as before, `item_tfms` and `batch_tfms`.

Note that we don't have to specify the `fn_col` and the `label_col` because they default to the first and second column respectively.

```python
dls = ImageDataLoaders.from_df(df, path, folder='train', valid_col='is_valid', label_delim=' ',
                               item_tfms=Resize(460), batch_tfms=aug_transforms(size=224))
```

As before, we can then have a look at the data with the `show_batch` method.

```python
dls.show_batch()
```


![png](output_66_0.png)


Training a model is as easy as before: the same functions can be applied and the fastai library will automatically detect that we are in a multi-label problem, thus picking the right loss function. The only difference is in the metric we pass: [`error_rate`](/metrics.html#error_rate) will not work for a multi-label problem, but we can use `accuracy_thresh`.

```python
learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.5))
```

As before, we can use `learn.lr_find` to pick a good learning rate:

```python
learn.lr_find()
```








    (0.025118863582611083, 0.033113110810518265)




![png](output_70_2.png)


We can pick the suggested learning rate and fine-tune our pretrained model:

```python
learn.fine_tune(4, 3e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.429346</td>
      <td>0.128341</td>
      <td>0.958785</td>
      <td>00:15</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.161289</td>
      <td>0.466793</td>
      <td>0.925856</td>
      <td>00:17</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.167091</td>
      <td>0.269290</td>
      <td>0.945757</td>
      <td>00:16</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.145694</td>
      <td>0.121434</td>
      <td>0.956355</td>
      <td>00:17</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.118979</td>
      <td>0.105355</td>
      <td>0.962570</td>
      <td>00:17</td>
    </tr>
  </tbody>
</table>


Like before, we can easily have a look at the results:

```python
learn.show_results()
```






![png](output_74_1.png)


Or get the predictions on a given image:

```python
learn.predict(path/'train/000005.jpg')
```








    ((#1) ['chair'],
     tensor([False, False, False, False, False, False, False, False,  True, False,
             False, False, False, False, False, False, False, False, False, False]),
     tensor([6.0259e-05, 6.2471e-04, 2.9542e-04, 3.7423e-04, 3.9715e-02, 1.9572e-03,
             7.3608e-04, 1.7575e-02, 9.2661e-01, 6.8919e-04, 4.9777e-01, 1.2133e-02,
             1.1415e-03, 1.6763e-03, 1.2275e-01, 7.3525e-02, 5.6224e-04, 2.1502e-01,
             1.7034e-03, 1.4736e-01]))



As for the single classification predictions, we get three things. The last one is the prediction of the model on each class (going from 0 to 1). The second to last cooresponds to a one-hot encoded targets (you get `True` for all predicted classes, the ones that get a probability > 0.5) and the first is the decoded, readable version.

And like before, we can check where the model did its worse:

```python
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>predicted</th>
      <th>probabilities</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chair;diningtable;person;pottedplant</td>
      <td>person</td>
      <td>tensor([1.3859e-07, 2.0057e-05, 1.3330e-05, 4.4968e-05, 2.4639e-05, 3.7735e-06,\n        8.4346e-04, 2.1878e-04, 1.1218e-03, 4.2916e-08, 1.4975e-03, 2.5902e-03,\n        2.4724e-05, 1.0664e-05, 9.9991e-01, 3.4751e-03, 1.5933e-05, 5.5496e-04,\n        3.3906e-05, 3.2141e-02])</td>
      <td>0.9498050808906555</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bottle;person</td>
      <td>person</td>
      <td>tensor([1.7861e-12, 2.4931e-09, 1.8502e-09, 1.2212e-09, 1.9618e-08, 1.3690e-11,\n        3.2873e-06, 5.8404e-08, 2.5140e-07, 2.7120e-14, 7.0966e-07, 6.9432e-04,\n        1.1541e-09, 1.0451e-09, 1.0000e+00, 6.0462e-07, 3.4618e-08, 4.7445e-06,\n        4.1124e-09, 9.9111e-04])</td>
      <td>0.8874256014823914</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bus;car;person</td>
      <td>person</td>
      <td>tensor([4.9122e-08, 5.5848e-06, 2.1049e-06, 8.9707e-06, 5.6819e-06, 1.3298e-05,\n        1.4959e-03, 5.9346e-05, 5.3715e-05, 2.1142e-09, 2.5905e-05, 6.1096e-03,\n        5.5608e-06, 3.3048e-06, 9.9968e-01, 5.6836e-05, 6.0220e-06, 2.0087e-04,\n        2.4676e-05, 3.3325e-03])</td>
      <td>0.8871580362319946</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bottle;person</td>
      <td>person</td>
      <td>tensor([5.2528e-13, 1.7562e-09, 3.2824e-10, 1.1594e-09, 2.1161e-08, 1.3134e-11,\n        2.4606e-06, 3.8796e-08, 8.5993e-07, 3.3807e-15, 1.6779e-06, 1.5134e-04,\n        1.5394e-10, 5.9556e-10, 1.0000e+00, 3.3567e-06, 9.9627e-09, 8.0715e-06,\n        2.5843e-09, 4.6214e-03])</td>
      <td>0.8837955594062805</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bottle;person</td>
      <td>person</td>
      <td>tensor([1.8547e-12, 2.8884e-09, 1.6598e-09, 2.4114e-09, 2.3059e-08, 6.5701e-11,\n        3.9855e-06, 4.8739e-08, 8.4072e-07, 1.9686e-14, 2.4176e-06, 3.3824e-04,\n        2.4855e-09, 1.2725e-09, 1.0000e+00, 1.8027e-06, 2.5171e-08, 6.1435e-06,\n        1.4883e-08, 3.2184e-03])</td>
      <td>0.8794390559196472</td>
    </tr>
    <tr>
      <th>5</th>
      <td>bottle;person</td>
      <td>person</td>
      <td>tensor([4.1999e-11, 1.2264e-08, 2.6133e-08, 2.0119e-08, 1.3637e-07, 6.2015e-10,\n        1.6232e-05, 1.7808e-07, 6.3830e-07, 1.1085e-12, 9.6746e-07, 4.1021e-04,\n        1.1263e-08, 1.4854e-08, 9.9999e-01, 2.1052e-06, 1.7178e-07, 8.3955e-06,\n        7.8824e-08, 1.1336e-03])</td>
      <td>0.7904742360115051</td>
    </tr>
    <tr>
      <th>6</th>
      <td>chair;diningtable;person</td>
      <td>person</td>
      <td>tensor([3.2536e-08, 5.8821e-06, 4.8512e-06, 1.5012e-05, 4.5857e-04, 1.0428e-06,\n        2.9736e-04, 4.4146e-05, 4.0318e-04, 3.7364e-09, 4.8184e-04, 1.9579e-03,\n        8.3174e-07, 3.4988e-06, 9.9994e-01, 1.4466e-03, 7.1675e-06, 4.5994e-04,\n        9.3516e-06, 2.8538e-02])</td>
      <td>0.774388313293457</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bus;person</td>
      <td>person</td>
      <td>tensor([3.3093e-08, 3.1500e-06, 5.7413e-06, 3.6701e-06, 2.1493e-05, 3.8484e-07,\n        2.8466e-04, 2.2827e-05, 9.6937e-05, 1.6873e-09, 6.1497e-05, 5.1155e-03,\n        1.3798e-06, 2.3416e-06, 9.9984e-01, 8.3490e-05, 6.5483e-06, 3.6999e-04,\n        5.8180e-06, 8.0087e-03])</td>
      <td>0.7392368316650391</td>
    </tr>
    <tr>
      <th>8</th>
      <td>bus;car</td>
      <td>train</td>
      <td>tensor([1.6565e-03, 4.2173e-04, 3.1245e-04, 1.5796e-03, 2.6684e-04, 9.8883e-03,\n        1.2766e-02, 1.8388e-04, 4.5448e-03, 6.0582e-04, 1.0346e-03, 5.2797e-04,\n        1.3014e-03, 1.6428e-03, 5.4353e-02, 5.3857e-04, 8.4819e-04, 3.2369e-04,\n        9.8810e-01, 6.4698e-04])</td>
      <td>0.674037516117096</td>
    </tr>
  </tbody>
</table>



![png](output_79_1.png)


### With the data block API

We can also use the data block API to get our data in a [`DataLoaders`](/data.core.html#DataLoaders). Like we said before, feel free to skip this part if you are not comfortable with learning new APIs just yet.

Remember how the data is structured in our dataframe:

```python
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
      <th>fname</th>
      <th>labels</th>
      <th>is_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000005.jpg</td>
      <td>chair</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000007.jpg</td>
      <td>car</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000009.jpg</td>
      <td>horse person</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000012.jpg</td>
      <td>car</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>000016.jpg</td>
      <td>bicycle</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



In this case we build the data block by providing:

- the types used: [`ImageBlock`](/vision.data.html#ImageBlock) and [`MultiCategoryBlock`](/data.block.html#MultiCategoryBlock).
- how to get the input items from our dataframe: here we read the column `fname` and need to add path/train/ at the beginning to get proper filenames.
- how to get the targets from our dataframe: here we read the column `labels` and need to split by space.
- how to split the items, here by using the column `is_valid`.
- the `item_tfms` and `batch_tfms` like before.

```python
pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter('is_valid'),
                   get_x=ColReader('fname', pref=str(path/'train') + os.path.sep),
                   get_y=ColReader('labels', label_delim=' '),
                   item_tfms = Resize(460),
                   batch_tfms=aug_transforms(size=224))
```

This block is slightly different than before: we don't need to pass a function to gather all our items as the dataframe we will give already has them all. However, we do need to preprocess the row of that dataframe to get out inputs, which is why we pass a `get_x`. It defaults to the fastai function `noop`, which is why we didn't need to pass it along before.

Like before, `pascal` is just a blueprint. We need to pass it the source of our data to be able to get [`DataLoaders`](/data.core.html#DataLoaders):

```python
dls = pascal.dataloaders(df)
```

Then we can look at some of our pictures with `dls.show_batch()`

```python
dls.show_batch(max_n=9)
```


![png](output_88_0.png)


## Segmentation

Segmentation is a problem where we have to predict a category for each pixel of the image. For this task, we will use the [Camvid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), a dataset of screenshots from cameras in cars. Each pixel of the image has a label such as "road", "car" or "pedestrian".

As usual, we can download the data with our [`untar_data`](/data.external.html#untar_data) function.

```python
path = untar_data(URLs.CAMVID_TINY)
path.ls()
```




    (#5) [Path('/home/sgugger/.fastai/data/camvid_tiny/codes.txt'),Path('/home/sgugger/.fastai/data/camvid_tiny/labels'),Path('/home/sgugger/.fastai/data/camvid_tiny/models'),Path('/home/sgugger/.fastai/data/camvid_tiny/export.pkl'),Path('/home/sgugger/.fastai/data/camvid_tiny/images')]



The `images` folder contains the images, and the corresponding segmentation masks of labels are in the `labels` folder. The `codes` file contains the corresponding integer to class (the masks have an int value for each pixel). 

```python
codes = np.loadtxt(path/'codes.txt', dtype=str)
codes
```




    array(['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car',
           'CartLuggagePram', 'Child', 'Column_Pole', 'Fence', 'LaneMkgsDriv',
           'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving',
           'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk',
           'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone',
           'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel',
           'VegetationMisc', 'Void', 'Wall'], dtype='<U17')



### Using the high-level API

As before, the [`get_image_files`](/data.transforms.html#get_image_files) function helps us grab all the image filenames:

```python
fnames = get_image_files(path/"images")
fnames[0]
```




    Path('/home/sgugger/.fastai/data/camvid_tiny/images/0016E5_05310.png')



Let's have a look in the labels folder:

```python
(path/"labels").ls()[0]
```




    Path('/home/sgugger/.fastai/data/camvid_tiny/labels/0016E5_00840_P.png')



It seems the segmentation masks have the same base names as the images but with an extra `_P`, so we can define a label function: 

```python
def label_func(fn): return path/"labels"/f"{fn.stem}_P{fn.suffix}"
```

We can then gather our data using [`SegmentationDataLoaders`](/vision.data.html#SegmentationDataLoaders):

```python
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = fnames, label_func = label_func, codes = codes
)
```

We do not need to pass `item_tfms` to resize our images here because they already are all of the same size.

As usual, we can have a look at our data with the `show_batch` method. In this instance, the fastai library is superimposing the masks with one specific color per pixel:

```python
dls.show_batch(max_n=6)
```


![png](output_104_0.png)


A traditional CNN won't work for segmentation, we have to use a special kind of model called a UNet, so we use [`unet_learner`](/vision.learner.html#unet_learner) to define our [`Learner`](/learner.html#Learner):

```python
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
      <td>3.227327</td>
      <td>2.570531</td>
      <td>00:05</td>
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
      <td>1.845436</td>
      <td>1.761499</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.595670</td>
      <td>1.565473</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.409528</td>
      <td>1.209965</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.283140</td>
      <td>1.103817</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.147777</td>
      <td>0.933692</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.030369</td>
      <td>0.924723</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.932455</td>
      <td>0.882634</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.855784</td>
      <td>0.881755</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>


And as before, we can get some idea of the predicted results with `show_results`

```python
learn.show_results(max_n=6, figsize=(7,8))
```






![png](output_108_1.png)


### With the data block API

We can also use the data block API to get our data in a [`DataLoaders`](/data.core.html#DataLoaders). Like it's been said before, feel free to skip this part if you are not comfortable with learning new APIs just yet.

In this case we build the data block by providing:

- the types used: [`ImageBlock`](/vision.data.html#ImageBlock) and [`MaskBlock`](/vision.data.html#MaskBlock). We provide the `codes` to [`MaskBlock`](/vision.data.html#MaskBlock) as there is no way to guess them from the data.
- how to gather our items, here by using [`get_image_files`](/data.transforms.html#get_image_files).
- how to get the targets from our items: by using `label_func`.
- how to split the items, here randomly.
- `batch_tfms` for data augmentation.

```python
camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items = get_image_files,
                   get_y = label_func,
                   splitter=RandomSplitter(),
                   batch_tfms=aug_transforms(size=(120,160)))
```

```python
dls = camvid.dataloaders(path/"images", path=path, bs=8)
```

```python
dls.show_batch(max_n=6)
```


![png](output_113_0.png)


## Points

This section uses the data block API, so if you skipped it before, we recommend you skip this section as well.

We will now look at a task where we want to predict points in a picture. For this, we will use the [Biwi Kinect Head Pose Dataset](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#db). First thing first, let's begin by downloading the dataset as usual.

```python
path = untar_data(URLs.BIWI_HEAD_POSE)
```

Let's see what we've got!

```python
path.ls()
```




    (#50) [Path('/home/sgugger/.fastai/data/biwi_head_pose/01.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/18.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/04'),Path('/home/sgugger/.fastai/data/biwi_head_pose/10.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/24'),Path('/home/sgugger/.fastai/data/biwi_head_pose/14.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/20.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/11.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/02.obj'),Path('/home/sgugger/.fastai/data/biwi_head_pose/07')...]



There are 24 directories numbered from 01 to 24 (they correspond to the different persons photographed) and a corresponding .obj file (we won't need them here). We'll take a look inside one of these directories:

```python
(path/'01').ls()
```




    (#1000) [Path('01/frame_00087_pose.txt'),Path('01/frame_00079_pose.txt'),Path('01/frame_00114_pose.txt'),Path('01/frame_00084_rgb.jpg'),Path('01/frame_00433_pose.txt'),Path('01/frame_00323_rgb.jpg'),Path('01/frame_00428_rgb.jpg'),Path('01/frame_00373_pose.txt'),Path('01/frame_00188_rgb.jpg'),Path('01/frame_00354_rgb.jpg')...]



Inside the subdirectories, we have different frames, each of them come with an image (`\_rgb.jpg`) and a pose file (`\_pose.txt`). We can easily get all the image files recursively with [`get_image_files`](/data.transforms.html#get_image_files), then write a function that converts an image filename to its associated pose file.

```python
img_files = get_image_files(path)
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')
img2pose(img_files[0])
```




    Path('04/frame_00084_pose.txt')



We can have a look at our first image:

```python
im = PILImage.create(img_files[0])
im.shape
```




    (480, 640)



```python
im.to_thumb(160)
```




![png](output_125_0.png)



The Biwi dataset web site explains the format of the pose text file associated with each image, which shows the location of the center of the head. The details of this aren't important for our purposes, so we'll just show the function we use to extract the head center point:

```python
cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1,c2])
```

This function returns the coordinates as a tensor of two items:

```python
get_ctr(img_files[0])
```




    tensor([372.4046, 245.8602])



We can pass this function to [`DataBlock`](/data.block.html#DataBlock) as `get_y`, since it is responsible for labeling each item. We'll resize the images to half their input size, just to speed up training a bit.

One important point to note is that we should not just use a random splitter. The reason for this is that the same person appears in multiple images in this dataset — but we want to ensure that our model can generalise to people that it hasn't seen yet. Each folder in the dataset contains the images for one person. Therefore, we can create a splitter function which returns true for just one person, resulting in a validation set containing just that person's images.

The only other difference to previous data block examples is that the second block is a [`PointBlock`](/vision.data.html#PointBlock). This is necessary so that fastai knows that the labels represent coordinates; that way, it knows that when doing data augmentation, it should do the same augmentation to these coordinates as it does to the images.

```python
biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name=='13'),
    batch_tfms=[*aug_transforms(size=(240,320)), 
                Normalize.from_stats(*imagenet_stats)]
)
```

```python
dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8,6))
```


![png](output_132_0.png)


Now that we have assembled our data, we can use the rest of the fastai API as usual. [`cnn_learner`](/vision.learner.html#cnn_learner) works perfectly in this case, and the library will infer the proper loss function from the data:

```python
learn = cnn_learner(dls, resnet18, y_range=(-1,1))
```

```python
learn.lr_find()
```








    (0.005754399299621582, 3.6307804407442745e-07)




![png](output_135_2.png)


Then we can train our model:

```python
learn.fine_tune(4, 5e-3)
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
      <td>0.057434</td>
      <td>0.002171</td>
      <td>00:31</td>
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
      <td>0.005320</td>
      <td>0.005426</td>
      <td>00:39</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.003624</td>
      <td>0.000698</td>
      <td>00:39</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.002163</td>
      <td>0.000099</td>
      <td>00:39</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.001325</td>
      <td>0.000233</td>
      <td>00:39</td>
    </tr>
  </tbody>
</table>


The loss is the mean squared error, so that means we make on average an error of 

```python
math.sqrt(0.0001)
```




    0.01



percent when predicting our points! And we can look at those results as usual:

```python
learn.show_results()
```






![png](output_141_1.png)

