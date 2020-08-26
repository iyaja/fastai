# Data block tutorial
> Using the data block across all applications


In this tutorial, we'll see how to use the data block API on a variety of tasks and how to debug data blocks. The data block API takes its name from the way it's designed: every bit needed to build the [`DataLoaders`](/data.core.html#DataLoaders) object (type of inputs, targets, how to label, split...) is encapsulated in a block, and you can mix and match those blocks 

## Building a [`DataBlock`](/data.block.html#DataBlock) from scratch

The rest of this tutorial will give many examples, but let's first build a [`DataBlock`](/data.block.html#DataBlock) from scratch on the dogs versus cats problem we saw in the [vision tutorial](http://docs.fast.ai/tutorial.vision). First we import everything needed in vision.

```python
from fastai.data.all import *
from fastai.vision.all import *
```

The first step is to download and decompress our data (if it's not already done) and get its location:

```python
path = untar_data(URLs.PETS)
```

And as we saw, all the filenames are in the "images" folder. The [`get_image_files`](/data.transforms.html#get_image_files) function helps get all the images in subfolders:

```python
fnames = get_image_files(path/"images")
```

Let's begin with an empty [`DataBlock`](/data.block.html#DataBlock).

```python
dblock = DataBlock()
```

By itself, a [`DataBlock`](/data.block.html#DataBlock) is just a blue print on how to assemble your data. It does not do anything until you pass it a source. You can choose to then convert that source into a [`Datasets`](/data.core.html#Datasets) or a [`DataLoaders`](/data.core.html#DataLoaders) by using the [`DataBlock.datasets`](/data.block.html#DataBlock.datasets) or [`DataBlock.dataloaders`](/data.block.html#DataBlock.dataloaders) method. Since we haven't done anything to get our data ready for batches, the `dataloaders` method will fail here, but we can have a look at how it gets converted in [`Datasets`](/data.core.html#Datasets). This is where we pass the source of our data, here all our filenames:

```python
dsets = dblock.datasets(fnames)
dsets.train[0]
```




    (Path('/home/jhoward/.fastai/data/oxford-iiit-pet/images/Maine_Coon_91.jpg'),
     Path('/home/jhoward/.fastai/data/oxford-iiit-pet/images/Maine_Coon_91.jpg'))



By default, the data block API assumes we have an input and a target, which is why we see our filename repeated twice. 

The first thing we can do is use a `get_items` function to actually assemble our items inside the data block:

```python
dblock = DataBlock(get_items = get_image_files)
```

The difference is that you then pass as a source the folder with the images and not all the filenames:

```python
dsets = dblock.datasets(path/"images")
dsets.train[0]
```




    (Path('/home/jhoward/.fastai/data/oxford-iiit-pet/images/Persian_76.jpg'),
     Path('/home/jhoward/.fastai/data/oxford-iiit-pet/images/Persian_76.jpg'))



Our inputs are ready to be processed as images (since images can be built from filenames), but our target is not. Since we are in a cat versus dog problem, we need to convert that filename to "cat" vs "dog" (or `True` vs `False`). Let's build a function for this:

```python
def label_func(fname):
    return "cat" if fname.name[0].isupper() else "dog"
```

We can then tell our data block to use it to label our target by passing it as `get_y`:

```python
dblock = DataBlock(get_items = get_image_files,
                   get_y     = label_func)

dsets = dblock.datasets(path/"images")
dsets.train[0]
```




    (Path('/home/jhoward/.fastai/data/oxford-iiit-pet/images/pug_160.jpg'), 'dog')



Now that our inputs and targets are ready, we can specify types to tell the data block API that our inputs are images and our targets are categories. Types are represented by blocks in the data block API, here we use [`ImageBlock`](/vision.data.html#ImageBlock) and [`CategoryBlock`](/data.block.html#CategoryBlock):

```python
dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func)

dsets = dblock.datasets(path/"images")
dsets.train[0]
```




    (PILImage mode=RGB size=500x375, TensorCategory(1))



We can see how the [`DataBlock`](/data.block.html#DataBlock) automatically added the transforms necessary to open the image, or how it changed the name "cat" to an index (with a special tensor type). To do this, it created a mapping from categories to index called "vocab" that we can access this way:

```python
dsets.vocab
```




    (#2) ['cat','dog']



Note that you can mix and match any block for input and targets, which is why the API is named data block API. You can also have more than two blocks (if you have multiple inputs and/or targets), you would just need to pass `n_inp` to the [`DataBlock`](/data.block.html#DataBlock) to tell the library how many inputs there are (the rest would be targets) and pass a list of functions to `get_x` and/or `get_y` (to explain how to process each item to be ready for his type). See the object detection below for such an example.

The next step is to control how our validation set is created. We do this by passing a `splitter` to [`DataBlock`](/data.block.html#DataBlock). For instance, here is how to do a random split.

```python
dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func,
                   splitter  = RandomSplitter())

dsets = dblock.datasets(path/"images")
dsets.train[0]
```




    (PILImage mode=RGB size=500x335, TensorCategory(0))



The last step is to specify item transforms and batch transforms (the same way we do it in [`ImageDataLoaders`](/vision.data.html#ImageDataLoaders) factory methods):

```python
dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func,
                   splitter  = RandomSplitter(),
                   item_tfms = Resize(224))
```

With that resize, we are now able to batch items together and can finally call `dataloaders` to convert our [`DataBlock`](/data.block.html#DataBlock) to a [`DataLoaders`](/data.core.html#DataLoaders) object:

```python
dls = dblock.dataloaders(path/"images")
dls.show_batch()
```


![png](output_30_0.png)


The way we usually build the data block in one go is by answering a list of questions:

- what is the types of your inputs/targets? Here images and categories
- where is your data? Here in filenames in subfolders
- does something need to be applied to inputs? Here no
- does something need to be applied to the target? Here the `label_func` function
- how to split the data? Here randomly
- do we need to apply something on formed items? Here a resize
- do we need to apply something on formed batches? Here no

This gives us this design:

```python
dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func,
                   splitter  = RandomSplitter(),
                   item_tfms = Resize(224))
```

For two questions that got a no, the corresponding arguments we would pass if the anwser was different would be `get_x` and `batch_tfms`.

## Image classification

Let's begin with examples of image classification problems. There are two kinds of image classification problems: problems with single-label (each image has one given label) or multi-label (each image can have multiple or no labels at all). We will cover those two kinds here.

```python
from fastai.vision.all import *
```

### MNIST (single label)

[MNIST](http://yann.lecun.com/exdb/mnist/) is a dataset of hand-written digits from 0 to 9. We can very easily load it in the data block API by answering the following questions:

- what are the types of our inputs and targets? Black and white images and labels.
- where is the data? In subfolders.
- how do we know if a sample is in the training or the validation set? By looking at the grandparent folder.
- how do we know the label of an image? By looking at the parent folder.

In terms of the API, those answers translate like this:

```python
mnist = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
                  get_items=get_image_files, 
                  splitter=GrandparentSplitter(),
                  get_y=parent_label)
```

Our types become blocks: one for images (using the black and white [`PILImageBW`](/vision.core.html#PILImageBW) class) and one for categories. Searching subfolder for all image filenames is done by the [`get_image_files`](/data.transforms.html#get_image_files) function. The split training/validation is done by using a [`GrandparentSplitter`](/data.transforms.html#GrandparentSplitter). And the function to get our targets (often called `y`) is [`parent_label`](/data.transforms.html#parent_label).

To get an idea of the objects the fastai library provides for reading, labelling or splitting, check the [`data.transforms`](/data.transforms.html) module.

In itself, a data block is just a blueprint. It does not do anything and does not check for errors. You have to feed it the source of the data to actually gather something. This is done with the `.dataloaders` method:

```python
dls = mnist.dataloaders(untar_data(URLs.MNIST_TINY))
dls.show_batch(max_n=9, figsize=(4,4))
```


![png](output_41_0.png)


If something went wrong in the previous step, or if you're just curious about what happened under the hood, use the `summary` method. It will go verbosely step by step, and you will see at which point the process failed.

```python
mnist.summary(untar_data(URLs.MNIST_TINY))
```

    Setting-up type transforms pipelines
    Collecting items from /home/jhoward/.fastai/data/mnist_tiny
    Found 1428 items
    2 datasets of sizes 709,699
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: parent_label -> Categorize
    
    Building one sample
      Pipeline: PILBase.create
        starting from
          /home/jhoward/.fastai/data/mnist_tiny/train/7/723.png
        applying PILBase.create gives
          PILImageBW mode=L size=28x28
      Pipeline: parent_label -> Categorize
        starting from
          /home/jhoward/.fastai/data/mnist_tiny/train/7/723.png
        applying parent_label gives
          7
        applying Categorize gives
          TensorCategory(1)
    
    Final sample: (PILImageBW mode=L size=28x28, TensorCategory(1))
    
    
    Setting up after_item: Pipeline: ToTensor
    Setting up before_batch: Pipeline: 
    Setting up after_batch: Pipeline: IntToFloatTensor
    
    Building one batch
    Applying item_tfms to the first sample:
      Pipeline: ToTensor
        starting from
          (PILImageBW mode=L size=28x28, TensorCategory(1))
        applying ToTensor gives
          (TensorImageBW of size 1x28x28, TensorCategory(1))
    
    Adding the next 3 samples
    
    No before_batch transform to apply
    
    Collating items in a batch
    
    Applying batch_tfms to the batch built
      Pipeline: IntToFloatTensor
        starting from
          (TensorImageBW of size 4x1x28x28, TensorCategory([1, 1, 1, 1], device='cuda:5'))
        applying IntToFloatTensor gives
          (TensorImageBW of size 4x1x28x28, TensorCategory([1, 1, 1, 1], device='cuda:5'))


Let's go over another example!

### Pets (single label)

The [Oxford IIIT Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) is a dataset of pictures of dogs and cats, with 37 different breeds. A slight (but very) important difference with MNIST is that images are now not all of the same size. In MNIST they were all 28 by 28 pixels, but here they have different aspect ratios or dimensions. Therefore, we will need to add something to make them all the same size to be able to assemble them together in a batch. We will also see how to add data augmentation.

So let's go over the same questions as before and add two more:

- what are the types of our inputs and targets? Images and labels.
- where is the data? In subfolders.
- how do we know if a sample is in the training or the validation set? We'll take a random split.
- how do we know the label of an image? By looking at the parent folder.
- do we want to apply a function to a given sample? Yes, we need to resize everything to a given size.
- do we want to apply a function to a batch after it's created? Yes, we want data augmentation.

```python
pets = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                 get_items=get_image_files, 
                 splitter=RandomSplitter(),
                 get_y=Pipeline([attrgetter("name"), RegexLabeller(pat = r'^(.*)_\d+.jpg$')]),
                 item_tfms=Resize(128),
                 batch_tfms=aug_transforms())
```

And like for MNIST, we can see how the answers to those questions directly translate in the API. Our types become blocks: one for images and one for categories. Searching subfolder for all image filenames is done by the [`get_image_files`](/data.transforms.html#get_image_files) function. The split training/validation is done by using a [`RandomSplitter`](/data.transforms.html#RandomSplitter). The function to get our targets (often called `y`) is a composition of two transforms: we get the name attribute of our `Path` filenames, then apply a regular expression to get the class. To compose those two transforms into one, we use a [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline).

Finally, We apply a resize at the item level and `aug_transforms()` at the batch level.

```python
dls = pets.dataloaders(untar_data(URLs.PETS)/"images")
dls.show_batch(max_n=9)
```


![png](output_49_0.png)


Now let's see how we can use the same API for a multi-label problem.

### Pascal (multi-label)

The [Pascal dataset](http://host.robots.ox.ac.uk/pascal/VOC/) is originally an object detection dataset (we have to predict where some objects are in pictures). But it contains lots of pictures with various objects in them, so it gives a great example for a multi-label problem. Let's download it and have a look at the data:

```python
pascal_source = untar_data(URLs.PASCAL_2007)
df = pd.read_csv(pascal_source/"train.csv")
```

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



So it looks like we have one column with filenames, one column with the labels (separated by space) and one column that tells us if the filename should go in the validation set or not.

There are multiple ways to put this in a [`DataBlock`](/data.block.html#DataBlock), let's go over them, but first, let's answer our usual questionnaire:

- what are the types of our inputs and targets? Images and multiple labels.
- where is the data? In a dataframe.
- how do we know if a sample is in the training or the validation set? A column of our dataframe.
- how do we get an image? By looking at the column fname.
- how do we know the label of an image? By looking at the column labels.
- do we want to apply a function to a given sample? Yes, we need to resize everything to a given size.
- do we want to apply a function to a batch after it's created? Yes, we want data augmentation.

Notice how there is one more question compared to before: we wont have to use a `get_items` function here because we already have all our data in one place. But we will need to do something to the raw dataframe to get our inputs, read the first column and add the proper folder before the filename. This is what we pass as `get_x`.

```python
pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter(),
                   get_x=ColReader(0, pref=pascal_source/"train"),
                   get_y=ColReader(1, label_delim=' '),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms())
```

Again, we can see how the answers to the questions directly translate in the API. Our types become blocks: one for images and one for multi-categories. The split is done by a [`ColSplitter`](/data.transforms.html#ColSplitter) (it defaults to the column named `is_valid`). The function to get our inputs (often called `x`) is a [`ColReader`](/data.transforms.html#ColReader) on the first column with a prefix, the function to get our targets (often called `y`) is [`ColReader`](/data.transforms.html#ColReader) on the second column, with a space delimiter. We apply a resize at the item level and `aug_transforms()` at the batch level.

```python
dls = pascal.dataloaders(df)
dls.show_batch()
```


![png](output_58_0.png)


Another way to do this is by directly using functions for `get_x` and `get_y`:

```python
pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter(),
                   get_x=lambda x:pascal_source/"train"/f'{x[0]}',
                   get_y=lambda x:x[1].split(' '),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms())

dls = pascal.dataloaders(df)
dls.show_batch()
```


![png](output_60_0.png)


Alternatively, we can use the names of the columns as attributes (since rows of a dataframe are pandas series).

```python
pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter(),
                   get_x=lambda o:f'{pascal_source}/train/'+o.fname,
                   get_y=lambda o:o.labels.split(),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms())

dls = pascal.dataloaders(df)
dls.show_batch()
```


![png](output_62_0.png)


The most efficient way (to avoid iterating over the rows of the dataframe, which can take a long time) is to use the `from_columns` method. It will use `get_items` to convert the columns into numpy arrays. The drawback is that since we lose the dataframe after extracting the relevant columns, we can't use a [`ColSplitter`](/data.transforms.html#ColSplitter) anymore. Here we used an [`IndexSplitter`](/data.transforms.html#IndexSplitter) after manually extracting the index of the validation set from the dataframe:

```python
def _pascal_items(x): return (
    f'{pascal_source}/train/'+x.fname, x.labels.str.split())
valid_idx = df[df['is_valid']].index.values

pascal = DataBlock.from_columns(blocks=(ImageBlock, MultiCategoryBlock),
                   get_items=_pascal_items,
                   splitter=IndexSplitter(valid_idx),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms())
```

```python
dls = pascal.dataloaders(df)
dls.show_batch()
```


![png](output_65_0.png)


## Image localization

There are various problems that fall in the image localization category: image segmentation (which is a task where you have to predict the class of each pixel of an image), coordinate predictions (predict one or several key points on an image) and object detection (draw a box around objects to detect).

Let's see an example of each of those and how to use the data block API in each case.

### Segmentation

We will use a small subset of the [CamVid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) for our example.

```python
path = untar_data(URLs.CAMVID_TINY)
```

Let's go over our usual questionnaire:

- what are the types of our inputs and targets? Images and segmentation masks.
- where is the data? In subfolders.
- how do we know if a sample is in the training or the validation set? We'll take a random split.
- how do we know the label of an image? By looking at a corresponding file in the "labels" folder.
- do we want to apply a function to a batch after it's created? Yes, we want data augmentation.

```python
camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes = np.loadtxt(path/'codes.txt', dtype=str))),
    get_items=get_image_files,
    splitter=RandomSplitter(),
    get_y=lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    batch_tfms=aug_transforms())
```

The [`MaskBlock`](/vision.data.html#MaskBlock) is generated with the `codes` that give the correpondence between pixel value of the masks and the object they correspond to (like car, road, pedestrian...). The rest should look pretty familiar by now.

```python
dls = camvid.dataloaders(path/"images")
dls.show_batch()
```


![png](output_74_0.png)


### Points

For this example we will use a small sample of the [BiWi kinect head pose dataset](https://www.kaggle.com/kmader/biwi-kinect-head-pose-database). It contains pictures of people and the task is to predict where the center of their head is. We have saved this small dataet with a dictionary filename to center:

```python
biwi_source = untar_data(URLs.BIWI_SAMPLE)
fn2ctr = (biwi_source/'centers.pkl').load()
```

Then we can go over our usual questions:

- what are the types of our inputs and targets? Images and points.
- where is the data? In subfolders.
- how do we know if a sample is in the training or the validation set? We'll take a random split.
- how do we know the label of an image? By using the `fn2ctr` dictionary.
- do we want to apply a function to a batch after it's created? Yes, we want data augmentation.

```python
biwi = DataBlock(blocks=(ImageBlock, PointBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=lambda o:fn2ctr[o.name].flip(0),
                 batch_tfms=aug_transforms())
```

And we can use it to create a [`DataLoaders`](/data.core.html#DataLoaders):

```python
dls = biwi.dataloaders(biwi_source)
dls.show_batch(max_n=9)
```


![png](output_81_0.png)


### Bounding boxes

For this task, we will use a small subset of the [COCO dataset](http://cocodataset.org/#home). It contains pictures with day-to-day objects and the goal is to predict where the objects are by drawing a rectangle around them. 

The fastai library comes with a function called [`get_annotations`](/vision.core.html#get_annotations) that will interpret the content of `train.json` and give us a dictionary filename to (bounding boxes, labels).

```python
coco_source = untar_data(URLs.COCO_TINY)
images, lbl_bbox = get_annotations(coco_source/'train.json')
img2bbox = dict(zip(images, lbl_bbox))
```

Then we can go over our usual questions:

- what are the types of our inputs and targets? Images and bounding boxes.
- where is the data? In subfolders.
- how do we know if a sample is in the training or the validation set? We'll take a random split.
- how do we know the label of an image? By using the `img2bbox` dictionary.
- do we want to apply a function to a given sample? Yes, we need to resize everything to a given size.
- do we want to apply a function to a batch after it's created? Yes, we want data augmentation.

```python
coco = DataBlock(blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=[lambda o: img2bbox[o.name][0], lambda o: img2bbox[o.name][1]], 
                 item_tfms=Resize(128),
                 batch_tfms=aug_transforms(),
                 n_inp=1)
```

Note that we provide three types, because we have two targets: the bounding boxes and the labels. That's why we pass `n_inp=1` at the end, to tell the library where the inputs stop and the targets begin.

This is also why we pass a list to `get_y`: since we have two targets, we must tell the library how to label for each of them (you can use `noop` if you don't want to do anything for one).

```python
dls = coco.dataloaders(coco_source)
dls.show_batch(max_n=9)
```


![png](output_88_0.png)


## Text

We will show two examples: language modeling and text classification. Note that with the data block API, you can adapt the example before for multi-label to a problem where the inputs are texts.

```python
from fastai.text.all import *
```

### Language model

We will use a dataset compose of movie reviews from IMDb. As usual, we can download it in one line of code with [`untar_data`](/data.external.html#untar_data).

```python
path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/'texts.csv')
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
      <th>label</th>
      <th>text</th>
      <th>is_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>negative</td>
      <td>Un-bleeping-believable! Meg Ryan doesn't even look her usual pert lovable self in this, which normally makes me forgive her shallow ticky acting schtick. Hard to believe she was the producer on this dog. Plus Kevin Kline: what kind of suicide trip has his career been on? Whoosh... Banzai!!! Finally this was directed by the guy who did Big Chill? Must be a replay of Jonestown - hollywood style. Wooofff!</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>positive</td>
      <td>This is a extremely well-made film. The acting, script and camera-work are all first-rate. The music is good, too, though it is mostly early in the film, when things are still relatively cheery. There are no really superstars in the cast, though several faces will be familiar. The entire cast does an excellent job with the script.&lt;br /&gt;&lt;br /&gt;But it is hard to watch, because there is no good end to a situation like the one presented. It is now fashionable to blame the British for setting Hindus and Muslims against each other, and then cruelly separating them into two countries. There is som...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>negative</td>
      <td>Every once in a long while a movie will come along that will be so awful that I feel compelled to warn people. If I labor all my days and I can save but one soul from watching this movie, how great will be my joy.&lt;br /&gt;&lt;br /&gt;Where to begin my discussion of pain. For starters, there was a musical montage every five minutes. There was no character development. Every character was a stereotype. We had swearing guy, fat guy who eats donuts, goofy foreign guy, etc. The script felt as if it were being written as the movie was being shot. The production value was so incredibly low that it felt li...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>positive</td>
      <td>Name just says it all. I watched this movie with my dad when it came out and having served in Korea he had great admiration for the man. The disappointing thing about this film is that it only concentrate on a short period of the man's life - interestingly enough the man's entire life would have made such an epic bio-pic that it is staggering to imagine the cost for production.&lt;br /&gt;&lt;br /&gt;Some posters elude to the flawed characteristics about the man, which are cheap shots. The theme of the movie "Duty, Honor, Country" are not just mere words blathered from the lips of a high-brassed offic...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>negative</td>
      <td>This movie succeeds at being one of the most unique movies you've seen. However this comes from the fact that you can't make heads or tails of this mess. It almost seems as a series of challenges set up to determine whether or not you are willing to walk out of the movie and give up the money you just paid. If you don't want to feel slighted you'll sit through this horrible film and develop a real sense of pity for the actors involved, they've all seen better days, but then you realize they actually got paid quite a bit of money to do this and you'll lose pity for them just like you've alr...</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



We can see it's composed of (pretty long!) reviews labeled positive or negative. Let's go over our usual questions:

- what are the types of our inputs and targets? Texts and we don't really have targets, since the targets is derived from the inputs.
- where is the data? In a dataframe.
- how do we know if a sample is in the training or the validation set? We have an `is_valid` column.
- how do we get our inputs? In the `text` column.

```python
imdb_lm = DataBlock(blocks=TextBlock.from_df('text', is_lm=True),
                    get_x=ColReader('text'),
                    splitter=ColSplitter())
```

Since there are no targets here, we only have one block to specify. [`TextBlock`](/text.data.html#TextBlock)s are a bit special compared to other [`TransformBlock`](/data.block.html#TransformBlock)s: to be able to efficiently tokenize all texts during setup, you need to use the class methods `from_folder` or `from_df`.

We can then get our data into [`DataLoaders`](/data.core.html#DataLoaders) by passing the dataframe to the `dataloaders` method:

```python
dls = imdb_lm.dataloaders(df, bs=64, seq_len=72)
dls.show_batch(max_n=6)
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>text_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos xxmaj if this is someone 's " favorite " movie , they need some serious help . xxmaj there is nothing funny or clever about this xxunk . i have n't seen the original movie this is the remake of ( some 1950s film ) , but it simply has to be better than this newer xxunk . \n\n a major gets kicked out of the military for being a xxunk</td>
      <td>xxmaj if this is someone 's " favorite " movie , they need some serious help . xxmaj there is nothing funny or clever about this xxunk . i have n't seen the original movie this is the remake of ( some 1950s film ) , but it simply has to be better than this newer xxunk . \n\n a major gets kicked out of the military for being a xxunk element</td>
    </tr>
    <tr>
      <th>1</th>
      <td>( in all fields ) , desperate to grab onto any " loser " attention he can for himself . xxmaj he is to be xxunk . xxbos xxmaj arnold once again in the 80 's demonstrated that he was the king of action and one liners in this futuristic film about a violent game show that no xxunk survives . xxmaj but as the tag line says xxmaj arnold has yet</td>
      <td>in all fields ) , desperate to grab onto any " loser " attention he can for himself . xxmaj he is to be xxunk . xxbos xxmaj arnold once again in the 80 's demonstrated that he was the king of action and one liners in this futuristic film about a violent game show that no xxunk survives . xxmaj but as the tag line says xxmaj arnold has yet to</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxmaj xxunk , meets up with xxmaj om xxmaj xxunk ( from whom he ran away some 30 years ago and xxunk to again ) and all xxmaj om xxmaj xxunk finds to say is to xxunk of his friendship with xxmaj xxunk xxrep 3 ! xxmaj what a load of crap . xxmaj seriously . xxmaj not to mention the b xxrep 3 a d soundtrack . xxmaj whatever happened to</td>
      <td>xxunk , meets up with xxmaj om xxmaj xxunk ( from whom he ran away some 30 years ago and xxunk to again ) and all xxmaj om xxmaj xxunk finds to say is to xxunk of his friendship with xxmaj xxunk xxrep 3 ! xxmaj what a load of crap . xxmaj seriously . xxmaj not to mention the b xxrep 3 a d soundtrack . xxmaj whatever happened to xxmaj</td>
    </tr>
    <tr>
      <th>3</th>
      <td>on more as she brings him to her cabin . \n\n xxmaj what little romance , sex , or for that matter , anything at all this film has besides bitter xxunk is hardly enough to justify the price of a rental unless you are one of those who love dramas where nothing interesting happens at all . xxmaj yes , the ending is very nicely done , but it is xxunk</td>
      <td>more as she brings him to her cabin . \n\n xxmaj what little romance , sex , or for that matter , anything at all this film has besides bitter xxunk is hardly enough to justify the price of a rental unless you are one of those who love dramas where nothing interesting happens at all . xxmaj yes , the ending is very nicely done , but it is xxunk reward</td>
    </tr>
    <tr>
      <th>4</th>
      <td>of the night before kicking in . \n\n xxmaj this is another of those films where there 's no ' plot ' to follow , as such , just a real life feel of these hopeless lives carrying on from one day to the next . xxmaj it 's been acclaimed by many ( including the xxmaj xxunk ! ) but it really was just too grim and bleak for me .</td>
      <td>the night before kicking in . \n\n xxmaj this is another of those films where there 's no ' plot ' to follow , as such , just a real life feel of these hopeless lives carrying on from one day to the next . xxmaj it 's been acclaimed by many ( including the xxmaj xxunk ! ) but it really was just too grim and bleak for me . i</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cry xxmaj freedom " is a must - see movie for it 's portrayal and story of xxmaj steve xxmaj biko . xxmaj it 's also a xxunk and devastating portrayal of a beautiful land divided and in the xxunk grips of racial xxunk and violence . xxbos xxmaj from a plot and movement standpoint , this movie was terrible . i found myself looking at the clock in theater hoping it</td>
      <td>xxmaj freedom " is a must - see movie for it 's portrayal and story of xxmaj steve xxmaj biko . xxmaj it 's also a xxunk and devastating portrayal of a beautiful land divided and in the xxunk grips of racial xxunk and violence . xxbos xxmaj from a plot and movement standpoint , this movie was terrible . i found myself looking at the clock in theater hoping it would</td>
    </tr>
  </tbody>
</table>


### Text classification

For the text classification, let's go over our usual questions:

- what are the types of our inputs and targets? Texts and categories.
- where is the data? In a dataframe.
- how do we know if a sample is in the training or the validation set? We have an `is_valid` column.
- how do we get our inputs? In the `text` column.
- how do we get our targets? In the `label` clolumn.

```python
imdb_clas = DataBlock(blocks=(TextBlock.from_df('text', seq_len=72, vocab=dls.vocab), CategoryBlock),
                      get_x=ColReader('text'),
                      get_y=ColReader('label'),
                      splitter=ColSplitter())
```

Like in the previous example, we use a class method to build a [`TextBlock`](/text.data.html#TextBlock). We can pass it the vocabulary of our language model (very useful for the ULMFit approach). We also show the `seq_len` argument (which defaults to 72) just because you need to make sure to use the same here and also in your [`text_classifier_learner`](/text.learner.html#text_classifier_learner).

{% include warning.html content='You need to make sure to use the same `seq_len` in [`TextBlock`](/text.data.html#TextBlock) and the [`Learner`](/learner.html#Learner) you will define later on.' %}

```python
dls = imdb_clas.dataloaders(df, bs=64)
dls.show_batch()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos xxmaj raising xxmaj victor xxmaj vargas : a xxmaj review \n\n xxmaj you know , xxmaj raising xxmaj victor xxmaj vargas is like sticking your hands into a big , xxunk bowl of xxunk . xxmaj it 's warm and gooey , but you 're not sure if it feels right . xxmaj try as i might , no matter how warm and gooey xxmaj raising xxmaj victor xxmaj vargas became i was always aware that something did n't quite feel right . xxmaj victor xxmaj vargas suffers from a certain xxunk on the director 's part . xxmaj apparently , the director thought that the ethnic backdrop of a xxmaj latino family on the lower east side , and an xxunk storyline would make the film critic proof . xxmaj he was right , but it did n't fool me . xxmaj raising xxmaj victor xxmaj vargas is</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxbos xxup the xxup shop xxup</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>3</th>
      <td>xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>6</th>
      <td>xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>7</th>
      <td>xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>8</th>
      <td>xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>


## Tabular data

Tabular data doesn't really use the data block API as it's relying on another API with [`TabularPandas`](/tabular.core.html#TabularPandas) for efficient preprocessing and batching (there will be some less efficient API that plays nicely with the data block API added in the near future). You can still use different blocks for the targets. 

```python
from fastai.tabular.core import *
```

For our example, we will look at a subset of the [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) which contains some census data and where the task is to predict if someone makes more than 50k or not.

```python
adult_source = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(adult_source/'adult.csv')
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



In a tabular problem, we need to split the columns between the ones that represent continuous variables (like the age) and the ones that represent categorical variables (like the education):

```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
```

Standard preprocessing in fastai, use those pre-processors: 

```python
procs = [Categorify, FillMissing, Normalize]
```

[`Categorify`](/tabular.core.html#Categorify) will change the categorical columns into indices, [`FillMissing`](/tabular.core.html#FillMissing) will fill the missing values in the continuous columns (if any) and add an na categorical column (if necessary). [`Normalize`](/data.transforms.html#Normalize) will normalize the continous columns (substract the mean and divide by the standard deviation).

We can still use any splitter to create the splits as we'd like them:

```python
splits = RandomSplitter()(range_of(df))
```

And then everything goes in a [`TabularPandas`](/tabular.core.html#TabularPandas) object:

```python
to = TabularPandas(df, procs, cat_names, cont_names, y_names="salary", splits=splits, y_block=CategoryBlock)
```

We put `y_block=CategoryBlock` just to show you how to customize the block for the targets, but it's usually inferred from the data, so you don't need to pass it, normally.

```python
dls = to.dataloaders()
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
      <td>Federal-gov</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>21.000000</td>
      <td>99199.000460</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>?</td>
      <td>Some-college</td>
      <td>Never-married</td>
      <td>?</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>21.000000</td>
      <td>116933.997502</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Self-emp-not-inc</td>
      <td>9th</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Other-relative</td>
      <td>White</td>
      <td>False</td>
      <td>56.000001</td>
      <td>201317.999844</td>
      <td>5.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Private</td>
      <td>Assoc-voc</td>
      <td>Divorced</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>36.000000</td>
      <td>211021.999814</td>
      <td>11.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Self-emp-not-inc</td>
      <td>HS-grad</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>58.000000</td>
      <td>204021.000322</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>?</td>
      <td>11th</td>
      <td>Never-married</td>
      <td>#na#</td>
      <td>Other-relative</td>
      <td>White</td>
      <td>True</td>
      <td>20.000001</td>
      <td>216562.998729</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>State-gov</td>
      <td>Doctorate</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>59.000001</td>
      <td>192258.000072</td>
      <td>16.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>?</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>Other</td>
      <td>False</td>
      <td>20.000001</td>
      <td>369678.000710</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>False</td>
      <td>43.000000</td>
      <td>178976.000199</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Local-gov</td>
      <td>Masters</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>38.000000</td>
      <td>40955.001812</td>
      <td>14.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>

