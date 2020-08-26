# Tutorial - Assemble the data on the pets dataset
> Using `Datasets`, `Pipeline`, `TfmdLists` and `Transform` in computer vision


## Overview

In this tutorial, we look in depth at the middle level API for collecting data in computer vision. First we will see how to use:

- [`Transform`](https://fastcore.fast.ai/transform#Transform) to process the data
- [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) to composes transforms

Those are just functions with added functionality. For dataset processing, we will look in a second part at 

- [`TfmdLists`](/data.core.html#TfmdLists) to apply one [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) of `Tranform`s on a collection of items
- [`Datasets`](/data.core.html#Datasets) to apply several [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) of [`Transform`](https://fastcore.fast.ai/transform#Transform)s on a collection of items in parallel and produce tuples

The general rule is to use [`TfmdLists`](/data.core.html#TfmdLists) when your transforms will output the tuple (input,target) and [`Datasets`](/data.core.html#Datasets) when you build separate [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline)s for each of your input(s)/target(s).

After this tutorial, you might be interested by the [siamese tutorial](http://docs.fast.ai/tutorial.siamese) that goes even more in depth in the data APIs, showing you how to write your custom types and how to customize the behavior of `show_batch` and `show_results`.

```python
from fastai.vision.all import *
```

## Processing data

Cleaning and processing data is one of the most time-consuming things in machine learning, which is why fastai tries to help you as much as it can. At its core, preparing the data for your model can be formalized as a sequence of transformations you apply to some raw items. For instance, in a classic image classification problem, we start with filenames. We have to open the corresponding images, resize them, convert them to tensors, maybe apply some kind of data augmentation, before we are ready to batch them. And that's just for the inputs of our model, for the targets, we need to extract the label of our filename and convert it to an integer.

This process needs to be somewhat reversible, because we often want to inspect our data to double check what we feed the model actually makes sense. That's why fastai represents all those operations by [`Transform`](https://fastcore.fast.ai/transform#Transform)s, which you can sometimes undo with a `decode` method.

### Transform

First we'll have a look at the basic steps using a single MNIST image. We'll start with a filename, and see step by step how it can be converted in to a labelled image that can be displayed and used for modeling. We use the usual [`untar_data`](/data.external.html#untar_data) to download our dataset (if necessary) and get all the image files:

```python
source = untar_data(URLs.MNIST_TINY)/'train'
items = get_image_files(source)
fn = items[0]; fn
```




    Path('/home/sgugger/.fastai/data/mnist_tiny/train/3/7861.png')



We'll look at each [`Transform`](https://fastcore.fast.ai/transform#Transform) needed in turn. Here's how we can open an image file:

```python
img = PILImage.create(fn); img
```




![png](output_10_0.png)



Then we can convert it to a `C*H*W` tensor (for channel x height x width, which is the convention in PyTorch):

```python
tconv = ToTensor()
img = tconv(img)
img.shape,type(img)
```




    (torch.Size([3, 28, 28]), fastai.torch_core.TensorImage)



Now that's done, we can create our labels. First extracting the text label:

```python
lbl = parent_label(fn); lbl
```




    '3'



And then converting to an int for modeling:

```python
tcat = Categorize(vocab=['3','7'])
lbl = tcat(lbl); lbl
```




    TensorCategory(0)



We use `decode` to reverse transforms for display. Reversing the [`Categorize`](/data.transforms.html#Categorize) transform result in a class name we can display:

```python
lbld = tcat.decode(lbl)
lbld
```




    '3'



### Pipeline

We can compose our image steps using [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline):

```python
pipe = Pipeline([PILImage.create,tconv])
img = pipe(fn)
img.shape
```




    torch.Size([3, 28, 28])



A [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) can decode and show an item.

```python
pipe.show(img, figsize=(1,1), cmap='Greys');
```


![png](output_23_0.png)


The show method works behind the scenes with types. Transforms will make sure the type of an element they receive is preserved. Here [`PILImage.create`](/vision.core.html#PILImage.create) returns a [`PILImage`](/vision.core.html#PILImage), which knows how to show itself. `tconv` converts it to a [`TensorImage`](/torch_core.html#TensorImage), which also knows how to show itself.

```python
type(img)
```




    fastai.torch_core.TensorImage



Those types are also used to enable different behaviors depending on the input received (for instance you don't do data augmentation the same way on an image, a segmentation mask or a bounding box).

### Creating your own [`Transform`](https://fastcore.fast.ai/transform#Transform)

Creating your own [`Transform`](https://fastcore.fast.ai/transform#Transform) is way easier than you think. In fact, each time you have passed a label function to the data block API or to [`ImageDataLoaders.from_name_func`](/vision.data.html#ImageDataLoaders.from_name_func), you have created a [`Transform`](https://fastcore.fast.ai/transform#Transform) without knowing it. At its base, a [`Transform`](https://fastcore.fast.ai/transform#Transform) is just a function. Let's show how you can easily add a transform by implementing one that wraps a data augmentation from the [albumentations library](https://github.com/albumentations-team/albumentations).

First things first, you will need to install the albumentations library. Uncomment the following cell to do so if needed:

```python

```

Then it's going to be easier to see the result of the transform on a color image bigger than the mnist one we had before, so let's load something from the PETS dataset.

```python
source = untar_data(URLs.PETS)
items = get_image_files(source/"images")
```

We can still open it with `PILIlmage.create`:

```python
img = PILImage.create(items[0])
img
```




![png](output_33_0.png)



We will show how to wrap one transform, but you can as easily wrap any set of transforms you wrapped in a `Compose` method. Here let's do some `ShiftScaleRotate`:

```python
from albumentations import ShiftScaleRotate
```

The albumentations transform work on numpy images, so we just convert our [`PILImage`](/vision.core.html#PILImage) to a numpy array before wrapping it back in [`PILImage.create`](/vision.core.html#PILImage.create) (this function takes filenames as well as arrays or tensors).

```python
aug = ShiftScaleRotate(p=1)
def aug_tfm(img): 
    np_img = np.array(img)
    aug_img = aug(image=np_img)['image']
    return PILImage.create(aug_img)
```

```python
aug_tfm(img)
```




![png](output_38_0.png)



We can pass this function each time a [`Transform`](https://fastcore.fast.ai/transform#Transform) is expected and the fastai library will automatically do the conversion. That's because you can directly pass such a function to create a [`Transform`](https://fastcore.fast.ai/transform#Transform):

```python
tfm = Transform(aug_tfm)
```

If you have some state in your transform, you might want to create a subclass of [`Transform`](https://fastcore.fast.ai/transform#Transform). In that case, the function you want to apply should be written in the <code>encodes</code> method (the same way you implement `forward` for PyTorch module):

```python
class AlbumentationsTransform(Transform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)
```

We also added a type annotation: this will make sure this transform is only applied to [`PILImage`](/vision.core.html#PILImage)s and their subclasses. For any other object, it won't do anything. You can also write as many <code>encodes</code> method you want with different type-annotations and the [`Transform`](https://fastcore.fast.ai/transform#Transform) will properly dispatch the objects it receives.

This is because in practice, the transform is often applied as an `item_tfms` (or a `batch_tfms`) that you pass in the data block API. Those items are a tuple of objects of different types, and the transform may have different behaviors on each part of the tuple.

Let's check here how this works:

```python
tfm = AlbumentationsTransform(ShiftScaleRotate(p=1))
a,b = tfm((img, 'dog'))
show_image(a, title=b);
```


![png](output_44_0.png)


The transform was applied over the tuple `(img, "dog")`. `img` is a [`PILImage`](/vision.core.html#PILImage), so it applied the <code>encodes</code> method we wrote. `"dog"` is a string, so the transform did nothing to it. 

Sometimes however, you need your transform to take your tuple as whole: for instance albumentations is applied simultaneously on images and segmentation masks. In this case you need to subclass `ItemTransfrom` instead of [`Transform`](https://fastcore.fast.ai/transform#Transform). Let's see how this works:

```python
cv_source = untar_data(URLs.CAMVID_TINY)
cv_items = get_image_files(cv_source/'images')
img = PILImage.create(cv_items[0])
mask = PILMask.create(cv_source/'labels'/f'{cv_items[0].stem}_P{cv_items[0].suffix}')
ax = img.show()
ax = mask.show(ctx=ax)
```


![png](output_46_0.png)


We then write a subclass of [`ItemTransform`](https://fastcore.fast.ai/transform#ItemTransform) that can wrap any albumentations augmentation transform, but only for a segmentation problem:

```python
class SegmentationAlbumentationsTransform(ItemTransform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, x):
        img,mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])
```

And we can check how it gets applied on the tuple `(img, mask)`. This means you can pass it as an `item_tfms` in any segmentation problem.

```python
tfm = SegmentationAlbumentationsTransform(ShiftScaleRotate(p=1))
a,b = tfm((img, mask))
ax = a.show()
ax = b.show(ctx=ax)
```


![png](output_50_0.png)


There is more you can implement in a [`Transform`](https://fastcore.fast.ai/transform#Transform): you can reverse it's behavior by adding a <code>decodes</code> and `setup` some state, we'll look at this in the next section:

## Loading the pets dataset using only [`Transform`](https://fastcore.fast.ai/transform#Transform)

Let's see how to use `fastai.data` to process the Pets dataset. If you are used to writing your own PyTorch `Dataset`s, what will feel more natural is to write everything in one [`Transform`](https://fastcore.fast.ai/transform#Transform). We use *source* to refer to the underlying source of our data (e.g. a directory on disk, a database connection, a network connection, etc). Then we grab the items.

```python
source = untar_data(URLs.PETS)/"images"
items = get_image_files(source)
```

We'll use this function to create consistently sized tensors from image files:

```python
def resized_image(fn:Path, sz=128):
    x = Image.open(fn).convert('RGB').resize((sz,sz))
    # Convert image to tensor for modeling
    return tensor(array(x)).permute(2,0,1).float()/255.
```

Before we can create a [`Transform`](https://fastcore.fast.ai/transform#Transform), we need a type that knows how to show itself (if we want to use the show method). Here we define a `TitledImage`:

```python
class TitledImage(fastuple):
    def show(self, ctx=None, **kwargs): show_titled_image(self, ctx=ctx, **kwargs)
```

Let's check it works:

```python
img = resized_image(items[0])
TitledImage(img,'test title').show()
```


![png](output_60_0.png)


### Using decodes for showing processed data

To decode data for showing purposes (like de-normalizing an image or converting back an index to its corresponding class), we implement a <code>decodes</code> method inside a [`Transform`](https://fastcore.fast.ai/transform#Transform).

```python
class PetTfm(Transform):
    def __init__(self, vocab, o2i, lblr): self.vocab,self.o2i,self.lblr = vocab,o2i,lblr
    def encodes(self, o): return [resized_image(o), self.o2i[self.lblr(o)]]
    def decodes(self, x): return TitledImage(x[0],self.vocab[x[1]])
```

The [`Transform`](https://fastcore.fast.ai/transform#Transform) opens and resizes the images on one side, label it and convert that label to an index using `o2i` on the other side. Inside the <code>decodes</code> method, we decode the index using the `vocab`. The image is left as is (we can't really show a filename!).

To use this [`Transform`](https://fastcore.fast.ai/transform#Transform), we need a label function. Here we use a regex on the `name` attribute of our filenames:

```python
labeller = using_attr(RegexLabeller(pat = r'^(.*)_\d+.jpg$'), 'name')
```

Then we gather all the possible labels, uniqueify them and ask for the two correspondences (vocab and o2i) using `bidir=True`. We can then use them to build our pet transform.

```python
vals = list(map(labeller, items))
vocab,o2i = uniqueify(vals, sort=True, bidir=True)
pets = PetTfm(vocab,o2i,labeller)
```

We can check how it's applied to a filename:

```python
x,y = pets(items[0])
x.shape,y
```




    (torch.Size([3, 128, 128]), 36)



And we can decode our transformed version and show it:

```python
dec = pets.decode([x,y])
dec.show()
```


![png](output_71_0.png)


Note that like `__call__ ` and <code>encodes</code>, we implemented a <code>decodes</code> method but we actually call `decode` on our [`Transform`](https://fastcore.fast.ai/transform#Transform).

Also note that our <code>decodes</code> method received the two objects (x and y). We said in the previous section [`Transform`](https://fastcore.fast.ai/transform#Transform) dispatch over tuples (for the encoding as well as the decodeing) but here it took our two elements as a whole and did not try to decode x and y separately. Why is that? It's because we pass a list `[x,y]` to decodes. [`Transform`](https://fastcore.fast.ai/transform#Transform)s dispatch over tuples, but tuples only. And as we saw as well, to prevent a [`Transform`](https://fastcore.fast.ai/transform#Transform) from dispatching over a tuple, we just have to make it an [`ItemTransform`](https://fastcore.fast.ai/transform#ItemTransform):

```python
class PetTfm(ItemTransform):
    def __init__(self, vocab, o2i, lblr): self.vocab,self.o2i,self.lblr = vocab,o2i,lblr
    def encodes(self, o): return (resized_image(o), self.o2i[self.lblr(o)])
    def decodes(self, x): return TitledImage(x[0],self.vocab[x[1]])
```

```python
dec = pets.decode(pets(items[0]))
dec.show()
```


![png](output_74_0.png)


### Setting up the internal state with a setups

We can now let's make our [`ItemTransform`](https://fastcore.fast.ai/transform#ItemTransform) automatically state its state form the data. This way, when we combine together our [`Transform`](https://fastcore.fast.ai/transform#Transform) with the data, it will automatically get setup without having to do anything. This is very easy to do: just copy the lines we had before to build the categories inside the transform in a <code>setups</code> method:

```python
class PetTfm(ItemTransform):
    def setups(self, items):
        self.labeller = using_attr(RegexLabeller(pat = r'^(.*)_\d+.jpg$'), 'name')
        vals = map(self.labeller, items)
        self.vocab,self.o2i = uniqueify(vals, sort=True, bidir=True)

    def encodes(self, o): return (resized_image(o), self.o2i[self.labeller(o)])
    def decodes(self, x): return TitledImage(x[0],self.vocab[x[1]])
```

Now we can create our [`Transform`](https://fastcore.fast.ai/transform#Transform), call its setup, and it will be ready to be used:

```python
pets = PetTfm()
pets.setup(items)
x,y = pets(items[0])
x.shape, y
```




    (torch.Size([3, 128, 128]), 36)



And like before, there is no problem to decode it:

```python
dec = pets.decode((x,y))
dec.show()
```


![png](output_81_0.png)


### Combining our [`Transform`](https://fastcore.fast.ai/transform#Transform) with data augmentation in a [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline).

We can take advantage of fastai's data augmentation transforms if we give the right type to our elements. Instead of returning a standard `PIL.Image`, if our transform returns the fastai type [`PILImage`](/vision.core.html#PILImage), we can then use any fastai's transform with it. Let's just return a [`PILImage`](/vision.core.html#PILImage) for our first element:

```python
class PetTfm(ItemTransform):
    def setups(self, items):
        self.labeller = using_attr(RegexLabeller(pat = r'^(.*)_\d+.jpg$'), 'name')
        vals = map(self.labeller, items)
        self.vocab,self.o2i = uniqueify(vals, sort=True, bidir=True)

    def encodes(self, o): return (PILImage.create(o), self.o2i[self.labeller(o)])
    def decodes(self, x): return TitledImage(x[0],self.vocab[x[1]])
```

We can then combine that transform with [`ToTensor`](/data.transforms.html#ToTensor), [`Resize`](/vision.augment.html#Resize) or [`FlipItem`](/vision.augment.html#FlipItem) to randomly flip our image in a [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline):

```python
tfms = Pipeline([PetTfm(), Resize(224), FlipItem(p=1), ToTensor()])
```

Calling `setup` on a [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) will set each transform in order:

```python
tfms.setup(items)
```

To check the setup was done properly, we want to see if we did build the vocab. One cool trick of [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) is that when asking for an attribute, it will look through each of its [`Transform`](https://fastcore.fast.ai/transform#Transform)s for that attribute and give you the result (or the list of results if the attribute is in multiple transforms):

```python
tfms.vocab
```




    (#37) ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair','Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue'...]



Then we can call our pipeline:

```python
x,y = tfms(items[0])
x.shape,y
```




    (torch.Size([3, 224, 224]), 36)



We can see [`ToTensor`](/data.transforms.html#ToTensor) and [`Resize`](/vision.augment.html#Resize) were applied to the first element of our tuple (which was of type [`PILImage`](/vision.core.html#PILImage)) but not the second. We can even have a look at our element to check the flip was also applied:

```python
tfms.show(tfms(items[0]))
```


![png](output_94_0.png)


[`Pipeline.show`](https://fastcore.fast.ai/transform#Pipeline.show) will call decode on each [`Transform`](https://fastcore.fast.ai/transform#Transform) until it gets a type that knows how to show itself. The library considers a tuple as knowing how to show itself if all its parts have a `show` method. Here it does not happen before reaching `PetTfm` since the second part of our tuple is an int. But after decoding the original `PetTfm`, we get a `TitledImage` which has a `show` method.

It's a good point to note that the [`Transform`](https://fastcore.fast.ai/transform#Transform)s of the [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) are sorted by their internal `order` attribute (with a default of `order=0`). You can always check the order in which the transforms are in a [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) by looking at its representation:

```python
tfms
```




    Pipeline: PetTfm -> FlipItem -> Resize -> ToTensor



Even if we define `tfms` with [`Resize`](/vision.augment.html#Resize) before [`FlipItem`](/vision.augment.html#FlipItem), we can see they have been reordered because we have:

```python
FlipItem.order,Resize.order
```




    (0, 1)



To customize the order of a [`Transform`](https://fastcore.fast.ai/transform#Transform), just set `order = ...` before the `__init__` (it's a class attribute). Let's make `PetTfm` of order -5 to be sure it's always run first:

```python
class PetTfm(ItemTransform):
    order = -5
    def setups(self, items):
        self.labeller = using_attr(RegexLabeller(pat = r'^(.*)_\d+.jpg$'), 'name')
        vals = map(self.labeller, items)
        self.vocab,self.o2i = uniqueify(vals, sort=True, bidir=True)

    def encodes(self, o): return (PILImage.create(o), self.o2i[self.labeller(o)])
    def decodes(self, x): return TitledImage(x[0],self.vocab[x[1]])
```

Then we can mess up the order of the transforms in our [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) but it will fix itself:

```python
tfms = Pipeline([Resize(224), PetTfm(), FlipItem(p=1), ToTensor()])
tfms
```




    Pipeline: PetTfm -> FlipItem -> Resize -> ToTensor



Now that we have a good [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) of transforms, let's add it to a list of filenames to build our dataset. A [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) combined with a collection is a [`TfmdLists`](/data.core.html#TfmdLists) in fastai.

## [`TfmdLists`](/data.core.html#TfmdLists) and [`Datasets`](/data.core.html#Datasets)

The main difference between [`TfmdLists`](/data.core.html#TfmdLists) and [`Datasets`](/data.core.html#Datasets) is the number of [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline)s you have: [`TfmdLists`](/data.core.html#TfmdLists) take one [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) to transform a list (like we currently have) whereas [`Datasets`](/data.core.html#Datasets) combines several [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline)s in parallel to create a tuple from one set of raw items, for instance a tuple (input, target).

### One pipeline makes a [`TfmdLists`](/data.core.html#TfmdLists)

Creating a [`TfmdLists`](/data.core.html#TfmdLists) just requires a list of items and a list of transforms that will be combined in a [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline):

```python
tls = TfmdLists(items, [Resize(224), PetTfm(), FlipItem(p=0.5), ToTensor()])
x,y = tls[0]
x.shape,y
```




    (torch.Size([3, 224, 224]), 36)



We did not need to pass anything to `PetTfm` thanks to our setup method: the [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline) was automatically setup on the `items` during the initialization, so `PetTfm` has created its vocab like before:

```python
tls.vocab
```




    (#37) ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair','Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue'...]



We can ask the [`TfmdLists`](/data.core.html#TfmdLists) to show the items we got:

```python
tls.show((x,y))
```


![png](output_112_0.png)


Or we have a shortcut with [`show_at`](/data.core.html#show_at):

```python
show_at(tls, 0)
```


![png](output_114_0.png)


### Traning and validation set

[`TfmdLists`](/data.core.html#TfmdLists) has an 's' in its name because it can represent several transformed lists: your training and validation sets. To use that functionality, we just need to pass `splits` to the initialization. `splits` should be a list of lists of indices (one list per set). To help create splits, we can use all the *splitters* of the fastai library:

```python
splits = RandomSplitter(seed=42)(items)
splits
```




    ((#5912) [5643,5317,5806,3460,613,5456,2968,3741,10,4908...],
     (#1478) [4512,4290,5770,706,2200,4320,6450,501,1290,6435...])



```python
tls = TfmdLists(items, [Resize(224), PetTfm(), FlipItem(p=0.5), ToTensor()], splits=splits)
```

Then your `tls` get a train and valid attributes (it also had them before, but the valid was empty and the train contained everything).

```python
show_at(tls.train, 0)
```


![png](output_120_0.png)


An interesting thing is that unless you pass `train_setup=False`, your transforms are setup on the training set only (which is best practices): the `items` received by <code>setups</code> are just the elements of the training set. 

### Getting to [`DataLoaders`](/data.core.html#DataLoaders)

From a [`TfmdLists`](/data.core.html#TfmdLists), getting a [`DataLoaders`](/data.core.html#DataLoaders) object is very easy, you just have to call the `dataloaders` method:

```python
dls = tls.dataloaders(bs=64)
```

And `show_batch` will just *work*:

```python
dls.show_batch()
```


![png](output_126_0.png)


You can even add augmentation transforms, since we have a proper fastai typed image. Just remember to add the [`IntToFloatTensor`](/data.transforms.html#IntToFloatTensor)  transform that deals with the conversion of int to float (augmentation transforms of fastai on the GPU require float tensors). When calling [`TfmdLists.dataloaders`](/data.core.html#TfmdLists.dataloaders), you pass the `batch_tfms` to `after_batch` (and potential new `item_tfms` to `after_item`):

```python
dls = tls.dataloaders(bs=64, after_batch=[IntToFloatTensor(), *aug_transforms()])
dls.show_batch()
```


![png](output_128_0.png)


### Using [`Datasets`](/data.core.html#Datasets)

[`Datasets`](/data.core.html#Datasets) applies a list of list of transforms (or list of [`Pipeline`](https://fastcore.fast.ai/transform#Pipeline)s) lazily to items of a collection, creating one output per list of transforms/[`Pipeline`](https://fastcore.fast.ai/transform#Pipeline). This makes it easier for us to separate out steps of a process, so that we can re-use them and modify the process more easily. This is what lays the foundation of the data block API: we can easily mix and match types as inputs or outputs as they are associated to certain pipelines of transforms.

For instacnce, let's write our own `ImageResizer` transform with two different implementations for images or masks:

```python
class ImageResizer(Transform):
    order=1
    "Resize image to `size` using `resample`"
    def __init__(self, size, resample=Image.BILINEAR):
        if not is_listy(size): size=(size,size)
        self.size,self.resample = (size[1],size[0]),resample

    def encodes(self, o:PILImage): return o.resize(size=self.size, resample=self.resample)
    def encodes(self, o:PILMask):  return o.resize(size=self.size, resample=Image.NEAREST)
```

Specifying the type-annotations makes it so that our transform does nothing to thigns that are neither [`PILImage`](/vision.core.html#PILImage) or [`PILMask`](/vision.core.html#PILMask), and resize images with `self.resample`, masks with the nearest neighbor interpolation. To create a [`Datasets`](/data.core.html#Datasets), we then pass two pipelines of transforms, one for the input and one for the target:

```python
tfms = [[PILImage.create, ImageResizer(128), ToTensor(), IntToFloatTensor()],
        [labeller, Categorize()]]
dsets = Datasets(items, tfms)
```

We can check that inputs and outputs have the right types:

```python
t = dsets[0]
type(t[0]),type(t[1])
```




    (fastai.torch_core.TensorImage, fastai.torch_core.TensorCategory)



We can decode and show using `dsets`:

```python
x,y = dsets.decode(t)
x.shape,y
```




    (torch.Size([3, 128, 128]), 'yorkshire_terrier')



```python
dsets.show(t);
```


![png](output_138_0.png)


And we can pass our train/validation split like in [`TfmdLists`](/data.core.html#TfmdLists):

```python
dsets = Datasets(items, tfms, splits=splits)
```

But we are not using the fact that [`Transform`](https://fastcore.fast.ai/transform#Transform)s dispatch over tuples here. `ImageResizer`, [`ToTensor`](/data.transforms.html#ToTensor) and [`IntToFloatTensor`](/data.transforms.html#IntToFloatTensor) could be passed as transforms over the tuple. This is done in `.dataloaders` by passing them to `after_item`. They won't do anything to the category but will only be applied to the inputs.

```python
tfms = [[PILImage.create], [labeller, Categorize()]]
dsets = Datasets(items, tfms, splits=splits)
dls = dsets.dataloaders(bs=64, after_item=[ImageResizer(128), ToTensor(), IntToFloatTensor()])
```

And we can check it works with `show_batch`:

```python
dls.show_batch()
```


![png](output_144_0.png)


If we just wanted to build one [`DataLoader`](/data.load.html#DataLoader) from our [`Datasets`](/data.core.html#Datasets) (or the previous [`TfmdLists`](/data.core.html#TfmdLists)), you can pass it directly to [`TfmdDL`](/data.core.html#TfmdDL):

```python
dsets = Datasets(items, tfms)
dl = TfmdDL(dsets, bs=64, after_item=[ImageResizer(128), ToTensor(), IntToFloatTensor()])
```

### Segmentation

By using the same transforms in `after_item` but a different kind of targets (here segmentation masks), the targets are automatically processed as they should with the type-dispatch system.

```python
cv_source = untar_data(URLs.CAMVID_TINY)
cv_items = get_image_files(cv_source/'images')
cv_splitter = RandomSplitter(seed=42)
cv_split = cv_splitter(cv_items)
cv_label = lambda o: cv_source/'labels'/f'{o.stem}_P{o.suffix}'
```

```python
tfms = [[PILImage.create], [cv_label, PILMask.create]]
cv_dsets = Datasets(cv_items, tfms, splits=cv_split)
dls = cv_dsets.dataloaders(bs=64, after_item=[ImageResizer(128), ToTensor(), IntToFloatTensor()])
```

    /opt/conda/conda-bld/pytorch_1585984269458/work/aten/src/ATen/native/BinaryOps.cpp:66: UserWarning: Integer division of tensors using div or / is deprecated, and in a future release div will perform true division as in Python 3. Use true_divide or floor_divide (// in Python) instead.


```python
dls.show_batch(max_n=4)
```


![png](output_151_0.png)


If we want to use the augmentation transform we created before, we just need to add one thing to it: we want it to be applied on the training set only, not the validation set. To do this, we specify it should only be applied on a specific `idx` of our splits by adding `split_idx=0` (0 is for the training set, 1 for the validation set):

```python
class SegmentationAlbumentationsTransform(ItemTransform):
    split_idx = 0
    def __init__(self, aug): self.aug = aug
    def encodes(self, x):
        img,mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])
```

And we can check how it gets applied on the tuple `(img, mask)`. This means you can pass it as an `item_tfms` in any segmentation problem.

```python
cv_dsets = Datasets(cv_items, tfms, splits=cv_split)
dls = cv_dsets.dataloaders(bs=64, after_item=[ImageResizer(128), ToTensor(), IntToFloatTensor(), 
                                              SegmentationAlbumentationsTransform(ShiftScaleRotate(p=1))])
```

```python
dls.show_batch(max_n=4)
```


![png](output_156_0.png)


## Adding a test dataloader for inference

Let's take back our pets dataset...

```python
tfms = [[PILImage.create], [labeller, Categorize()]]
dsets = Datasets(items, tfms, splits=splits)
dls = dsets.dataloaders(bs=64, after_item=[ImageResizer(128), ToTensor(), IntToFloatTensor()])
```

...and imagine we have some new files to classify.

```python
path = untar_data(URLs.PETS)
tst_files = get_image_files(path/"images")
```

```python
len(tst_files)
```




    7390



We can create a dataloader that takes those files and applies the same transforms as the validation set with [`DataLoaders.test_dl`](/data.core.html#DataLoaders.test_dl):

```python
tst_dl = dls.test_dl(tst_files)
```

```python
tst_dl.show_batch(max_n=9)
```


![png](output_165_0.png)


**Extra:**  
You can call `learn.get_preds` passing this newly created dataloaders to make predictions on our new images!  
What is really cool is that after you finished training your model, you can save it with `learn.export`, this is also going to save all the transforms that need to be applied to your data. In inference time you just need to load your learner with [`load_learner`](/learner.html#load_learner) and you can immediately create a dataloader with `test_dl` to use it to generate new predictions!
