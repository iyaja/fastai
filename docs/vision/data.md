# Vision data
> Helper functions to get data in a `DataLoaders` in the vision application and higher class `ImageDataLoaders`


The main classes defined in this module are [`ImageDataLoaders`](/vision.data.html#ImageDataLoaders) and [`SegmentationDataLoaders`](/vision.data.html#SegmentationDataLoaders), so you probably want to jump to their definitions. They provide factory methods that are a great way to quickly get your data ready for training, see the [vision tutorial](http://docs.fast.ai/tutorial.vision) for examples.

## Helper functions


<h4 id="get_grid" class="doc_header"><code>get_grid</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L12" class="source_link" style="float:right">[source]</a></h4>

> <code>get_grid</code>(**`n`**, **`nrows`**=*`None`*, **`ncols`**=*`None`*, **`add_vert`**=*`0`*, **`figsize`**=*`None`*, **`double`**=*`False`*, **`title`**=*`None`*, **`return_fig`**=*`False`*, **`imsize`**=*`3`*, **`sharex`**=*`False`*, **`sharey`**=*`False`*, **`squeeze`**=*`True`*, **`subplot_kw`**=*`None`*, **`gridspec_kw`**=*`None`*)

Return a grid of `n` axes, `rows` by `cols`


This is used by the type-dispatched versions of `show_batch` and `show_results` for the vision application. By default, there will be `int(math.sqrt(n))` rows and `ceil(n/rows)` columns. `double` will double the number of columns and `n`. The default `figsize` is `(cols*imsize, rows*imsize+add_vert)`. If a `title` is passed it is set to the figure. `sharex`, `sharey`, `squeeze`, `subplot_kw` and `gridspec_kw` are all passed down to `plt.subplots`. If `return_fig` is `True`, returns `fig,axs`, otherwise just `axs`.


<h4 id="clip_remove_empty" class="doc_header"><code>clip_remove_empty</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L24" class="source_link" style="float:right">[source]</a></h4>

> <code>clip_remove_empty</code>(**`bbox`**, **`label`**)

Clip bounding boxes with image border and label background the empty ones


```python
bb = tensor([[-2,-0.5,0.5,1.5], [-0.5,-0.5,0.5,0.5], [1,0.5,0.5,0.75], [-0.5,-0.5,0.5,0.5]])
bb,lbl = clip_remove_empty(bb, tensor([1,2,3,2]))
test_eq(bb, tensor([[-1,-0.5,0.5,1.], [-0.5,-0.5,0.5,0.5], [-0.5,-0.5,0.5,0.5]]))
test_eq(lbl, tensor([1,2,2]))
```


<h4 id="bb_pad" class="doc_header"><code>bb_pad</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L31" class="source_link" style="float:right">[source]</a></h4>

> <code>bb_pad</code>(**`samples`**, **`pad_idx`**=*`0`*)

Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`.


```python
img1,img2 = TensorImage(torch.randn(16,16,3)),TensorImage(torch.randn(16,16,3))
bb1 = tensor([[-2,-0.5,0.5,1.5], [-0.5,-0.5,0.5,0.5], [1,0.5,0.5,0.75], [-0.5,-0.5,0.5,0.5]])
lbl1 = tensor([1, 2, 3, 2])
bb2 = tensor([[-0.5,-0.5,0.5,0.5], [-0.5,-0.5,0.5,0.5]])
lbl2 = tensor([2, 2])
samples = [(img1, bb1, lbl1), (img2, bb2, lbl2)]
res = bb_pad(samples)
non_empty = tensor([True,True,False,True])
test_eq(res[0][0], img1)
test_eq(res[0][1], tensor([[-1,-0.5,0.5,1.], [-0.5,-0.5,0.5,0.5], [-0.5,-0.5,0.5,0.5]]))
test_eq(res[0][2], tensor([1,2,2]))
test_eq(res[1][0], img2)
test_eq(res[1][1], tensor([[-0.5,-0.5,0.5,0.5], [-0.5,-0.5,0.5,0.5], [0,0,0,0]]))
test_eq(res[1][2], tensor([2,2,0]))      
```

## [`TransformBlock`](/data.block.html#TransformBlock)s for vision

These are the blocks the vision application provide for the [data block API](http://docs.fast.ai/data.block).


<h4 id="ImageBlock" class="doc_header"><code>ImageBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L57" class="source_link" style="float:right">[source]</a></h4>

> <code>ImageBlock</code>()

A [`TransformBlock`](/data.block.html#TransformBlock) for images of `cls`



<h4 id="MaskBlock" class="doc_header"><code>MaskBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L62" class="source_link" style="float:right">[source]</a></h4>

> <code>MaskBlock</code>(**`codes`**=*`None`*)

A [`TransformBlock`](/data.block.html#TransformBlock) for segmentation masks, potentially with `codes`



<h4 id="PointBlock" class="doc_header"><code>PointBlock</code><a href="" class="source_link" style="float:right">[source]</a></h4>

A [`TransformBlock`](/data.block.html#TransformBlock) for points in an image



<h4 id="BBoxBlock" class="doc_header"><code>BBoxBlock</code><a href="" class="source_link" style="float:right">[source]</a></h4>

A [`TransformBlock`](/data.block.html#TransformBlock) for bounding boxes in an image



<h4 id="BBoxLblBlock" class="doc_header"><code>BBoxLblBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L74" class="source_link" style="float:right">[source]</a></h4>

> <code>BBoxLblBlock</code>(**`vocab`**=*`None`*, **`add_na`**=*`True`*)

A [`TransformBlock`](/data.block.html#TransformBlock) for labeled bounding boxes, potentially with `vocab`


If `add_na` is `True`, a new category is added for NaN (that will represent the background class).


<h2 id="ImageDataLoaders" class="doc_header"><code>class</code> <code>ImageDataLoaders</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L79" class="source_link" style="float:right">[source]</a></h2>

> <code>ImageDataLoaders</code>(**\*`loaders`**, **`path`**=*`'.'`*, **`device`**=*`None`*) :: [`DataLoaders`](/data.core.html#DataLoaders)

Basic wrapper around several [`DataLoader`](/data.load.html#DataLoader)s with factory methods for computer vision problems


This class should not be used directly, one of the factory methods should be preferred instead. All those factory methods accept as arguments:

- `item_tfms`: one or several transforms applied to the items before batching them
- `batch_tfms`: one or several transforms applied to the batches once they are formed
- `bs`: the batch size
- `val_bs`: the batch size for the validation [`DataLoader`](/data.load.html#DataLoader) (defaults to `bs`)
- `shuffle_train`: if we shuffle the training [`DataLoader`](/data.load.html#DataLoader) or not
- `device`: the PyTorch device to use (defaults to `default_device()`)


<h4 id="ImageDataLoaders.from_folder" class="doc_header"><code>ImageDataLoaders.from_folder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L81" class="source_link" style="float:right">[source]</a></h4>

> <code>ImageDataLoaders.from_folder</code>(**`path`**, **`train`**=*`'train'`*, **`valid`**=*`'valid'`*, **`valid_pct`**=*`None`*, **`seed`**=*`None`*, **`vocab`**=*`None`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from imagenet style dataset in `path` with `train` and `valid` subfolders (or provide `valid_pct`)


If `valid_pct` is provided, a random split is performed (with an optional `seed`) by setting aside that percentage of the data for the validation set (instead of looking at the grandparents folder). If a `vocab` is passed, only the folders with names in `vocab` are kept.

Here is an example loading a subsample of MNIST:

```python
path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_folder(path)
```

Passing `valid_pct` will ignore the valid/train folders and do a new random split:

```python
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2)
dls.valid_ds.items[:3]
```




    [Path('/home/jhoward/.fastai/data/mnist_tiny/test/5071.png'),
     Path('/home/jhoward/.fastai/data/mnist_tiny/train/3/8684.png'),
     Path('/home/jhoward/.fastai/data/mnist_tiny/train/3/8188.png')]




<h4 id="ImageDataLoaders.from_path_func" class="doc_header"><code>ImageDataLoaders.from_path_func</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L96" class="source_link" style="float:right">[source]</a></h4>

> <code>ImageDataLoaders.from_path_func</code>(**`path`**, **`fnames`**, **`label_func`**, **`valid_pct`**=*`0.2`*, **`seed`**=*`None`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from list of `fnames` in `path`s with `label_func`


The validation set is a random `subset` of `valid_pct`, optionally created with `seed` for reproducibility.

Here is how to create the same [`DataLoaders`](/data.core.html#DataLoaders) on the MNIST dataset as the previous example with a `label_func`:

```python
fnames = get_image_files(path)
def label_func(x): return x.parent.name
dls = ImageDataLoaders.from_path_func(path, fnames, label_func)
```

Here is another example on the pets dataset. Here filenames are all in an "images" folder and their names have the form `class_name_123.jpg`. One way to properly label them is thus to throw away everything after the last `_`:


<h4 id="ImageDataLoaders.from_path_re" class="doc_header"><code>ImageDataLoaders.from_path_re</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L113" class="source_link" style="float:right">[source]</a></h4>

> <code>ImageDataLoaders.from_path_re</code>(**`path`**, **`fnames`**, **`pat`**, **`valid_pct`**=*`0.2`*, **`seed`**=*`None`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from list of `fnames` in `path`s with re expression `pat`


The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility.

Here is how to create the same [`DataLoaders`](/data.core.html#DataLoaders) on the MNIST dataset as the previous example (you will need to change the initial two / by a \ on Windows):

```python
pat = r'/([^/]*)/\d+.png$'
dls = ImageDataLoaders.from_path_re(path, fnames, pat)
```


<h4 id="ImageDataLoaders.from_name_func" class="doc_header"><code>ImageDataLoaders.from_name_func</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L107" class="source_link" style="float:right">[source]</a></h4>

> <code>ImageDataLoaders.from_name_func</code>(**`path`**, **`fnames`**, **`label_func`**, **`valid_pct`**=*`0.2`*, **`seed`**=*`None`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from the name attrs of `fnames` in `path`s with `label_func`


The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility. This method does the same as [`ImageDataLoaders.from_path_func`](/vision.data.html#ImageDataLoaders.from_path_func) except `label_func` is applied to the name of each filenames, and not the full path.


<h4 id="ImageDataLoaders.from_name_re" class="doc_header"><code>ImageDataLoaders.from_name_re</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L118" class="source_link" style="float:right">[source]</a></h4>

> <code>ImageDataLoaders.from_name_re</code>(**`path`**, **`fnames`**, **`pat`**, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from the name attrs of `fnames` in `path`s with re expression `pat`


The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility. This method does the same as [`ImageDataLoaders.from_path_re`](/vision.data.html#ImageDataLoaders.from_path_re) except `pat` is applied to the name of each filenames, and not the full path.


<h4 id="ImageDataLoaders.from_df" class="doc_header"><code>ImageDataLoaders.from_df</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L124" class="source_link" style="float:right">[source]</a></h4>

> <code>ImageDataLoaders.from_df</code>(**`df`**, **`path`**=*`'.'`*, **`valid_pct`**=*`0.2`*, **`seed`**=*`None`*, **`fn_col`**=*`0`*, **`folder`**=*`None`*, **`suff`**=*`''`*, **`label_col`**=*`1`*, **`label_delim`**=*`None`*, **`y_block`**=*`None`*, **`valid_col`**=*`None`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from `df` using `fn_col` and `label_col`


The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility. Alternatively, if your `df` contains a `valid_col`, give its name or its index to that argument (the column should have `True` for the elements going to the validation set). 

You can add an additional `folder` to the filenames in `df` if they should not be concatenated directly to `path`. If they do not contain the proper extensions, you can add `suff`. If your label column contains multiple labels on each row, you can use `label_delim` to warn the library you have a multi-label problem. 

`y_block` should be passed when the task automatically picked by the library is wrong, you should then give [`CategoryBlock`](/data.block.html#CategoryBlock), [`MultiCategoryBlock`](/data.block.html#MultiCategoryBlock) or [`RegressionBlock`](/data.block.html#RegressionBlock). For more advanced uses, you should use the data block API.

The tiny mnist example from before also contains a version in a dataframe:

```python
path = untar_data(URLs.MNIST_TINY)
df = pd.read_csv(path/'labels.csv')
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
      <th>name</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train/3/7463.png</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train/3/9829.png</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train/3/7881.png</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train/3/8065.png</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train/3/7046.png</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Here is how to load it using [`ImageDataLoaders.from_df`](/vision.data.html#ImageDataLoaders.from_df):

```python
dls = ImageDataLoaders.from_df(df, path)
```

Here is another example with a multi-label problem:

```python
path = untar_data(URLs.PASCAL_2007)
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



```python
dls = ImageDataLoaders.from_df(df, path, folder='train', valid_col='is_valid')
```

Note that can also pass `2` to valid_col (the index, starting with 0).


<h4 id="ImageDataLoaders.from_csv" class="doc_header"><code>ImageDataLoaders.from_csv</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L142" class="source_link" style="float:right">[source]</a></h4>

> <code>ImageDataLoaders.from_csv</code>(**`path`**, **`csv_fname`**=*`'labels.csv'`*, **`header`**=*`'infer'`*, **`delimiter`**=*`None`*, **`valid_pct`**=*`0.2`*, **`seed`**=*`None`*, **`fn_col`**=*`0`*, **`folder`**=*`None`*, **`suff`**=*`''`*, **`label_col`**=*`1`*, **`label_delim`**=*`None`*, **`y_block`**=*`None`*, **`valid_col`**=*`None`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from `path/csv_fname` using `fn_col` and `label_col`


Same as [`ImageDataLoaders.from_df`](/vision.data.html#ImageDataLoaders.from_df) after loading the file with `header` and `delimiter`.

Here is how to load the same dataset as before with this method:

```python
dls = ImageDataLoaders.from_csv(path, 'train.csv', folder='train', valid_col='is_valid')
```


<h4 id="ImageDataLoaders.from_lists" class="doc_header"><code>ImageDataLoaders.from_lists</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L148" class="source_link" style="float:right">[source]</a></h4>

> <code>ImageDataLoaders.from_lists</code>(**`path`**, **`fnames`**, **`labels`**, **`valid_pct`**=*`0.2`*, **`seed`**:`int`=*`None`*, **`y_block`**=*`None`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from list of `fnames` and `labels` in `path`


The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility. `y_block` can be passed to specify the type of the targets.

```python
path = untar_data(URLs.PETS)
fnames = get_image_files(path/"images")
labels = ['_'.join(x.name.split('_')[:-1]) for x in fnames]
dls = ImageDataLoaders.from_lists(path, fnames, labels)
```


<h2 id="SegmentationDataLoaders" class="doc_header"><code>class</code> <code>SegmentationDataLoaders</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L168" class="source_link" style="float:right">[source]</a></h2>

> <code>SegmentationDataLoaders</code>(**\*`loaders`**, **`path`**=*`'.'`*, **`device`**=*`None`*) :: [`DataLoaders`](/data.core.html#DataLoaders)

Basic wrapper around several [`DataLoader`](/data.load.html#DataLoader)s with factory methods for segmentation problems



<h4 id="SegmentationDataLoaders.from_label_func" class="doc_header"><code>SegmentationDataLoaders.from_label_func</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/data.py#L170" class="source_link" style="float:right">[source]</a></h4>

> <code>SegmentationDataLoaders.from_label_func</code>(**`path`**, **`fnames`**, **`label_func`**, **`valid_pct`**=*`0.2`*, **`seed`**=*`None`*, **`codes`**=*`None`*, **`item_tfms`**=*`None`*, **`batch_tfms`**=*`None`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from list of `fnames` in `path`s with `label_func`.


The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility. `codes` contain the mapping index to label.

```python
path = untar_data(URLs.CAMVID_TINY)
fnames = get_image_files(path/'images')
def label_func(x): return path/'labels'/f'{x.stem}_P{x.suffix}'
codes = np.loadtxt(path/'codes.txt', dtype=str)
    
dls = SegmentationDataLoaders.from_label_func(path, fnames, label_func, codes=codes)
```
