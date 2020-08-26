# Helper functions for processing data and basic transforms
> Functions for getting, splitting, and labeling data, as well as generic transforms


## Get, split, and label

For most data source creation we need functions to get a list of items, split them in to train/valid sets, and label them. fastai provides functions to make each of these steps easy (especially when combined with `fastai.data.blocks`).

### Get

First we'll look at functions that *get* a list of items (generally file names).

We'll use *tiny MNIST* (a subset of MNIST with just two classes, `7`s and `3`s) for our examples/tests throughout this page.

```python
path = untar_data(URLs.MNIST_TINY)
(path/'train').ls()
```




    (#2) [Path('/home/jhoward/.fastai/data/mnist_tiny/train/7'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/3')]




<h4 id="get_files" class="doc_header"><code>get_files</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L26" class="source_link" style="float:right">[source]</a></h4>

> <code>get_files</code>(**`path`**, **`extensions`**=*`None`*, **`recurse`**=*`True`*, **`folders`**=*`None`*, **`followlinks`**=*`True`*)

Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified.


This is the most general way to grab a bunch of file names from disk. If you pass `extensions` (including the `.`) then returned file names are filtered by that list. Only those files directly in `path` are included, unless you pass `recurse`, in which case all child folders are also searched recursively. `folders` is an optional list of directories to limit the search to.

```python
t3 = get_files(path/'train'/'3', extensions='.png', recurse=False)
t7 = get_files(path/'train'/'7', extensions='.png', recurse=False)
t  = get_files(path/'train', extensions='.png', recurse=True)
test_eq(len(t), len(t3)+len(t7))
test_eq(len(get_files(path/'train'/'3', extensions='.jpg', recurse=False)),0)
test_eq(len(t), len(get_files(path, extensions='.png', recurse=True, folders='train')))
t
```




    (#709) [Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/723.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/7446.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/8566.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/9200.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/7085.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/8665.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/7348.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/9283.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/9854.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/9548.png')...]



It's often useful to be able to create functions with customized behavior. `fastai.data` generally uses functions named as CamelCase verbs ending in `er` to create these functions. [`FileGetter`](/data.transforms.html#FileGetter) is a simple example of such a function creator.


<h4 id="FileGetter" class="doc_header"><code>FileGetter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L45" class="source_link" style="float:right">[source]</a></h4>

> <code>FileGetter</code>(**`suf`**=*`''`*, **`extensions`**=*`None`*, **`recurse`**=*`True`*, **`folders`**=*`None`*)

Create [`get_files`](/data.transforms.html#get_files) partial function that searches path suffix `suf`, only in `folders`, if specified, and passes along args


```python
fpng = FileGetter(extensions='.png', recurse=False)
test_eq(len(t7), len(fpng(path/'train'/'7')))
test_eq(len(t), len(fpng(path/'train', recurse=True)))
fpng_r = FileGetter(extensions='.png', recurse=True)
test_eq(len(t), len(fpng_r(path/'train')))
```


<h4 id="get_image_files" class="doc_header"><code>get_image_files</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L55" class="source_link" style="float:right">[source]</a></h4>

> <code>get_image_files</code>(**`path`**, **`recurse`**=*`True`*, **`folders`**=*`None`*)

Get image files in `path` recursively, only in `folders`, if specified.


This is simply [`get_files`](/data.transforms.html#get_files) called with a list of standard image extensions.

```python
test_eq(len(t), len(get_image_files(path, recurse=True, folders='train')))
```


<h4 id="ImageGetter" class="doc_header"><code>ImageGetter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L60" class="source_link" style="float:right">[source]</a></h4>

> <code>ImageGetter</code>(**`suf`**=*`''`*, **`recurse`**=*`True`*, **`folders`**=*`None`*)

Create [`get_image_files`](/data.transforms.html#get_image_files) partial function that searches path suffix `suf` and passes along `kwargs`, only in `folders`, if specified.


Same as [`FileGetter`](/data.transforms.html#FileGetter), but for image extensions.

```python
test_eq(len(get_files(path/'train', extensions='.png', recurse=True, folders='3')),
        len(ImageGetter(   'train',                    recurse=True, folders='3')(path)))
```


<h4 id="get_text_files" class="doc_header"><code>get_text_files</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L66" class="source_link" style="float:right">[source]</a></h4>

> <code>get_text_files</code>(**`path`**, **`recurse`**=*`True`*, **`folders`**=*`None`*)

Get text files in `path` recursively, only in `folders`, if specified.



<h2 id="ItemGetter" class="doc_header"><code>class</code> <code>ItemGetter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L71" class="source_link" style="float:right">[source]</a></h2>

> <code>ItemGetter</code>(**`i`**) :: [`ItemTransform`](https://fastcore.fast.ai/transform#ItemTransform)

Creates a proper transform that applies `itemgetter(i)` (even on a tuple)


```python
test_eq(ItemGetter(1)((1,2,3)),  2)
test_eq(ItemGetter(1)(L(1,2,3)), 2)
test_eq(ItemGetter(1)([1,2,3]),  2)
test_eq(ItemGetter(1)(np.array([1,2,3])),  2)
```


<h2 id="AttrGetter" class="doc_header"><code>class</code> <code>AttrGetter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L78" class="source_link" style="float:right">[source]</a></h2>

> <code>AttrGetter</code>(**`nm`**, **`default`**=*`None`*) :: [`ItemTransform`](https://fastcore.fast.ai/transform#ItemTransform)

Creates a proper transform that applies `attrgetter(nm)` (even on a tuple)


```python
test_eq(AttrGetter('shape')(torch.randn([4,5])), [4,5])
test_eq(AttrGetter('shape', [0])([4,5]), [0])
```

### Split

The next set of functions are used to *split* data into training and validation sets. The functions return two lists - a list of indices or masks for each of training and validation sets.


<h4 id="RandomSplitter" class="doc_header"><code>RandomSplitter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L85" class="source_link" style="float:right">[source]</a></h4>

> <code>RandomSplitter</code>(**`valid_pct`**=*`0.2`*, **`seed`**=*`None`*)

Create function that splits `items` between train/val with `valid_pct` randomly.


```python
src = list(range(30))
f = RandomSplitter(seed=42)
trn,val = f(src)
assert 0<len(trn)<len(src)
assert all(o not in val for o in trn)
test_eq(len(trn), len(src)-len(val))
# test random seed consistency
test_eq(f(src)[0], trn)
```

Use scikit-learn train_test_split. This allow to *split* items in a stratified fashion (uniformely according to the ‘labels‘ distribution)


<h4 id="TrainTestSplitter" class="doc_header"><code>TrainTestSplitter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L95" class="source_link" style="float:right">[source]</a></h4>

> <code>TrainTestSplitter</code>(**`test_size`**=*`0.2`*, **`random_state`**=*`None`*, **`stratify`**=*`None`*, **`train_size`**=*`None`*, **`shuffle`**=*`True`*)

Split `items` into random train and test subsets using sklearn train_test_split utility.


```python
src = list(range(30))
labels = [0] * 20 + [1] * 10
test_size = 0.2

f = TrainTestSplitter(test_size=test_size, random_state=42, stratify=labels)
trn,val = f(src)
assert 0<len(trn)<len(src)
assert all(o not in val for o in trn)
test_eq(len(trn), len(src)-len(val))

# test random seed consistency
test_eq(f(src)[0], trn)

# test labels distribution consistency
# there should be test_size % of zeroes and ones respectively in the validation set
test_eq(len([t for t in val if t < 20]) / 20, test_size)
test_eq(len([t for t in val if t > 20]) / 10, test_size)
```


<h4 id="IndexSplitter" class="doc_header"><code>IndexSplitter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L103" class="source_link" style="float:right">[source]</a></h4>

> <code>IndexSplitter</code>(**`valid_idx`**)

Split `items` so that `val_idx` are in the validation set and the others in the training set


```python
items = list(range(10))
splitter = IndexSplitter([3,7,9])
test_eq(splitter(items),[[0,1,2,4,5,6,8],[3,7,9]])
```


<h4 id="GrandparentSplitter" class="doc_header"><code>GrandparentSplitter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L116" class="source_link" style="float:right">[source]</a></h4>

> <code>GrandparentSplitter</code>(**`train_name`**=*`'train'`*, **`valid_name`**=*`'valid'`*)

Split `items` from the grand parent folder names (`train_name` and `valid_name`).


```python
fnames = [path/'train/3/9932.png', path/'valid/7/7189.png', 
          path/'valid/7/7320.png', path/'train/7/9833.png',  
          path/'train/3/7666.png', path/'valid/3/925.png',
          path/'train/7/724.png', path/'valid/3/93055.png']
splitter = GrandparentSplitter()
test_eq(splitter(fnames),[[0,3,4,6],[1,2,5,7]])
```

```python
fnames2 = fnames + [path/'test/3/4256.png', path/'test/7/2345.png', path/'valid/7/6467.png']
splitter = GrandparentSplitter(train_name=('train', 'valid'), valid_name='test')
test_eq(splitter(fnames2),[[0,3,4,6,1,2,5,7,10],[8,9]])
```


<h4 id="FuncSplitter" class="doc_header"><code>FuncSplitter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L123" class="source_link" style="float:right">[source]</a></h4>

> <code>FuncSplitter</code>(**`func`**)

Split `items` by result of `func` (`True` for validation, `False` for training set).


```python
splitter = FuncSplitter(lambda o: Path(o).parent.parent.name == 'valid')
test_eq(splitter(fnames),[[0,3,4,6],[1,2,5,7]])
```


<h4 id="MaskSplitter" class="doc_header"><code>MaskSplitter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L131" class="source_link" style="float:right">[source]</a></h4>

> <code>MaskSplitter</code>(**`mask`**)

Split `items` depending on the value of `mask`.


```python
items = list(range(6))
splitter = MaskSplitter([True,False,False,True,False,True])
test_eq(splitter(items),[[1,2,4],[0,3,5]])
```


<h4 id="FileSplitter" class="doc_header"><code>FileSplitter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L137" class="source_link" style="float:right">[source]</a></h4>

> <code>FileSplitter</code>(**`fname`**)

Split `items` by providing file `fname` (contains names of valid items separated by newline).


```python
with tempfile.TemporaryDirectory() as d:
    fname = Path(d)/'valid.txt'
    fname.write('\n'.join([Path(fnames[i]).name for i in [1,3,4]]))
    splitter = FileSplitter(fname)
    test_eq(splitter(fnames),[[0,2,5,6,7],[1,3,4]])
```


<h4 id="ColSplitter" class="doc_header"><code>ColSplitter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L145" class="source_link" style="float:right">[source]</a></h4>

> <code>ColSplitter</code>(**`col`**=*`'is_valid'`*)

Split `items` (supposed to be a dataframe) by value in `col`


```python
df = pd.DataFrame({'a': [0,1,2,3,4], 'b': [True,False,True,True,False]})
splits = ColSplitter('b')(df)
test_eq(splits, [[1,4], [0,2,3]])
#Works with strings or index
splits = ColSplitter(1)(df)
test_eq(splits, [[1,4], [0,2,3]])
```


<h4 id="RandomSubsetSplitter" class="doc_header"><code>RandomSubsetSplitter</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L154" class="source_link" style="float:right">[source]</a></h4>

> <code>RandomSubsetSplitter</code>(**`train_sz`**, **`valid_sz`**, **`seed`**=*`None`*)

Take randoms subsets of `splits` with `train_sz` and `valid_sz`


```python
items = list(range(100))
valid_idx = list(np.arange(70,100))
splits = RandomSubsetSplitter(0.3, 0.1)(items)
test_eq(len(splits[0]), 30)
test_eq(len(splits[1]), 10)
```

### Label

The final set of functions is used to *label* a single item of data.


<h4 id="parent_label" class="doc_header"><code>parent_label</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L168" class="source_link" style="float:right">[source]</a></h4>

> <code>parent_label</code>(**`o`**)

Label `item` with the parent folder name.


Note that [`parent_label`](/data.transforms.html#parent_label) doesn't have anything customize, so it doesn't return a function - you can just use it directly.

```python
test_eq(parent_label(fnames[0]), '3')
test_eq(parent_label("fastai_dev/dev/data/mnist_tiny/train/3/9932.png"), '3')
[parent_label(o) for o in fnames]
```




    ['3', '7', '7', '7', '3', '3', '7', '3']




<h2 id="RegexLabeller" class="doc_header"><code>class</code> <code>RegexLabeller</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L173" class="source_link" style="float:right">[source]</a></h2>

> <code>RegexLabeller</code>(**`pat`**, **`match`**=*`False`*)

Label `item` with regex `pat`.


[`RegexLabeller`](/data.transforms.html#RegexLabeller) is a very flexible function since it handles any regex search of the stringified item. Pass `match=True` to use `re.match` (i.e. check only start of string), or `re.search` otherwise (default).

For instance, here's an example the replicates the previous [`parent_label`](/data.transforms.html#parent_label) results.

```python
f = RegexLabeller(fr'{os.path.sep}(\d){os.path.sep}')
test_eq(f(fnames[0]), '3')
[f(o) for o in fnames]
```




    ['3', '7', '7', '7', '3', '3', '7', '3']



```python
f = RegexLabeller(r'(\d*)', match=True)
test_eq(f(fnames[0].name), '9932')
```


<h2 id="ColReader" class="doc_header"><code>class</code> <code>ColReader</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L185" class="source_link" style="float:right">[source]</a></h2>

> <code>ColReader</code>(**`cols`**, **`pref`**=*`''`*, **`suff`**=*`''`*, **`label_delim`**=*`None`*) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

Read `cols` in `row` with potential `pref` and `suff`


`cols` can be a list of column names or a list of indices (or a mix of both). If `label_delim` is passed, the result is split using it.

```python
df = pd.DataFrame({'a': 'a b c d'.split(), 'b': ['1 2', '0', '', '1 2 3']})
f = ColReader('a', pref='0', suff='1')
test_eq([f(o) for o in df.itertuples()], '0a1 0b1 0c1 0d1'.split())

f = ColReader('b', label_delim=' ')
test_eq([f(o) for o in df.itertuples()], [['1', '2'], ['0'], [], ['1', '2', '3']])

df['a1'] = df['a']
f = ColReader(['a', 'a1'], pref='0', suff='1')
test_eq([f(o) for o in df.itertuples()], [L('0a1', '0a1'), L('0b1', '0b1'), L('0c1', '0c1'), L('0d1', '0d1')])

df = pd.DataFrame({'a': [L(0,1), L(2,3,4), L(5,6,7)]})
f = ColReader('a')
test_eq([f(o) for o in df.itertuples()], [L(0,1), L(2,3,4), L(5,6,7)])

df['name'] = df['a']
f = ColReader('name')
test_eq([f(df.iloc[0,:])], [L(0,1)])
```


<h2 id="CategoryMap" class="doc_header"><code>class</code> <code>CategoryMap</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L204" class="source_link" style="float:right">[source]</a></h2>

> <code>CategoryMap</code>(**`col`**, **`sort`**=*`True`*, **`add_na`**=*`False`*, **`strict`**=*`False`*) :: [`CollBase`](https://fastcore.fast.ai/foundation#CollBase)

Collection of categories with the reverse mapping in `o2i`


```python
t = CategoryMap([4,2,3,4])
test_eq(t, [2,3,4])
test_eq(t.o2i, {2:0,3:1,4:2})
test_eq(t.map_objs([2,3]), [0,1])
test_eq(t.map_ids([0,1]), [2,3])
test_fail(lambda: t.o2i['unseen label'])
```

```python
t = CategoryMap([4,2,3,4], add_na=True)
test_eq(t, ['#na#',2,3,4])
test_eq(t.o2i, {'#na#':0,2:1,3:2,4:3})
```

```python
t = CategoryMap(pd.Series([4,2,3,4]), sort=False)
test_eq(t, [4,2,3])
test_eq(t.o2i, {4:0,2:1,3:2})
```

```python
col = pd.Series(pd.Categorical(['M','H','L','M'], categories=['H','M','L'], ordered=True))
t = CategoryMap(col)
test_eq(t, ['H','M','L'])
test_eq(t.o2i, {'H':0,'M':1,'L':2})
```

```python
col = pd.Series(pd.Categorical(['M','H','M'], categories=['H','M','L'], ordered=True))
t = CategoryMap(col, strict=True)
test_eq(t, ['H','M'])
test_eq(t.o2i, {'H':0,'M':1})
```


<h2 id="Categorize" class="doc_header"><code>class</code> <code>Categorize</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L230" class="source_link" style="float:right">[source]</a></h2>

> <code>Categorize</code>(**`vocab`**=*`None`*, **`sort`**=*`True`*, **`add_na`**=*`False`*) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

Reversible transform of category string to `vocab` id



<h2 id="Category" class="doc_header"><code>class</code> <code>Category</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L245" class="source_link" style="float:right">[source]</a></h2>

> <code>Category</code>() :: `str`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


```python
cat = Categorize()
tds = Datasets(['cat', 'dog', 'cat'], tfms=[cat])
test_eq(cat.vocab, ['cat', 'dog'])
test_eq(cat('cat'), 0)
test_eq(cat.decode(1), 'dog')
test_stdout(lambda: show_at(tds,2), 'cat')
```

```python
cat = Categorize(add_na=True)
tds = Datasets(['cat', 'dog', 'cat'], tfms=[cat])
test_eq(cat.vocab, ['#na#', 'cat', 'dog'])
test_eq(cat('cat'), 1)
test_eq(cat.decode(2), 'dog')
test_stdout(lambda: show_at(tds,2), 'cat')
```

```python
cat = Categorize(vocab=['dog', 'cat'], sort=False, add_na=True)
tds = Datasets(['cat', 'dog', 'cat'], tfms=[cat])
test_eq(cat.vocab, ['#na#', 'dog', 'cat'])
test_eq(cat('dog'), 1)
test_eq(cat.decode(2), 'cat')
test_stdout(lambda: show_at(tds,2), 'cat')
```


<h2 id="MultiCategorize" class="doc_header"><code>class</code> <code>MultiCategorize</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L248" class="source_link" style="float:right">[source]</a></h2>

> <code>MultiCategorize</code>(**`vocab`**=*`None`*, **`add_na`**=*`False`*) :: [`Categorize`](/data.transforms.html#Categorize)

Reversible transform of multi-category strings to `vocab` id



<h2 id="MultiCategory" class="doc_header"><code>class</code> <code>MultiCategory</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L264" class="source_link" style="float:right">[source]</a></h2>

> <code>MultiCategory</code>(**`items`**=*`None`*, **\*`rest`**, **`use_list`**=*`False`*, **`match`**=*`None`*) :: [`L`](https://fastcore.fast.ai/foundation#L)

Behaves like a list of `items` but can also index with list of indices or masks


```python
cat = MultiCategorize()
tds = Datasets([['b', 'c'], ['a'], ['a', 'c'], []], tfms=[cat])
test_eq(tds[3][0], TensorMultiCategory([]))
test_eq(cat.vocab, ['a', 'b', 'c'])
test_eq(cat(['a', 'c']), tensor([0,2]))
test_eq(cat([]), tensor([]))
test_eq(cat.decode([1]), ['b'])
test_eq(cat.decode([0,2]), ['a', 'c'])
test_stdout(lambda: show_at(tds,2), 'a;c')
```


<h2 id="OneHotEncode" class="doc_header"><code>class</code> <code>OneHotEncode</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L269" class="source_link" style="float:right">[source]</a></h2>

> <code>OneHotEncode</code>(**`c`**=*`None`*) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

One-hot encodes targets


Works in conjunction with ` MultiCategorize` or on its own if you have one-hot encoded targets (pass a `vocab` for decoding and `do_encode=False` in this case)

```python
_tfm = OneHotEncode(c=3)
test_eq(_tfm([0,2]), tensor([1.,0,1]))
test_eq(_tfm.decode(tensor([0,1,1])), [1,2])
```

```python
tds = Datasets([['b', 'c'], ['a'], ['a', 'c'], []], [[MultiCategorize(), OneHotEncode()]])
test_eq(tds[1], [tensor([1.,0,0])])
test_eq(tds[3], [tensor([0.,0,0])])
test_eq(tds.decode([tensor([False, True, True])]), [['b','c']])
test_eq(type(tds[1][0]), TensorMultiCategory)
test_stdout(lambda: show_at(tds,2), 'a;c')
```


<h2 id="EncodedMultiCategorize" class="doc_header"><code>class</code> <code>EncodedMultiCategorize</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L283" class="source_link" style="float:right">[source]</a></h2>

> <code>EncodedMultiCategorize</code>(**`vocab`**) :: [`Categorize`](/data.transforms.html#Categorize)

Transform of one-hot encoded multi-category that decodes with `vocab`


```python
_tfm = EncodedMultiCategorize(vocab=['a', 'b', 'c'])
test_eq(_tfm([1,0,1]), tensor([1., 0., 1.]))
test_eq(type(_tfm([1,0,1])), TensorMultiCategory)
test_eq(_tfm.decode(tensor([False, True, True])), ['b','c'])
```

```python
_tfm
```




    EncodedMultiCategorize -- {'vocab': (#3) ['a','b','c'], 'add_na': False}:
    encodes: (object,object) -> encodes
    (object,object) -> encodes
    decodes: (object,object) -> decodes
    (object,object) -> decodes




<h2 id="RegressionSetup" class="doc_header"><code>class</code> <code>RegressionSetup</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L293" class="source_link" style="float:right">[source]</a></h2>

> <code>RegressionSetup</code>(**`c`**=*`None`*) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

Transform that floatifies targets


```python
_tfm = RegressionSetup()
dsets = Datasets([0, 1, 2], RegressionSetup)
test_eq(dsets.c, 1)
test_eq_type(dsets[0], (tensor(0.),))

dsets = Datasets([[0, 1, 2], [3,4,5]], RegressionSetup)
test_eq(dsets.c, 3)
test_eq_type(dsets[0], (tensor([0.,1.,2.]),))
```


<h4 id="get_c" class="doc_header"><code>get_c</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L307" class="source_link" style="float:right">[source]</a></h4>

> <code>get_c</code>(**`dls`**)




## End-to-end dataset example with MNIST

Let's show how to use those functions to grab the mnist dataset in a [`Datasets`](/data.core.html#Datasets). First we grab all the images.

```python
path = untar_data(URLs.MNIST_TINY)
items = get_image_files(path)
```

Then we split between train and validation depending on the folder.

```python
splitter = GrandparentSplitter()
splits = splitter(items)
train,valid = (items[i] for i in splits)
train[:3],valid[:3]
```




    ((#3) [Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/723.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/7446.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/train/7/8566.png')],
     (#3) [Path('/home/jhoward/.fastai/data/mnist_tiny/valid/7/946.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/valid/7/9608.png'),Path('/home/jhoward/.fastai/data/mnist_tiny/valid/7/825.png')])



Our inputs are images that we open and convert to tensors, our targets are labeled depending on the parent directory and are categories.

```python
from PIL import Image
```

```python
def open_img(fn:Path): return Image.open(fn).copy()
def img2tensor(im:Image.Image): return TensorImage(array(im)[None])

tfms = [[open_img, img2tensor],
        [parent_label, Categorize()]]
train_ds = Datasets(train, tfms)
```

```python
x,y = train_ds[3]
xd,yd = decode_at(train_ds,3)
test_eq(parent_label(train[3]),yd)
test_eq(array(Image.open(train[3])),xd[0].numpy())
```

```python
ax = show_at(train_ds, 3, cmap="Greys", figsize=(1,1))
```


![png](output_122_0.png)


```python
assert ax.title.get_text() in ('3','7')
test_fig_exists(ax)
```


<h2 id="ToTensor" class="doc_header"><code>class</code> <code>ToTensor</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L316" class="source_link" style="float:right">[source]</a></h2>

> <code>ToTensor</code>(**`enc`**=*`None`*, **`dec`**=*`None`*, **`split_idx`**=*`None`*, **`order`**=*`None`*) :: [`Transform`](https://fastcore.fast.ai/transform#Transform)

Convert item to appropriate tensor class



<h2 id="IntToFloatTensor" class="doc_header"><code>class</code> <code>IntToFloatTensor</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L321" class="source_link" style="float:right">[source]</a></h2>

> <code>IntToFloatTensor</code>(**`div`**=*`255.0`*, **`div_mask`**=*`1`*) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

Transform image to float tensor, optionally dividing by 255 (e.g. for images).


```python
t = (TensorImage(tensor(1)),tensor(2).long(),TensorMask(tensor(3)))
tfm = IntToFloatTensor()
ft = tfm(t)
test_eq(ft, [1./255, 2, 3])
test_eq(type(ft[0]), TensorImage)
test_eq(type(ft[2]), TensorMask)
test_eq(ft[0].type(),'torch.FloatTensor')
test_eq(ft[1].type(),'torch.LongTensor')
test_eq(ft[2].type(),'torch.LongTensor')
```


<h4 id="broadcast_vec" class="doc_header"><code>broadcast_vec</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L331" class="source_link" style="float:right">[source]</a></h4>

> <code>broadcast_vec</code>(**`dim`**, **`ndim`**, **\*`t`**, **`cuda`**=*`True`*)

Make a vector broadcastable over `dim` (out of `ndim` total) by prepending and appending unit axes



<h2 id="Normalize" class="doc_header"><code>class</code> <code>Normalize</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/transforms.py#L340" class="source_link" style="float:right">[source]</a></h2>

> <code>Normalize</code>(**`mean`**=*`None`*, **`std`**=*`None`*, **`axes`**=*`(0, 2, 3)`*) :: [`DisplayedTransform`](https://fastcore.fast.ai/transform#DisplayedTransform)

Normalize/denorm batch of [`TensorImage`](/torch_core.html#TensorImage)


```python
mean,std = [0.5]*3,[0.5]*3
mean,std = broadcast_vec(1, 4, mean, std)
batch_tfms = [IntToFloatTensor(), Normalize.from_stats(mean,std)]
tdl = TfmdDL(train_ds, after_batch=batch_tfms, bs=4, device=default_device())
```

```python
x,y  = tdl.one_batch()
xd,yd = tdl.decode((x,y))

test_eq(x.type(), 'torch.cuda.FloatTensor' if default_device().type=='cuda' else 'torch.FloatTensor')
test_eq(xd.type(), 'torch.LongTensor')
test_eq(type(x), TensorImage)
test_eq(type(y), TensorCategory)
assert x.mean()<0.0
assert x.std()>0.5
assert 0<xd.float().mean()/255.<1
assert 0<xd.float().std()/255.<0.5
```

```python
from fastai.vision.core import *
```

```python
tdl.show_batch((x,y))
```


![png](output_136_0.png)



![png](output_136_1.png)



![png](output_136_2.png)



![png](output_136_3.png)

