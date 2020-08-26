# Text data
> Functions and transforms to help gather text data in a `Datasets`


## Backwards

Reversing the text can provide higher accuracy with an ensemble with a forward model. All that is needed is a `type_tfm` that will reverse the text as it is brought in:


<h4 id="reverse_text" class="doc_header"><code>reverse_text</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L12" class="source_link" style="float:right">[source]</a></h4>

> <code>reverse_text</code>(**`x`**)




```python
t = tensor([0,1,2])
r = reverse_text(t)
test_eq(r, tensor([2,1,0]))
```

## Numericalizing

Numericalization is the step in which we convert tokens to integers. The first step is to build a correspondence token to index that is called a vocab.


<h4 id="make_vocab" class="doc_header"><code>make_vocab</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L15" class="source_link" style="float:right">[source]</a></h4>

> <code>make_vocab</code>(**`count`**, **`min_freq`**=*`3`*, **`max_vocab`**=*`60000`*, **`special_toks`**=*`None`*)

Create a vocab of `max_vocab` size from `Counter` `count` with items present more than `min_freq`


If there are more than `max_vocab` tokens, the ones kept are the most frequent.
{% include note.html content='For performance when using mixed precision, the vocabulary is always made of size a multiple of 8, potentially by adding `xxfake` tokens.' %}

```python
count = Counter(['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'd'])
test_eq(set([x for x in make_vocab(count) if not x.startswith('xxfake')]), 
        set(defaults.text_spec_tok + 'a'.split()))
test_eq(len(make_vocab(count))%8, 0)
test_eq(set([x for x in make_vocab(count, min_freq=1) if not x.startswith('xxfake')]), 
        set(defaults.text_spec_tok + 'a b c d'.split()))
test_eq(set([x for x in make_vocab(count,max_vocab=12, min_freq=1) if not x.startswith('xxfake')]), 
        set(defaults.text_spec_tok + 'a b c'.split()))
```


<h3 id="TensorText" class="doc_header"><code>class</code> <code>TensorText</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L26" class="source_link" style="float:right">[source]</a></h3>

> <code>TensorText</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorBase`](/torch_core.html#TensorBase)

Semantic type for a tensor representing text



<h3 id="LMTensorText" class="doc_header"><code>class</code> <code>LMTensorText</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L27" class="source_link" style="float:right">[source]</a></h3>

> <code>LMTensorText</code>(**`x`**, **\*\*`kwargs`**) :: [`TensorText`](/text.data.html#TensorText)

Semantic type for a tensor representing text in language modeling



<h3 id="Numericalize" class="doc_header"><code>class</code> <code>Numericalize</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L33" class="source_link" style="float:right">[source]</a></h3>

> <code>Numericalize</code>(**`vocab`**=*`None`*, **`min_freq`**=*`3`*, **`max_vocab`**=*`60000`*, **`special_toks`**=*`None`*, **`pad_tok`**=*`None`*) :: [`Transform`](https://fastcore.fast.ai/transform#Transform)

Reversible transform of tokenized texts to numericalized ids


If no `vocab` is passed, one is created at setup from the data, using [`make_vocab`](/text.data.html#make_vocab) with `min_freq` and `max_vocab`.

```python
start = 'This is an example of text'
num = Numericalize(min_freq=1)
num.setup(L(start.split(), 'this is another text'.split()))
test_eq(set([x for x in num.vocab if not x.startswith('xxfake')]), 
        set(defaults.text_spec_tok + 'This is an example of text this another'.split()))
test_eq(len(num.vocab)%8, 0)
t = num(start.split())

test_eq(t, tensor([11, 9, 12, 13, 14, 10]))
test_eq(num.decode(t), start.split())
```

```python
num = Numericalize(min_freq=2)
num.setup(L('This is an example of text'.split(), 'this is another text'.split()))
test_eq(set([x for x in num.vocab if not x.startswith('xxfake')]), 
        set(defaults.text_spec_tok + 'is text'.split()))
test_eq(len(num.vocab)%8, 0)
t = num(start.split())
test_eq(t, tensor([0, 9, 0, 0, 0, 10]))
test_eq(num.decode(t), f'{UNK} is {UNK} {UNK} {UNK} text'.split())
```


<h2 id="LMDataLoader" class="doc_header"><code>class</code> <code>LMDataLoader</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L72" class="source_link" style="float:right">[source]</a></h2>

> <code>LMDataLoader</code>(**`dataset`**, **`lens`**=*`None`*, **`cache`**=*`2`*, **`bs`**=*`64`*, **`seq_len`**=*`72`*, **`num_workers`**=*`0`*, **`shuffle`**=*`False`*, **`verbose`**=*`False`*, **`do_setup`**=*`True`*, **`pin_memory`**=*`False`*, **`timeout`**=*`0`*, **`batch_size`**=*`None`*, **`drop_last`**=*`False`*, **`indexed`**=*`None`*, **`n`**=*`None`*, **`device`**=*`None`*, **`wif`**=*`None`*, **`before_iter`**=*`None`*, **`after_item`**=*`None`*, **`before_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_iter`**=*`None`*, **`create_batches`**=*`None`*, **`create_item`**=*`None`*, **`create_batch`**=*`None`*, **`retain`**=*`None`*, **`get_idxs`**=*`None`*, **`sample`**=*`None`*, **`shuffle_fn`**=*`None`*, **`do_batch`**=*`None`*) :: [`TfmdDL`](/data.core.html#TfmdDL)

A [`DataLoader`](/data.load.html#DataLoader) suitable for language modeling


`dataset` should be a collection of numericalized texts for this to work. `lens` can be passed for optimizing the creation, otherwise, the [`LMDataLoader`](/text.data.html#LMDataLoader) will do a full pass of the `dataset` to compute them. `cache` is used to avoid reloading items unnecessarily.

The [`LMDataLoader`](/text.data.html#LMDataLoader) will concatenate all texts (maybe `shuffle`d) in one big stream, split it in `bs` contiguous sentences, then go through those `seq_len` at a time.

```python
bs,sl = 4,3
ints = L([0,1,2,3,4],[5,6,7,8,9,10],[11,12,13,14,15,16,17,18],[19,20],[21,22,23],[24]).map(tensor)
```

```python
dl = LMDataLoader(ints, bs=bs, seq_len=sl)
test_eq(list(dl),
    [[tensor([[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]]),
      tensor([[1, 2, 3], [7, 8, 9], [13, 14, 15], [19, 20, 21]])],
     [tensor([[3, 4, 5], [ 9, 10, 11], [15, 16, 17], [21, 22, 23]]),
      tensor([[4, 5, 6], [10, 11, 12], [16, 17, 18], [22, 23, 24]])]])
```

```python
dl = LMDataLoader(ints, bs=bs, seq_len=sl, shuffle=True)
for x,y in dl: test_eq(x[:,1:], y[:,:-1])
((x0,y0), (x1,y1)) = tuple(dl)
#Second batch begins where first batch ended
test_eq(y0[:,-1], x1[:,0]) 
test_eq(type(x0), LMTensorText)
```

## Classification

For classification, we deal with the fact that texts don't all have the same length by using padding.


<h4 id="pad_input" class="doc_header"><code>pad_input</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L124" class="source_link" style="float:right">[source]</a></h4>

> <code>pad_input</code>(**`samples`**, **`pad_idx`**=*`1`*, **`pad_fields`**=*`0`*, **`pad_first`**=*`False`*, **`backwards`**=*`False`*)

Function that collect `samples` and adds padding


`pad_idx` is used for the padding, and the padding is applied to the `pad_fields` of the samples. The padding is applied at the beginning if `pad_first` is `True`, and if `backwards` is added, the tensors are flipped.

```python
test_eq(pad_input([(tensor([1,2,3]),1), (tensor([4,5]), 2), (tensor([6]), 3)], pad_idx=0), 
        [(tensor([1,2,3]),1), (tensor([4,5,0]),2), (tensor([6,0,0]), 3)])
test_eq(pad_input([(tensor([1,2,3]), (tensor([6]))), (tensor([4,5]), tensor([4,5])), (tensor([6]), (tensor([1,2,3])))], pad_idx=0, pad_fields=1), 
        [(tensor([1,2,3]),(tensor([6,0,0]))), (tensor([4,5]),tensor([4,5,0])), ((tensor([6]),tensor([1, 2, 3])))])
test_eq(pad_input([(tensor([1,2,3]),1), (tensor([4,5]), 2), (tensor([6]), 3)], pad_idx=0, pad_first=True), 
        [(tensor([1,2,3]),1), (tensor([0,4,5]),2), (tensor([0,0,6]), 3)])
test_eq(pad_input([(tensor([1,2,3]),1), (tensor([4,5]), 2), (tensor([6]), 3)], pad_idx=0, backwards=True), 
        [(tensor([3,2,1]),1), (tensor([5,4,0]),2), (tensor([6,0,0]), 3)])
x = test_eq(pad_input([(tensor([1,2,3]),1), (tensor([4,5]), 2), (tensor([6]), 3)], pad_idx=0, backwards=True), 
        [(tensor([3,2,1]),1), (tensor([5,4,0]),2), (tensor([6,0,0]), 3)])
```


<h4 id="pad_input_chunk" class="doc_header"><code>pad_input_chunk</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L140" class="source_link" style="float:right">[source]</a></h4>

> <code>pad_input_chunk</code>(**`samples`**, **`pad_idx`**=*`1`*, **`pad_first`**=*`True`*, **`seq_len`**=*`72`*)

Pad `samples` by adding padding by chunks of size `seq_len`


The difference with the base [`pad_input`](/text.data.html#pad_input) is that most of the padding is applied first (if `pad_first=True`) or at the end (if `pad_first=False`) but only by a round multiple of `seq_len`. The rest of the padding is applied to the end (or the beginning if `pad_first=False`). This is to work with `SequenceEncoder` with recurrent models.

```python
test_eq(pad_input_chunk([(tensor([1,2,3,4,5,6]),1), (tensor([1,2,3]), 2), (tensor([1,2]), 3)], pad_idx=0, seq_len=2), 
        [(tensor([1,2,3,4,5,6]),1), (tensor([0,0,1,2,3,0]),2), (tensor([0,0,0,0,1,2]), 3)])
test_eq(pad_input_chunk([(tensor([1,2,3,4,5,6]),), (tensor([1,2,3]),), (tensor([1,2]),)], pad_idx=0, seq_len=2), 
        [(tensor([1,2,3,4,5,6]),), (tensor([0,0,1,2,3,0]),), (tensor([0,0,0,0,1,2]),)])
test_eq(pad_input_chunk([(tensor([1,2,3,4,5,6]),), (tensor([1,2,3]),), (tensor([1,2]),)], pad_idx=0, seq_len=2, pad_first=False), 
        [(tensor([1,2,3,4,5,6]),), (tensor([1,2,3,0,0,0]),), (tensor([1,2,0,0,0,0]),)])
```


<h3 id="SortedDL" class="doc_header"><code>class</code> <code>SortedDL</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L155" class="source_link" style="float:right">[source]</a></h3>

> <code>SortedDL</code>(**`dataset`**, **`sort_func`**=*`None`*, **`res`**=*`None`*, **`bs`**=*`64`*, **`shuffle`**=*`False`*, **`num_workers`**=*`None`*, **`verbose`**=*`False`*, **`do_setup`**=*`True`*, **`pin_memory`**=*`False`*, **`timeout`**=*`0`*, **`batch_size`**=*`None`*, **`drop_last`**=*`False`*, **`indexed`**=*`None`*, **`n`**=*`None`*, **`device`**=*`None`*, **`wif`**=*`None`*, **`before_iter`**=*`None`*, **`after_item`**=*`None`*, **`before_batch`**=*`None`*, **`after_batch`**=*`None`*, **`after_iter`**=*`None`*, **`create_batches`**=*`None`*, **`create_item`**=*`None`*, **`create_batch`**=*`None`*, **`retain`**=*`None`*, **`get_idxs`**=*`None`*, **`sample`**=*`None`*, **`shuffle_fn`**=*`None`*, **`do_batch`**=*`None`*) :: [`TfmdDL`](/data.core.html#TfmdDL)

A [`DataLoader`](/data.load.html#DataLoader) that goes throught the item in the order given by `sort_func`


`res` is the result of `sort_func` applied on all elements of the `dataset`. You can pass it if available to make the init much faster by avoiding an initial pass over the whole dataset. For example if sorting by text length (as in the default `sort_func`, called `_default_sort`) you should pass a list with the length of each element in `dataset` to `res` to take advantage of this speed-up. 

To get the same init speed-up for the validation set, `val_res` (a list of text lengths for your validation set) can be passed to the `kwargs` argument of [`SortedDL`](/text.data.html#SortedDL). Below is an example to reduce the init time by passing a list of text lengths for both the training set and the validation set:

```
# Pass the training dataset text lengths to SortedDL
srtd_dl=partial(SortedDL, res = train_text_lens)

# Pass the validation dataset text lengths 
dl_kwargs = [{},{'val_res': val_text_lens}]

# init our Datasets 
dsets = Datasets(...)   

# init our Dataloaders
dls = dsets.dataloaders(...,dl_type = srtd_dl, dl_kwargs = dl_kwargs)
```

If `shuffle` is `True`, this will shuffle a bit the results of the sort to have items of roughly the same size in batches, but not in the exact sorted order.

```python
ds = [(tensor([1,2]),1), (tensor([3,4,5,6]),2), (tensor([7]),3), (tensor([8,9,10]),4)]
dl = SortedDL(ds, bs=2, before_batch=partial(pad_input, pad_idx=0))
test_eq(list(dl), [(tensor([[ 3,  4,  5,  6], [ 8,  9, 10,  0]]), tensor([2, 4])), 
                   (tensor([[1, 2], [7, 0]]), tensor([1, 3]))])
```

```python
ds = [(tensor(range(random.randint(1,10))),i) for i in range(101)]
dl = SortedDL(ds, bs=2, create_batch=partial(pad_input, pad_idx=-1), shuffle=True, num_workers=0)
batches = list(dl)
max_len = len(batches[0][0])
for b in batches: 
    assert(len(b[0])) <= max_len 
    test_ne(b[0][-1], -1)
```

## TransformBlock for text

To use the data block API, you will need this build block for texts.


<h3 id="TextBlock" class="doc_header"><code>class</code> <code>TextBlock</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L191" class="source_link" style="float:right">[source]</a></h3>

> <code>TextBlock</code>(**`tok_tfm`**, **`vocab`**=*`None`*, **`is_lm`**=*`False`*, **`seq_len`**=*`72`*, **`backwards`**=*`False`*, **`min_freq`**=*`3`*, **`max_vocab`**=*`60000`*, **`special_toks`**=*`None`*, **`pad_tok`**=*`None`*) :: [`TransformBlock`](/data.block.html#TransformBlock)

A [`TransformBlock`](/data.block.html#TransformBlock) for texts


For efficient tokenization, you probably want to use one of the factory methods. Otherwise, you can pass your custom `tok_tfm` that will deal with tokenization (if your texts are already tokenized, you can pass `noop`), a `vocab`, or leave it to be inferred on the texts using `min_freq` and `max_vocab`.

`is_lm` indicates if we want to use texts for language modeling or another task, `seq_len` is only necessary to tune if `is_lm=False`, and is passed along to [`pad_input_chunk`](/text.data.html#pad_input_chunk).


<h4 id="TextBlock.from_df" class="doc_header"><code>TextBlock.from_df</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L201" class="source_link" style="float:right">[source]</a></h4>

> <code>TextBlock.from_df</code>(**`text_cols`**, **`vocab`**=*`None`*, **`is_lm`**=*`False`*, **`seq_len`**=*`72`*, **`backwards`**=*`False`*, **`min_freq`**=*`3`*, **`max_vocab`**=*`60000`*, **`tok`**=*`None`*, **`rules`**=*`None`*, **`sep`**=*`' '`*, **`n_workers`**=*`4`*, **`mark_fields`**=*`None`*, **`res_col_name`**=*`'text'`*, **\*\*`kwargs`**)

Build a [`TextBlock`](/text.data.html#TextBlock) from a dataframe using `text_cols`


Here is an example using a sample of IMDB stored as a CSV file:

```python
path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/'texts.csv')

imdb_clas = DataBlock(
    blocks=(TextBlock.from_df('text', seq_len=72), CategoryBlock),
    get_x=ColReader('text'), get_y=ColReader('label'), splitter=ColSplitter())

dls = imdb_clas.dataloaders(df, bs=64)
dls.show_batch(max_n=2)
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
  </tbody>
</table>
</div>



`vocab`,  `is_lm`, `seq_len`, `min_freq` and `max_vocab` are passed to the main init, the other argument to [`Tokenizer.from_df`](/text.core.html#Tokenizer.from_df).


<h4 id="TextBlock.from_folder" class="doc_header"><code>TextBlock.from_folder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L208" class="source_link" style="float:right">[source]</a></h4>

> <code>TextBlock.from_folder</code>(**`path`**, **`vocab`**=*`None`*, **`is_lm`**=*`False`*, **`seq_len`**=*`72`*, **`backwards`**=*`False`*, **`min_freq`**=*`3`*, **`max_vocab`**=*`60000`*, **`tok`**=*`None`*, **`rules`**=*`None`*, **`extensions`**=*`None`*, **`folders`**=*`None`*, **`output_dir`**=*`None`*, **`skip_if_exists`**=*`True`*, **`output_names`**=*`None`*, **`n_workers`**=*`4`*, **`encoding`**=*`'utf8'`*, **\*\*`kwargs`**)

Build a [`TextBlock`](/text.data.html#TextBlock) from a `path`


`vocab`, `is_lm`, `seq_len`, `min_freq` and `max_vocab` are passed to the main init, the other argument to [`Tokenizer.from_folder`](/text.core.html#Tokenizer.from_folder).


<h2 id="TextDataLoaders" class="doc_header"><code>class</code> <code>TextDataLoaders</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L216" class="source_link" style="float:right">[source]</a></h2>

> <code>TextDataLoaders</code>(**\*`loaders`**, **`path`**=*`'.'`*, **`device`**=*`None`*) :: [`DataLoaders`](/data.core.html#DataLoaders)

Basic wrapper around several [`DataLoader`](/data.load.html#DataLoader)s with factory methods for NLP problems


You should not use the init directly but one of the following factory methods. All those factory methods accept as arguments:

- `text_vocab`: the vocabulary used for numericalizing texts (if not passed, it's inferred from the data)
- `tok_tfm`: if passed, uses this `tok_tfm` instead of the default
- `seq_len`: the sequence length used for batch
- `bs`: the batch size
- `val_bs`: the batch size for the validation [`DataLoader`](/data.load.html#DataLoader) (defaults to `bs`)
- `shuffle_train`: if we shuffle the training [`DataLoader`](/data.load.html#DataLoader) or not
- `device`: the PyTorch device to use (defaults to `default_device()`)


<h4 id="TextDataLoaders.from_folder" class="doc_header"><code>TextDataLoaders.from_folder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L218" class="source_link" style="float:right">[source]</a></h4>

> <code>TextDataLoaders.from_folder</code>(**`path`**, **`train`**=*`'train'`*, **`valid`**=*`'valid'`*, **`valid_pct`**=*`None`*, **`seed`**=*`None`*, **`vocab`**=*`None`*, **`text_vocab`**=*`None`*, **`is_lm`**=*`False`*, **`tok_tfm`**=*`None`*, **`seq_len`**=*`72`*, **`backwards`**=*`False`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from imagenet style dataset in `path` with `train` and `valid` subfolders (or provide `valid_pct`)


If `valid_pct` is provided, a random split is performed (with an optional `seed`) by setting aside that percentage of the data for the validation set (instead of looking at the grandparents folder). If a `vocab` is passed, only the folders with names in `vocab` are kept.

Here is an example on a sample of the IMDB movie review dataset:

```python
path = untar_data(URLs.IMDB)
dls = TextDataLoaders.from_folder(path)
dls.show_batch(max_n=3)
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
      <td>▁xxbos ▁xxmaj ▁match ▁1: ▁xxmaj ▁tag ▁xxmaj ▁team ▁xxmaj ▁table ▁xxmaj ▁match ▁xxmaj ▁ bub ba ▁xxmaj ▁ray ▁and ▁xxmaj ▁spike ▁xxmaj ▁dudley ▁vs ▁xxmaj ▁eddie ▁xxmaj ▁guerrero ▁and ▁xxmaj ▁chris ▁xxmaj ▁benoit ▁xxmaj ▁ bub ba ▁xxmaj ▁ray ▁and ▁xxmaj ▁spike ▁xxmaj ▁dudley ▁started ▁things ▁off ▁with ▁a ▁xxmaj ▁tag ▁xxmaj ▁team ▁xxmaj ▁table ▁xxmaj ▁match ▁against ▁xxmaj ▁eddie ▁xxmaj ▁guerrero ▁and ▁xxmaj ▁chris ▁xxmaj ▁benoit . ▁xxmaj ▁according ▁to ▁the ▁rules ▁of ▁the ▁match , ▁both ▁opponents ▁have ▁to ▁go ▁through ▁tables ▁in ▁order ▁to ▁get ▁the ▁win . ▁xxmaj ▁benoit ▁and ▁xxmaj ▁guerrero ▁heated ▁up ▁early ▁on ▁by ▁taking ▁turns ▁hammer ing ▁first ▁xxmaj ▁spike ▁and ▁then ▁xxmaj ▁ bub ba ▁xxmaj ▁ray . ▁a ▁xxmaj ▁german ▁su plex ▁by ▁xxmaj ▁benoit ▁to ▁xxmaj ▁ bub ba ▁took ▁the ▁wind ▁out ▁of ▁the ▁xxmaj ▁dudley ▁brother . ▁xxmaj ▁spike ▁tried ▁to ▁help ▁his ▁brother , ▁but ▁the</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad xxpad</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>



<h4 id="TextDataLoaders.from_df" class="doc_header"><code>TextDataLoaders.from_df</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L233" class="source_link" style="float:right">[source]</a></h4>

> <code>TextDataLoaders.from_df</code>(**`df`**, **`path`**=*`'.'`*, **`valid_pct`**=*`0.2`*, **`seed`**=*`None`*, **`text_col`**=*`0`*, **`label_col`**=*`1`*, **`label_delim`**=*`None`*, **`y_block`**=*`None`*, **`text_vocab`**=*`None`*, **`is_lm`**=*`False`*, **`valid_col`**=*`None`*, **`tok_tfm`**=*`None`*, **`seq_len`**=*`72`*, **`backwards`**=*`False`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from `df` in `path` with `valid_pct`


`seed` can optionally be passed for reproducibility. `text_col`, `label_col` and optionally `valid_col` are indices or names of columns for texts/labels and the validation flag. `label_delim` can be passed for a multi-label problem if your labels are in one column, separated by a particular char. `y_block` should be passed to indicate your type of targets, in case the library did no infer it properly.

Here are examples on subsets of IMDB:

```python
dls = TextDataLoaders.from_df(df, path=path, text_col='text', label_col='label', valid_col='is_valid')
dls.show_batch(max_n=3)
```

```python
dls = TextDataLoaders.from_df(df, path=path, text_col='text', is_lm=True, valid_col='is_valid')
dls.show_batch(max_n=3)
```


<h4 id="TextDataLoaders.from_csv" class="doc_header"><code>TextDataLoaders.from_csv</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/data.py#L249" class="source_link" style="float:right">[source]</a></h4>

> <code>TextDataLoaders.from_csv</code>(**`path`**, **`csv_fname`**=*`'labels.csv'`*, **`header`**=*`'infer'`*, **`delimiter`**=*`None`*, **`valid_pct`**=*`0.2`*, **`seed`**=*`None`*, **`text_col`**=*`0`*, **`label_col`**=*`1`*, **`label_delim`**=*`None`*, **`y_block`**=*`None`*, **`text_vocab`**=*`None`*, **`is_lm`**=*`False`*, **`valid_col`**=*`None`*, **`tok_tfm`**=*`None`*, **`seq_len`**=*`72`*, **`backwards`**=*`False`*, **`bs`**=*`64`*, **`val_bs`**=*`None`*, **`shuffle_train`**=*`True`*, **`device`**=*`None`*)

Create from `csv` file in `path/csv_fname`


Opens the csv file with `header` and `delimiter`, then pass all the other arguments to [`TextDataLoaders.from_df`](/text.data.html#TextDataLoaders.from_df).

```python
dls = TextDataLoaders.from_csv(path=path, csv_fname='texts.csv', text_col='text', label_col='label', valid_col='is_valid')
dls.show_batch(max_n=3)
```
