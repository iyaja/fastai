# Text core
> Basic function to preprocess text before assembling it in a `DataLoaders`.


## Preprocessing rules

The following are rules applied to texts before or after it's tokenized.


<h4 id="spec_add_spaces" class="doc_header"><code>spec_add_spaces</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L28" class="source_link" style="float:right">[source]</a></h4>

> <code>spec_add_spaces</code>(**`t`**)

Add spaces around / and #


```python
test_eq(spec_add_spaces('#fastai'), ' # fastai')
test_eq(spec_add_spaces('/fastai'), ' / fastai')
test_eq(spec_add_spaces('\\fastai'), ' \\ fastai')
```


<h4 id="rm_useless_spaces" class="doc_header"><code>rm_useless_spaces</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L35" class="source_link" style="float:right">[source]</a></h4>

> <code>rm_useless_spaces</code>(**`t`**)

Remove multiple spaces


```python
test_eq(rm_useless_spaces('a  b   c'), 'a b c')
```


<h4 id="replace_rep" class="doc_header"><code>replace_rep</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L42" class="source_link" style="float:right">[source]</a></h4>

> <code>replace_rep</code>(**`t`**)

Replace repetitions at the character level: cccc -- TK_REP 4 c


It starts replacing at 3 repetitions of the same character or more.

```python
test_eq(replace_rep('aa'), 'aa')
test_eq(replace_rep('aaaa'), f' {TK_REP} 4 a ')
```


<h4 id="replace_wrep" class="doc_header"><code>replace_wrep</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L53" class="source_link" style="float:right">[source]</a></h4>

> <code>replace_wrep</code>(**`t`**)

Replace word repetitions: word word word word -- TK_WREP 4 word


It starts replacing at 3 repetitions of the same word or more.

```python
test_eq(replace_wrep('ah ah'), 'ah ah')
test_eq(replace_wrep('ah ah ah'), f' {TK_WREP} 3 ah ')
test_eq(replace_wrep('ah ah   ah  ah'), f' {TK_WREP} 4 ah ')
test_eq(replace_wrep('ah ah ah ah '), f' {TK_WREP} 4 ah  ')
test_eq(replace_wrep('ah ah ah ah.'), f' {TK_WREP} 4 ah .')
test_eq(replace_wrep('ah ah ahi'), f'ah ah ahi')
```


<h4 id="fix_html" class="doc_header"><code>fix_html</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L61" class="source_link" style="float:right">[source]</a></h4>

> <code>fix_html</code>(**`x`**)

Various messy things we've seen in documents


```python
test_eq(fix_html('#39;bli#146;'), "'bli'")
test_eq(fix_html('Sarah amp; Duck...'), 'Sarah & Duck …')
test_eq(fix_html('a nbsp; #36;'), 'a   $')
test_eq(fix_html('\\" <unk>'), f'" {UNK}')
test_eq(fix_html('quot;  @.@  @-@ '), "' .-")
test_eq(fix_html('<br />text\\n'), '\ntext\n')
```


<h4 id="replace_all_caps" class="doc_header"><code>replace_all_caps</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L72" class="source_link" style="float:right">[source]</a></h4>

> <code>replace_all_caps</code>(**`t`**)

Replace tokens in ALL CAPS by their lower version and add `TK_UP` before.


```python
test_eq(replace_all_caps("I'M SHOUTING"), f"{TK_UP} i'm {TK_UP} shouting")
test_eq(replace_all_caps("I'm speaking normally"), "I'm speaking normally")
test_eq(replace_all_caps("I am speaking normally"), "i am speaking normally")
```


<h4 id="replace_maj" class="doc_header"><code>replace_maj</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L83" class="source_link" style="float:right">[source]</a></h4>

> <code>replace_maj</code>(**`t`**)

Replace tokens in ALL CAPS by their lower version and add `TK_UP` before.


```python
test_eq(replace_maj("Jeremy Howard"), f'{TK_MAJ} jeremy {TK_MAJ} howard')
test_eq(replace_maj("I don't think there is any maj here"), ("i don't think there is any maj here"),)
```


<h4 id="lowercase" class="doc_header"><code>lowercase</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L91" class="source_link" style="float:right">[source]</a></h4>

> <code>lowercase</code>(**`t`**, **`add_bos`**=*`True`*, **`add_eos`**=*`False`*)

Converts `t` to lowercase



<h4 id="replace_space" class="doc_header"><code>replace_space</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L96" class="source_link" style="float:right">[source]</a></h4>

> <code>replace_space</code>(**`t`**)

Replace embedded spaces in a token with unicode line char to allow for split/join


## Tokenizing

A tokenizer is a class that must implement `__call__`. This method receives a iterator of texts and must return a generator with their tokenized versions. Here is the most basic example:


<h3 id="BaseTokenizer" class="doc_header"><code>class</code> <code>BaseTokenizer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L107" class="source_link" style="float:right">[source]</a></h3>

> <code>BaseTokenizer</code>(**`split_char`**=*`' '`*, **\*\*`kwargs`**)

Basic tokenizer that just splits on spaces


```python
tok = BaseTokenizer()
test_eq(tok(["This is a text"]), [["This", "is", "a", "text"]])
tok = BaseTokenizer('x')
test_eq(tok(["This is a text"]), [["This is a te", "t"]])
```


<h3 id="SpacyTokenizer" class="doc_header"><code>class</code> <code>SpacyTokenizer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L113" class="source_link" style="float:right">[source]</a></h3>

> <code>SpacyTokenizer</code>(**`lang`**=*`'en'`*, **`special_toks`**=*`None`*, **`buf_sz`**=*`5000`*)

Spacy tokenizer for `lang`


```python
tok = SpacyTokenizer()
inp,exp = "This isn't the easiest text.",["This", "is", "n't", "the", "easiest", "text", "."]
test_eq(L(tok([inp,inp])), [exp,exp])
```


<h3 id="TokenizeWithRules" class="doc_header"><code>class</code> <code>TokenizeWithRules</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L128" class="source_link" style="float:right">[source]</a></h3>

> <code>TokenizeWithRules</code>(**`tok`**, **`rules`**=*`None`*, **`post_rules`**=*`None`*)

A wrapper around `tok` which applies `rules`, then tokenizes, then applies `post_rules`


```python
f = TokenizeWithRules(BaseTokenizer(),rules=[replace_all_caps])
test_eq(f(["THIS isn't a problem"]), [[TK_UP, 'this', "isn't", 'a', 'problem']])
f = TokenizeWithRules(SpacyTokenizer())
test_eq(f(["This isn't a problem"]), [[BOS, TK_MAJ, 'this', 'is', "n't", 'a', 'problem']])
f = TokenizeWithRules(BaseTokenizer(split_char="'"), rules=[])
test_eq(f(["This isn't a problem"]), [['This▁isn', 't▁a▁problem']])
```

The main function that will be called during one of the processes handling tokenization. It will iterate through the `batch` of texts, apply them `rules` and tokenize them.

```python
texts = ["this is a text", "this is another text"]
tok = TokenizeWithRules(BaseTokenizer(), texts.__getitem__)
test_eq(tok([0,1]), [['this', 'is', 'a', 'text'],['this', 'is', 'another', 'text']])
```


<h4 id="tokenize1" class="doc_header"><code>tokenize1</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L139" class="source_link" style="float:right">[source]</a></h4>

> <code>tokenize1</code>(**`text`**, **`tok`**, **`rules`**=*`None`*, **`post_rules`**=*`None`*)

Call [`TokenizeWithRules`](/text.core.html#TokenizeWithRules) with a single text


```python
test_eq(tokenize1("This isn't a problem", SpacyTokenizer()),
        [BOS, TK_MAJ, 'this', 'is', "n't", 'a', 'problem'])
test_eq(tokenize1("This isn't a problem", tok=BaseTokenizer(), rules=[]),
        ['This',"isn't",'a','problem'])
```


<h4 id="parallel_tokenize" class="doc_header"><code>parallel_tokenize</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L145" class="source_link" style="float:right">[source]</a></h4>

> <code>parallel_tokenize</code>(**`items`**, **`tok`**=*`None`*, **`rules`**=*`None`*, **`n_workers`**=*`4`*, **\*\*`kwargs`**)

Calls optional `setup` on `tok` before launching [`TokenizeWithRules`](/text.core.html#TokenizeWithRules) using `parallel_gen


Note that since this uses [`parallel_gen`](https://fastcore.fast.ai/utils#parallel_gen) behind the scenes, the generator returned contains tuples of indices and results. There is no guarantee that the results are returned in order, so you should sort by the first item of the tuples (the indices) if you need them ordered.

```python
res  = parallel_tokenize(['0 1', '1 2'], rules=[], n_workers=2)
idxs,toks = zip(*L(res).sorted(itemgetter(0)))
test_eq(toks, [['0','1'],['1','2']])
```





### Tokenize texts in files

Preprocessing function for texts in filenames. Tokenized texts will be saved in a similar fashion in a directory suffixed with `_tok` in the parent folder of `path` (override with `output_dir`). This directory is the return value.


<h4 id="tokenize_folder" class="doc_header"><code>tokenize_folder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L178" class="source_link" style="float:right">[source]</a></h4>

> <code>tokenize_folder</code>(**`path`**, **`extensions`**=*`None`*, **`folders`**=*`None`*, **`output_dir`**=*`None`*, **`skip_if_exists`**=*`True`*, **`output_names`**=*`None`*, **`n_workers`**=*`4`*, **`rules`**=*`None`*, **`tok`**=*`None`*, **`encoding`**=*`'utf8'`*)

Tokenize text files in `path` in parallel using `n_workers`


The result will be in `output_dir` (defaults to a folder in the same parent directory as `path`, with `_tok` added to `path.name`) with the same structure as in `path`. Tokenized texts for a given file will be in the file having the same name in `output_dir`. Additionally, a file with a .len suffix contains the number of tokens and the count of all words is stored in `output_dir/counter.pkl`.

`extensions` will default to `['.txt']` and all text files in `path` are treated unless you specify a list of folders in `include`. `rules` (that defaults to [`defaults.text_proc_rules`](/text.core.html#defaults.text_proc_rules)) are applied to each text before going in the tokenizer.


<h4 id="tokenize_files" class="doc_header"><code>tokenize_files</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L187" class="source_link" style="float:right">[source]</a></h4>

> <code>tokenize_files</code>(**`files`**, **`path`**, **`output_dir`**, **`output_names`**=*`None`*, **`n_workers`**=*`4`*, **`rules`**=*`None`*, **`tok`**=*`None`*, **`encoding`**=*`'utf8'`*, **`skip_if_exists`**=*`False`*)

Tokenize text `files` in parallel using `n_workers`


### Tokenize texts in a dataframe


<h4 id="tokenize_texts" class="doc_header"><code>tokenize_texts</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L203" class="source_link" style="float:right">[source]</a></h4>

> <code>tokenize_texts</code>(**`texts`**, **`n_workers`**=*`4`*, **`rules`**=*`None`*, **`tok`**=*`None`*)

Tokenize `texts` in parallel using `n_workers`



<h4 id="tokenize_df" class="doc_header"><code>tokenize_df</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L211" class="source_link" style="float:right">[source]</a></h4>

> <code>tokenize_df</code>(**`df`**, **`text_cols`**, **`n_workers`**=*`4`*, **`rules`**=*`None`*, **`mark_fields`**=*`None`*, **`tok`**=*`None`*, **`res_col_name`**=*`'text'`*)

Tokenize texts in `df[text_cols]` in parallel using `n_workers`


This function returns a new dataframe with the same non-text columns, a column named text that contains the tokenized texts and a column named text_lengths that contains their respective length. It also returns a counter of all seen words to quickly build a vocabulary afterward.

`rules` (that defaults to [`defaults.text_proc_rules`](/text.core.html#defaults.text_proc_rules)) are applied to each text before going in the tokenizer. If `mark_fields` isn't specified, it defaults to `False` when there is a single text column, `True` when there are several. In that case, the texts in each of those columns are joined with `FLD` markers followed by the number of the field.


<h4 id="tokenize_csv" class="doc_header"><code>tokenize_csv</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L229" class="source_link" style="float:right">[source]</a></h4>

> <code>tokenize_csv</code>(**`fname`**, **`text_cols`**, **`outname`**=*`None`*, **`n_workers`**=*`4`*, **`rules`**=*`None`*, **`mark_fields`**=*`None`*, **`tok`**=*`None`*, **`header`**=*`'infer'`*, **`chunksize`**=*`50000`*)

Tokenize texts in the `text_cols` of the csv `fname` in parallel using `n_workers`



<h4 id="load_tokenized_csv" class="doc_header"><code>load_tokenized_csv</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L246" class="source_link" style="float:right">[source]</a></h4>

> <code>load_tokenized_csv</code>(**`fname`**)

Utility function to quickly load a tokenized csv ans the corresponding counter


The result will be written in a new csv file in `outname` (defaults to the same as `fname` with the suffix `_tok.csv`) and will have the same header as the original file, the same non-text columns, a text and a text_lengths column as described in [`tokenize_df`](/text.core.html#tokenize_df).

`rules` (that defaults to [`defaults.text_proc_rules`](/text.core.html#defaults.text_proc_rules)) are applied to each text before going in the tokenizer. If `mark_fields` isn't specified, it defaults to `False` when there is a single text column, `True` when there are several. In that case, the texts in each of those columns are joined with `FLD` markers followed by the number of the field.

The csv file is opened with `header` and optionally with blocks of `chunksize` at a time. If this argument is passed, each chunk is processed independently and saved in the output file to save memory usage.

```python
def _prepare_texts(tmp_d):
    "Prepare texts in a folder struct in tmp_d, a csv file and returns a dataframe"
    path = Path(tmp_d)/'tmp'
    path.mkdir()
    for d in ['a', 'b', 'c']: 
        (path/d).mkdir()
        for i in range(5):
            with open(path/d/f'text{i}.txt', 'w') as f: f.write(f"This is an example of text {d} {i}")
    
    texts = [f"This is an example of text {d} {i}" for i in range(5) for d in ['a', 'b', 'c']]
    df = pd.DataFrame({'text': texts, 'label': list(range(15))}, columns=['text', 'label'])
    csv_fname = tmp_d/'input.csv'
    df.to_csv(csv_fname, index=False)
    return path,df,csv_fname
```


<h3 id="Tokenizer" class="doc_header"><code>class</code> <code>Tokenizer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L255" class="source_link" style="float:right">[source]</a></h3>

> <code>Tokenizer</code>(**`tok`**, **`rules`**=*`None`*, **`counter`**=*`None`*, **`lengths`**=*`None`*, **`mode`**=*`None`*, **`sep`**=*`' '`*) :: [`Transform`](https://fastcore.fast.ai/transform#Transform)

Provides a consistent [`Transform`](https://fastcore.fast.ai/transform#Transform) interface to tokenizers operating on `DataFrame`s and folders


```python
with tempfile.TemporaryDirectory() as tmp_d:
    path,df,csv_fname = _prepare_texts(Path(tmp_d))
    items = get_text_files(path)
    splits = RandomSplitter()(items)
    dsets = Datasets(items, [Tokenizer.from_folder(path)], splits=splits)
    print(dsets.train[0])
    
    dsets = Datasets(df, [Tokenizer.from_df('text')], splits=splits)
    print(dsets.train[0][0].text)
```





    ((#10) ['xxbos','xxmaj','this','is','an','example','of','text','c','2'],)






    ('xxbos', 'xxmaj', 'this', 'is', 'an', 'example', 'of', 'text', 'c', '0')


```python
tst = test_set(dsets, ['This is a test', 'this is another test'])
test_eq(tst, [(['xxbos', 'xxmaj', 'this','is','a','test'],), 
              (['xxbos','this','is','another','test'],)])
```

## Sentencepiece


<h3 id="SentencePieceTokenizer" class="doc_header"><code>class</code> <code>SentencePieceTokenizer</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/core.py#L315" class="source_link" style="float:right">[source]</a></h3>

> <code>SentencePieceTokenizer</code>(**`lang`**=*`'en'`*, **`special_toks`**=*`None`*, **`sp_model`**=*`None`*, **`vocab_sz`**=*`None`*, **`max_vocab_sz`**=*`30000`*, **`model_type`**=*`'unigram'`*, **`char_coverage`**=*`None`*, **`cache_dir`**=*`'tmp'`*)

SentencePiece tokenizer for `lang`


```python
texts = [f"This is an example of text {i}" for i in range(10)]
df = pd.DataFrame({'text': texts, 'label': list(range(10))}, columns=['text', 'label'])
out,cnt = tokenize_df(df, text_cols='text', tok=SentencePieceTokenizer(vocab_sz=34), n_workers=1)
```









```python
with tempfile.TemporaryDirectory() as tmp_d:
    path,df,csv_fname = _prepare_texts(Path(tmp_d))
    items = get_text_files(path)
    splits = RandomSplitter()(items)
    tok = SentencePieceTokenizer(special_toks=[])
    dsets = Datasets(items, [Tokenizer.from_folder(path, tok=tok)], splits=splits)
    print(dsets.train[0])
    
    dsets = Datasets(df, [Tokenizer.from_df('text', tok=tok)], splits=splits)
    print(dsets.train[0][0].text)
```









    ((#33) ['▁xx','b','o','s','▁xx','m','a','j','▁t','h'...],)






    (#33) ['▁xx','b','o','s','▁xx','m','a','j','▁t','h'...]

