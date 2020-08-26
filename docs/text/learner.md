# Learner for the text application
> All the functions necessary to build `Learner` suitable for transfer learning in NLP


The most important functions of this module are [`language_model_learner`](/text.learner.html#language_model_learner) and [`text_classifier_learner`](/text.learner.html#text_classifier_learner). They will help you define a [`Learner`](/learner.html#Learner) using a pretrained model. See the [text tutorial](http://docs.fast.ai/tutorial.text) for exmaples of use.

## Loading a pretrained model

In text, to load a pretrained model, we need to adapt the embeddings of the vocabulary used for the pre-training to the vocabulary of our current corpus.


<h4 id="match_embeds" class="doc_header"><code>match_embeds</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L16" class="source_link" style="float:right">[source]</a></h4>

> <code>match_embeds</code>(**`old_wgts`**, **`old_vocab`**, **`new_vocab`**)

Convert the embedding in `old_wgts` to go from `old_vocab` to `new_vocab`.


For words in `new_vocab` that don't have a corresponding match in `old_vocab`, we use the mean of all pretrained embeddings. 

```python
wgts = {'0.encoder.weight': torch.randn(5,3)}
new_wgts = match_embeds(wgts.copy(), ['a', 'b', 'c'], ['a', 'c', 'd', 'b'])
old,new = wgts['0.encoder.weight'],new_wgts['0.encoder.weight']
test_eq(new[0], old[0])
test_eq(new[1], old[2])
test_eq(new[2], old.mean(0))
test_eq(new[3], old[1])
```


<h4 id="load_ignore_keys" class="doc_header"><code>load_ignore_keys</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L42" class="source_link" style="float:right">[source]</a></h4>

> <code>load_ignore_keys</code>(**`model`**, **`wgts`**)

Load `wgts` in `model` ignoring the names of the keys, just taking parameters in order



<h4 id="clean_raw_keys" class="doc_header"><code>clean_raw_keys</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L59" class="source_link" style="float:right">[source]</a></h4>

> <code>clean_raw_keys</code>(**`wgts`**)





<h4 id="load_model_text" class="doc_header"><code>load_model_text</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L68" class="source_link" style="float:right">[source]</a></h4>

> <code>load_model_text</code>(**`file`**, **`model`**, **`opt`**, **`with_opt`**=*`None`*, **`device`**=*`None`*, **`strict`**=*`True`*)

Load `model` from `file` along with `opt` (if available, and if `with_opt`)



<h2 id="TextLearner" class="doc_header"><code>class</code> <code>TextLearner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L86" class="source_link" style="float:right">[source]</a></h2>

> <code>TextLearner</code>(**`dls`**, **`model`**, **`alpha`**=*`2.0`*, **`beta`**=*`1.0`*, **`moms`**=*`(0.8, 0.7, 0.8)`*, **`loss_func`**=*`None`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`splitter`**=*`trainable_params`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*) :: [`Learner`](/learner.html#Learner)

Basic class for a [`Learner`](/learner.html#Learner) in NLP.


Adds a [`ModelResetter`](/callback.rnn.html#ModelResetter) and an [`RNNRegularizer`](/callback.rnn.html#RNNRegularizer) with `alpha` and `beta` to the callbacks, the rest is the same as [`Learner`](/learner.html#Learner) init. 

This [`Learner`](/learner.html#Learner) adds functionality to the base class:


<h4 id="TextLearner.load_pretrained" class="doc_header"><code>TextLearner.load_pretrained</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L110" class="source_link" style="float:right">[source]</a></h4>

> <code>TextLearner.load_pretrained</code>(**`wgts_fname`**, **`vocab_fname`**, **`model`**=*`None`*)

Load a pretrained model and adapt it to the data vocabulary.


`wgts_fname` should point to the weights of the pretrained model and `vocab_fname` to the vocabulary used to pretrain it.


<h4 id="TextLearner.save_encoder" class="doc_header"><code>TextLearner.save_encoder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L92" class="source_link" style="float:right">[source]</a></h4>

> <code>TextLearner.save_encoder</code>(**`file`**)

Save the encoder to `file` in the model directory


The model directory is [`Learner.path/Learner.model_dir`](/learner.html#Learner.path/Learner.model_dir).


<h4 id="TextLearner.load_encoder" class="doc_header"><code>TextLearner.load_encoder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L99" class="source_link" style="float:right">[source]</a></h4>

> <code>TextLearner.load_encoder</code>(**`file`**, **`device`**=*`None`*)

Load the encoder `file` from the model directory, optionally ensuring it's on `device`


## Language modeling predictions

For language modeling, the predict method is quite different form the other applications, which is why it needs its own subclass.


<h4 id="decode_spec_tokens" class="doc_header"><code>decode_spec_tokens</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L132" class="source_link" style="float:right">[source]</a></h4>

> <code>decode_spec_tokens</code>(**`tokens`**)

Decode the special tokens in `tokens`


```python
test_eq(decode_spec_tokens(['xxmaj', 'text']), ['Text'])
test_eq(decode_spec_tokens(['xxup', 'text']), ['TEXT'])
test_eq(decode_spec_tokens(['xxrep', '3', 'a']), ['aaa'])
test_eq(decode_spec_tokens(['xxwrep', '3', 'word']), ['word', 'word', 'word'])
```


<h3 id="LMLearner" class="doc_header"><code>class</code> <code>LMLearner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L154" class="source_link" style="float:right">[source]</a></h3>

> <code>LMLearner</code>(**`dls`**, **`model`**, **`alpha`**=*`2.0`*, **`beta`**=*`1.0`*, **`moms`**=*`(0.8, 0.7, 0.8)`*, **`loss_func`**=*`None`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`splitter`**=*`trainable_params`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*) :: [`TextLearner`](/text.learner.html#TextLearner)

Add functionality to [`TextLearner`](/text.learner.html#TextLearner) when dealingwith a language model



<h4 id="LMLearner.predict" class="doc_header"><code>LMLearner.predict</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L156" class="source_link" style="float:right">[source]</a></h4>

> <code>LMLearner.predict</code>(**`text`**, **`n_words`**=*`1`*, **`no_unk`**=*`True`*, **`temperature`**=*`1.0`*, **`min_p`**=*`None`*, **`no_bar`**=*`False`*, **`decoder`**=*`decode_spec_tokens`*, **`only_last_word`**=*`False`*)

Return `text` and the `n_words` that come after


The words are picked randomly among the predictions, depending on the probability of each index. `no_unk` means we never pick the `UNK` token, `temperature` is applied to the predictions, if `min_p` is passed, we don't consider the indices with a probability lower than it. Set `no_bar` to `True` if you don't want any progress bar, and you can pass a long a custom `decoder` to process the predicted tokens.

## [`Learner`](/learner.html#Learner) convenience functions


<h4 id="language_model_learner" class="doc_header"><code>language_model_learner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L193" class="source_link" style="float:right">[source]</a></h4>

> <code>language_model_learner</code>(**`dls`**, **`arch`**, **`config`**=*`None`*, **`drop_mult`**=*`1.0`*, **`backwards`**=*`False`*, **`pretrained`**=*`True`*, **`pretrained_fnames`**=*`None`*, **`loss_func`**=*`None`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`splitter`**=*`trainable_params`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*, **`moms`**=*`(0.95, 0.85, 0.95)`*)

Create a [`Learner`](/learner.html#Learner) with a language model from `dls` and `arch`.


You can use the `config` to customize the architecture used (change the values from [`awd_lstm_lm_config`](/text.models.awdlstm.html#awd_lstm_lm_config) for this), `pretrained` will use fastai's pretrained model for this `arch` (if available) or you can pass specific `pretrained_fnames` containing your own pretrained model and the corresponding vocabulary. All other arguments are passed to [`Learner`](/learner.html#Learner).

```python
path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/'texts.csv')
dls = TextDataLoaders.from_df(df, path=path, text_col='text', is_lm=True, valid_col='is_valid')
learn = language_model_learner(dls, AWD_LSTM)
```





You can then use the `.predict` method to generate new text.

```python
learn.predict('This movie is about', n_words=20)
```








    'This movie is about front - line highlights for fifteen United States Navy and the US Navy , four'



By default the entire sentence is feed again to the model after each predicted word, this little trick shows an improvement on the quality of the generated text. If you want to feed only the last word, specify argument `only_last_word`.

```python
learn.predict('This movie is about', n_words=20, only_last_word=True)
```








    'This movie is about 6 No person in Suppose he was used searching for the late in West of other important'




<h4 id="text_classifier_learner" class="doc_header"><code>text_classifier_learner</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/learner.py#L215" class="source_link" style="float:right">[source]</a></h4>

> <code>text_classifier_learner</code>(**`dls`**, **`arch`**, **`seq_len`**=*`72`*, **`config`**=*`None`*, **`backwards`**=*`False`*, **`pretrained`**=*`True`*, **`drop_mult`**=*`0.5`*, **`n_out`**=*`None`*, **`lin_ftrs`**=*`None`*, **`ps`**=*`None`*, **`max_len`**=*`1440`*, **`y_range`**=*`None`*, **`loss_func`**=*`None`*, **`opt_func`**=*`Adam`*, **`lr`**=*`0.001`*, **`splitter`**=*`trainable_params`*, **`cbs`**=*`None`*, **`metrics`**=*`None`*, **`path`**=*`None`*, **`model_dir`**=*`'models'`*, **`wd`**=*`None`*, **`wd_bn_bias`**=*`False`*, **`train_bn`**=*`True`*, **`moms`**=*`(0.95, 0.85, 0.95)`*)

Create a [`Learner`](/learner.html#Learner) with a text classifier from `dls` and `arch`.


You can use the `config` to customize the architecture used (change the values from [`awd_lstm_clas_config`](/text.models.awdlstm.html#awd_lstm_clas_config) for this), `pretrained` will use fastai's pretrained model for this `arch` (if available). `drop_mult` is a global multiplier applied to control all dropouts. `n_out` is usually inferred from the `dls` but you may pass it.

The model uses a [`SentenceEncoder`](/text.models.core.html#SentenceEncoder), which means the texts are passed `seq_len` tokens at a time, and will only compute the gradients on the last `max_len` steps. `lin_ftrs` and `ps` are passed to [`get_text_classifier`](/text.models.core.html#get_text_classifier).

All other arguments are passed to [`Learner`](/learner.html#Learner).

```python
path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/'texts.csv')
dls = TextDataLoaders.from_df(df, path=path, text_col='text', label_col='label', valid_col='is_valid')
learn = text_classifier_learner(dls, AWD_LSTM)
```




