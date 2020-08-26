# Core text modules
> Contain the modules common between different architectures and the generic functions to get models


## Language models


<h3 id="LinearDecoder" class="doc_header"><code>class</code> <code>LinearDecoder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/core.py#L26" class="source_link" style="float:right">[source]</a></h3>

> <code>LinearDecoder</code>(**`n_out`**, **`n_hid`**, **`output_p`**=*`0.1`*, **`tie_encoder`**=*`None`*, **`bias`**=*`True`*) :: [`Module`](/torch_core.html#Module)

To go on top of a RNNCore module and create a Language Model.


```python
from fastai.text.models.awdlstm import *
```

```python
enc = AWD_LSTM(100, 20, 10, 2)
x = torch.randint(0, 100, (10,5))
r = enc(x)

tst = LinearDecoder(100, 20, 0.1)
y = tst(r)
test_eq(y[1], r)
test_eq(y[2].shape, r.shape)
test_eq(y[0].shape, [10, 5, 100])

tst = LinearDecoder(100, 20, 0.1, tie_encoder=enc.encoder)
test_eq(tst.decoder.weight, enc.encoder.weight)
```


<h3 id="SequentialRNN" class="doc_header"><code>class</code> <code>SequentialRNN</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/core.py#L42" class="source_link" style="float:right">[source]</a></h3>

> <code>SequentialRNN</code>(**\*`args`**:`Any`) :: `Sequential`

A sequential module that passes the reset call to its children.


```python
class _TstMod(Module):
    def reset(self): print('reset')

tst = SequentialRNN(_TstMod(), _TstMod())
test_stdout(tst.reset, 'reset\nreset')
```


<h4 id="get_language_model" class="doc_header"><code>get_language_model</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/core.py#L48" class="source_link" style="float:right">[source]</a></h4>

> <code>get_language_model</code>(**`arch`**, **`vocab_sz`**, **`config`**=*`None`*, **`drop_mult`**=*`1.0`*)

Create a language model from `arch` and its `config`.


The default `config` used can be found in `_model_meta[arch]['config_lm']`. `drop_mult` is applied to all the probabilities of dropout in that config.

```python
config = awd_lstm_lm_config.copy()
config.update({'n_hid':10, 'emb_sz':20})

tst = get_language_model(AWD_LSTM, 100, config=config)
x = torch.randint(0, 100, (10,5))
y = tst(x)
test_eq(y[0].shape, [10, 5, 100])
test_eq(y[1].shape, [10, 5, 20])
test_eq(y[2].shape, [10, 5, 20])
test_eq(tst[1].decoder.weight, tst[0].encoder.weight)
```

```python
tst = get_language_model(AWD_LSTM, 100, config=config, drop_mult=0.5)
test_eq(tst[1].output_dp.p, config['output_p']*0.5)
for rnn in tst[0].rnns: test_eq(rnn.weight_p, config['weight_p']*0.5)
for dp in tst[0].hidden_dps: test_eq(dp.p, config['hidden_p']*0.5)
test_eq(tst[0].encoder_dp.embed_p, config['embed_p']*0.5)
test_eq(tst[0].input_dp.p, config['input_p']*0.5)
```

## Classification models


<h3 id="SentenceEncoder" class="doc_header"><code>class</code> <code>SentenceEncoder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/core.py#L68" class="source_link" style="float:right">[source]</a></h3>

> <code>SentenceEncoder</code>(**`bptt`**, **`module`**, **`pad_idx`**=*`1`*, **`max_len`**=*`None`*) :: [`Module`](/torch_core.html#Module)

Create an encoder over `module` that can process a full sentence.


{% include warning.html content='This module expects the inputs padded with most of the padding first, with the sequence beginning at a round multiple of `bptt` (and the rest of the padding at the end). Use [`pad_input_chunk`](/text.data.html#pad_input_chunk) to get your data in a suitable format.' %}

```python
mod = nn.Embedding(5, 10)
tst = SentenceEncoder(5, mod, pad_idx=0)
x = torch.randint(1, 5, (3, 15))
x[2,:5]=0
out,mask = tst(x)

test_eq(out[:1], mod(x)[:1])
test_eq(out[2,5:], mod(x)[2,5:])
test_eq(mask, x==0)
```


<h4 id="masked_concat_pool" class="doc_header"><code>masked_concat_pool</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/core.py#L90" class="source_link" style="float:right">[source]</a></h4>

> <code>masked_concat_pool</code>(**`output`**, **`mask`**, **`bptt`**)

Pool `MultiBatchEncoder` outputs into one vector [last_hidden, max_pool, avg_pool]


```python
out = torch.randn(2,4,5)
mask = tensor([[True,True,False,False], [False,False,False,True]])
x = masked_concat_pool(out, mask, 2)

test_close(x[0,:5], out[0,-1])
test_close(x[1,:5], out[1,-2])
test_close(x[0,5:10], out[0,2:].max(dim=0)[0])
test_close(x[1,5:10], out[1,:3].max(dim=0)[0])
test_close(x[0,10:], out[0,2:].mean(dim=0))
test_close(x[1,10:], out[1,:3].mean(dim=0))
```

```python
out1 = torch.randn(2,4,5)
out1[0,2:] = out[0,2:].clone()
out1[1,:3] = out[1,:3].clone()
x1 = masked_concat_pool(out1, mask, 2)
test_eq(x, x1)
```


<h3 id="PoolingLinearClassifier" class="doc_header"><code>class</code> <code>PoolingLinearClassifier</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/core.py#L101" class="source_link" style="float:right">[source]</a></h3>

> <code>PoolingLinearClassifier</code>(**`dims`**, **`ps`**, **`bptt`**, **`y_range`**=*`None`*) :: [`Module`](/torch_core.html#Module)

Create a linear classifier with pooling


```python
mod = nn.Embedding(5, 10)
tst = SentenceEncoder(5, mod, pad_idx=0)
x = torch.randint(1, 5, (3, 15))
x[2,:5]=0
out,mask = tst(x)

test_eq(out[:1], mod(x)[:1])
test_eq(out[2,5:], mod(x)[2,5:])
test_eq(mask, x==0)
```


<h4 id="get_text_classifier" class="doc_header"><code>get_text_classifier</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/core.py#L118" class="source_link" style="float:right">[source]</a></h4>

> <code>get_text_classifier</code>(**`arch`**, **`vocab_sz`**, **`n_class`**, **`seq_len`**=*`72`*, **`config`**=*`None`*, **`drop_mult`**=*`1.0`*, **`lin_ftrs`**=*`None`*, **`ps`**=*`None`*, **`pad_idx`**=*`1`*, **`max_len`**=*`1440`*, **`y_range`**=*`None`*)

Create a text classifier from `arch` and its `config`, maybe `pretrained`


```python
config = awd_lstm_clas_config.copy()
config.update({'n_hid':10, 'emb_sz':20})

tst = get_text_classifier(AWD_LSTM, 100, 3, config=config)
x = torch.randint(2, 100, (10,5))
y = tst(x)
test_eq(y[0].shape, [10, 3])
test_eq(y[1].shape, [10, 5, 20])
test_eq(y[2].shape, [10, 5, 20])
```

```python
tst.eval()
y = tst(x)
x1 = torch.cat([x, tensor([2,1,1,1,1,1,1,1,1,1])[:,None]], dim=1)
y1 = tst(x1)
test_close(y[0][1:],y1[0][1:])
```

```python
tst = get_text_classifier(AWD_LSTM, 100, 3, config=config, drop_mult=0.5)
test_eq(tst[1].layers[1][1].p, 0.1)
test_eq(tst[1].layers[0][1].p, config['output_p']*0.5)
for rnn in tst[0].module.rnns: test_eq(rnn.weight_p, config['weight_p']*0.5)
for dp in tst[0].module.hidden_dps: test_eq(dp.p, config['hidden_p']*0.5)
test_eq(tst[0].module.encoder_dp.embed_p, config['embed_p']*0.5)
test_eq(tst[0].module.input_dp.p, config['input_p']*0.5)
```
