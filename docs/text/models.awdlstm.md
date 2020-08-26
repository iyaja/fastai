# AWD-LSTM
> AWD LSTM from <a href='https://arxiv.org/pdf/1708.02182.pdf'>Smerity et al.</a> 


## Basic NLP modules

On top of the pytorch or the fastai [`layers`](/layers.html), the language models use some custom layers specific to NLP.


<h4 id="dropout_mask" class="doc_header"><code>dropout_mask</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/awdlstm.py#L12" class="source_link" style="float:right">[source]</a></h4>

> <code>dropout_mask</code>(**`x`**, **`sz`**, **`p`**)

Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.


```python
t = dropout_mask(torch.randn(3,4), [4,3], 0.25)
test_eq(t.shape, [4,3])
assert ((t == 4/3) + (t==0)).all()
```


<h3 id="RNNDropout" class="doc_header"><code>class</code> <code>RNNDropout</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/awdlstm.py#L17" class="source_link" style="float:right">[source]</a></h3>

> <code>RNNDropout</code>(**`p`**=*`0.5`*) :: [`Module`](/torch_core.html#Module)

Dropout with probability `p` that is consistent on the seq_len dimension.


```python
dp = RNNDropout(0.3)
tst_inp = torch.randn(4,3,7)
tst_out = dp(tst_inp)
for i in range(4):
    for j in range(7):
        if tst_out[i,0,j] == 0: assert (tst_out[i,:,j] == 0).all()
        else: test_close(tst_out[i,:,j], tst_inp[i,:,j]/(1-0.3))
```


<h3 id="WeightDropout" class="doc_header"><code>class</code> <code>WeightDropout</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/awdlstm.py#L26" class="source_link" style="float:right">[source]</a></h3>

> <code>WeightDropout</code>(**`module`**, **`weight_p`**, **`layer_names`**=*`'weight_hh_l0'`*) :: [`Module`](/torch_core.html#Module)

A module that warps another layer in which some weights will be replaced by 0 during training.


```python
module = nn.LSTM(5,7)
dp_module = WeightDropout(module, 0.4)
wgts = dp_module.module.weight_hh_l0
tst_inp = torch.randn(10,20,5)
h = torch.zeros(1,20,7), torch.zeros(1,20,7)
dp_module.reset()
x,h = dp_module(tst_inp,h)
loss = x.sum()
loss.backward()
new_wgts = getattr(dp_module.module, 'weight_hh_l0')
test_eq(wgts, getattr(dp_module, 'weight_hh_l0_raw'))
assert 0.2 <= (new_wgts==0).sum().float()/new_wgts.numel() <= 0.6
assert dp_module.weight_hh_l0_raw.requires_grad
assert dp_module.weight_hh_l0_raw.grad is not None
assert ((dp_module.weight_hh_l0_raw.grad == 0.) & (new_wgts == 0.)).any()
```


<h3 id="EmbeddingDropout" class="doc_header"><code>class</code> <code>EmbeddingDropout</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/awdlstm.py#L64" class="source_link" style="float:right">[source]</a></h3>

> <code>EmbeddingDropout</code>(**`emb`**, **`embed_p`**) :: [`Module`](/torch_core.html#Module)

Apply dropout with probabily `embed_p` to an embedding layer `emb`.


```python
enc = nn.Embedding(10, 7, padding_idx=1)
enc_dp = EmbeddingDropout(enc, 0.5)
tst_inp = torch.randint(0,10,(8,))
tst_out = enc_dp(tst_inp)
for i in range(8):
    assert (tst_out[i]==0).all() or torch.allclose(tst_out[i], 2*enc.weight[tst_inp[i]])
```


<h3 id="AWD_LSTM" class="doc_header"><code>class</code> <code>AWD_LSTM</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/awdlstm.py#L81" class="source_link" style="float:right">[source]</a></h3>

> <code>AWD_LSTM</code>(**`vocab_sz`**, **`emb_sz`**, **`n_hid`**, **`n_layers`**, **`pad_token`**=*`1`*, **`hidden_p`**=*`0.2`*, **`input_p`**=*`0.6`*, **`embed_p`**=*`0.1`*, **`weight_p`**=*`0.5`*, **`bidir`**=*`False`*) :: [`Module`](/torch_core.html#Module)

AWD-LSTM inspired by https://arxiv.org/abs/1708.02182


This is the core of an AWD-LSTM model, with embeddings from `vocab_sz` and `emb_sz`, `n_layers` LSTMs potentially `bidir` stacked, the first one going from `emb_sz` to `n_hid`, the last one from `n_hid` to `emb_sz` and all the inner ones from `n_hid` to `n_hid`. `pad_token` is passed to the PyTorch embedding layer. The dropouts are applied as such:
- the embeddings are wrapped in [`EmbeddingDropout`](/text.models.awdlstm.html#EmbeddingDropout) of probability `embed_p`;
- the result of this embedding layer goes through an [`RNNDropout`](/text.models.awdlstm.html#RNNDropout) of probability `input_p`;
- each LSTM has [`WeightDropout`](/text.models.awdlstm.html#WeightDropout) applied with probability `weight_p`;
- between two of the inner LSTM, an [`RNNDropout`](/text.models.awdlstm.html#RNNDropout) is applied with probability `hidden_p`.

THe module returns two lists: the raw outputs (without being applied the dropout of `hidden_p`) of each inner LSTM and the list of outputs with dropout. Since there is no dropout applied on the last output, those two lists have the same last element, which is the output that should be fed to a decoder (in the case of a language model).

```python
tst = AWD_LSTM(100, 20, 10, 2, hidden_p=0.2, embed_p=0.02, input_p=0.1, weight_p=0.2)
x = torch.randint(0, 100, (10,5))
r = tst(x)
test_eq(tst.bs, 10)
test_eq(len(tst.hidden), 2)
test_eq([h_.shape for h_ in tst.hidden[0]], [[1,10,10], [1,10,10]])
test_eq([h_.shape for h_ in tst.hidden[1]], [[1,10,20], [1,10,20]])

test_eq(r.shape, [10,5,20])
test_eq(r[:,-1], tst.hidden[-1][0][0]) #hidden state is the last timestep in raw outputs

tst.eval()
tst.reset()
tst(x);
tst(x);
```


<h4 id="awd_lstm_lm_split" class="doc_header"><code>awd_lstm_lm_split</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/awdlstm.py#L139" class="source_link" style="float:right">[source]</a></h4>

> <code>awd_lstm_lm_split</code>(**`model`**)

Split a RNN `model` in groups for differential learning rates.



<h4 id="awd_lstm_clas_split" class="doc_header"><code>awd_lstm_clas_split</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/awdlstm.py#L150" class="source_link" style="float:right">[source]</a></h4>

> <code>awd_lstm_clas_split</code>(**`model`**)

Split a RNN `model` in groups for differential learning rates.


## QRNN


<h3 id="AWD_QRNN" class="doc_header"><code>class</code> <code>AWD_QRNN</code><a href="https://github.com/fastai/fastai/tree/master/fastai/text/models/awdlstm.py#L162" class="source_link" style="float:right">[source]</a></h3>

> <code>AWD_QRNN</code>(**`vocab_sz`**, **`emb_sz`**, **`n_hid`**, **`n_layers`**, **`pad_token`**=*`1`*, **`hidden_p`**=*`0.2`*, **`input_p`**=*`0.6`*, **`embed_p`**=*`0.1`*, **`weight_p`**=*`0.5`*, **`bidir`**=*`False`*) :: [`AWD_LSTM`](/text.models.awdlstm.html#AWD_LSTM)

Same as an AWD-LSTM, but using QRNNs instead of LSTMs


```python
# cpp
model = AWD_QRNN(vocab_sz=10, emb_sz=20, n_hid=16, n_layers=2, bidir=False)
x = torch.randint(0, 10, (7,5))
y = model(x)
test_eq(y.shape, (7, 5, 20))
```
