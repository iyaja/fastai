# Tutorial - Transformers
> An example of how to incorporate the transfomers library from HuggingFace with fastai


In this tutorial, we will see how we can use the fastai library to fine-tune a pretrained transformer model from the [transformers library](https://github.com/huggingface/transformers) by HuggingFace. We will use the mid-level API to gather the data. Even if this tutorial is self contained, it might help to check the [imagenette tutorial](http://docs.fast.ai/tutorial.imagenette) to have a second look on the mid-level API (with a gentle introduction using the higher level APIs) in computer vision.

## Importing a transformers pretrained model

First things first, we will need to install the transformers library. If you haven't done it yet, install the library:

```
!pip install -Uq transformers
```

Then let's import what will need: we will fine-tune the GPT2 pretrained model and fine-tune on wikitext-2 here. For this, we need the `GPT2LMHeadModel` (since we want a language model) and the `GPT2Tokenizer` to prepare the data.

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
```

We can use several versions of this GPT2 model, look at the [transformers documentation](https://huggingface.co/transformers/pretrained_models.html) for more details. Here we will use the basic version (that already takes a lot of space in memory!) You can change the model used by changing the content of `pretrained_weights` (if it's not a GPT2 model, you'll need to change the classes used for the model and the tokenizer of course).

```python
pretrained_weights = 'gpt2'
tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_weights)
model = GPT2LMHeadModel.from_pretrained(pretrained_weights)
```

    Some weights of GPT2LMHeadModel were not initialized from the model checkpoint at gpt2 and are newly initialized: ['h.0.attn.masked_bias', 'h.1.attn.masked_bias', 'h.2.attn.masked_bias', 'h.3.attn.masked_bias', 'h.4.attn.masked_bias', 'h.5.attn.masked_bias', 'h.6.attn.masked_bias', 'h.7.attn.masked_bias', 'h.8.attn.masked_bias', 'h.9.attn.masked_bias', 'h.10.attn.masked_bias', 'h.11.attn.masked_bias', 'lm_head.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


Before we move on to the fine-tuning part, let's have a look at this `tokenizer` and this `model`. The tokenizers in HuggingFace usually do the tokenization and the numericalization in one step (we ignore the padding warning for now):

```python
ids = tokenizer.encode('This is an example of text, and')
ids
```




    [1212, 318, 281, 1672, 286, 2420, 11, 290]



Like fastai [`Transform`](https://fastcore.fast.ai/transform#Transform)s, the tokenizer has a `decode` method to give you back a text from ids:

```python
tokenizer.decode(ids)
```




    'This is an example of text, and'



The model can be used to generate predictions (it is pretrained). It has a `generate` method that expects a batch of prompt, so we feed it our ids and add one batch dimension (there is a padding warning we can ignore as well):

```python
import torch
```

```python
t = torch.LongTensor(ids)[None]
preds = model.generate(t)
```

    Setting `pad_token_id` to 50256 (first `eos_token_id`) to generate sequence


The predictions, by default, are of length 20:

```python
preds.shape,preds[0]
```




    (torch.Size([1, 20]),
     tensor([1212,  318,  281, 1672,  286, 2420,   11,  290,  340,  338,  407,  257,
              922,  530,   13,  198,  198,  464,  717, 1517]))



We can use the decode method (that prefers a numpy array to a tensor):

```python
tokenizer.decode(preds[0].numpy())
```




    "This is an example of text, and it's not a good one.\n\nThe first thing"



## Bridging the gap with fastai

Now let's see how we can use fastai to fine-tune this model on wikitext-2, using all the training utilities (learning rate finder, 1cycle policy etc...). First, we import all the text utilities:

```python
from fastai.text.all import *
```

### Preparing the data

Then we download the dataset (if not present), it comes as two csv files:

```python
path = untar_data(URLs.WIKITEXT_TINY)
path.ls()
```




    (#2) [Path('/home/jhoward/.fastai/data/wikitext-2/test.csv'),Path('/home/jhoward/.fastai/data/wikitext-2/train.csv')]



Let's have a look at what those csv files look like:

```python
df_train = pd.read_csv(path/'train.csv', header=None)
df_valid = pd.read_csv(path/'test.csv', header=None)
df_train.head()
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>\n = 2013 – 14 York City F.C. season = \n \n The 2013 – 14 season was the &lt;unk&gt; season of competitive association football and 77th season in the Football League played by York City Football Club , a professional football club based in York , North Yorkshire , England . Their 17th @-@ place finish in 2012 – 13 meant it was their second consecutive season in League Two . The season ran from 1 July 2013 to 30 June 2014 . \n Nigel Worthington , starting his first full season as York manager , made eight permanent summer signings . By the turn of the year York were only above the relegation z...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>\n = Big Boy ( song ) = \n \n " Big Boy " &lt;unk&gt; " I 'm A Big Boy Now " was the first single ever recorded by the Jackson 5 , which was released by Steeltown Records in January 1968 . The group played instruments on many of their Steeltown compositions , including " Big Boy " . The song was neither a critical nor commercial success , but the Jackson family were delighted with the outcome nonetheless . \n The Jackson 5 would release a second single with Steeltown Records before moving to Motown Records . The group 's recordings at Steeltown Records were thought to be lost , but they were re...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>\n = The Remix ( Lady Gaga album ) = \n \n The Remix is a remix album by American recording artist Lady Gaga . Released in Japan on March 3 , 2010 , it contains remixes of the songs from her first studio album , The Fame ( 2008 ) , and her third extended play , The Fame Monster ( 2009 ) . A revised version of the track list was prepared for release in additional markets , beginning with Mexico on May 3 , 2010 . A number of recording artists have produced the songs , including Pet Shop Boys , Passion Pit and The Sound of Arrows . The remixed versions feature both uptempo and &lt;unk&gt; composit...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>\n = New Year 's Eve ( Up All Night ) = \n \n " New Year 's Eve " is the twelfth episode of the first season of the American comedy television series Up All Night . The episode originally aired on NBC in the United States on January 12 , 2012 . It was written by Erica &lt;unk&gt; and was directed by Beth McCarthy @-@ Miller . The episode also featured a guest appearance from Jason Lee as Chris and Reagan 's neighbor and Ava 's boyfriend , Kevin . \n During Reagan ( Christina Applegate ) and Chris 's ( Will &lt;unk&gt; ) first New Year 's Eve game night , Reagan 's competitiveness comes out causing Ch...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>\n = Geopyxis carbonaria = \n \n Geopyxis carbonaria is a species of fungus in the genus Geopyxis , family &lt;unk&gt; . First described to science in 1805 , and given its current name in 1889 , the species is commonly known as the charcoal loving elf @-@ cup , dwarf &lt;unk&gt; cup , &lt;unk&gt; &lt;unk&gt; cup , or pixie cup . The small , &lt;unk&gt; @-@ shaped fruitbodies of the fungus are reddish @-@ brown with a whitish fringe and measure up to 2 cm ( 0 @.@ 8 in ) across . They have a short , tapered stalk . Fruitbodies are commonly found on soil where brush has recently been burned , sometimes in great numbers ....</td>
    </tr>
  </tbody>
</table>
</div>



We gather all texts in one numpy array (since it will be easier to use this way with fastai):

```python
all_texts = np.concatenate([df_train[0].values, df_valid[0].values])
```

To process this data to train a model, we need to build a [`Transform`](https://fastcore.fast.ai/transform#Transform) that will be applied lazily. In this case we might could do the pre-processing once and for all and only use the transform for decoding (we will see how just after), but the fast tokenizer from HuggingFace is, as its name indicates, fast, so it doesn't really impact performance to do it this way.

In a fastai [`Transform`](https://fastcore.fast.ai/transform#Transform) you can define:
- an <code>encodes</code> method that is applied when you call the transform (a bit like the `forward` method in a `nn.Module`)
- a <code>decodes</code> method that is applied when you call the `decode` method of the transform, if you need to decode anything for showing purposes (like converting ids to a text here)
- a <code>setups</code> method that sets some inner state of the [`Transform`](https://fastcore.fast.ai/transform#Transform) (not needed here so we skip it)

```python
class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x): 
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))
```

Two comments on the code above:
- in <code>encodes</code> we don't use the `tokenizer.encode` method since it does some additional preprocessing for the model after tokenizing and numericalizing (the part throwing a warning before). Here we don't need any post-processing so it's fine to skip it.
- in <code>decodes</code> we return a [`TitledStr`](/torch_core.html#TitledStr) object and not just a plain string. That's a fastai class that adds a `show` method to the string, which will allow us to use all the fastai show methods.

You can then group your data with this [`Transform`](https://fastcore.fast.ai/transform#Transform) using a [`TfmdLists`](/data.core.html#TfmdLists). It has an s in its name because it contains the training and validation set. We indicate the indices of the training set and the validation set with `splits` (here all the first indices until `len(df_train)` and then all the remaining indices):

```python
splits = [list(range_of(df_train)), list(range(len(df_train), len(all_texts)))]
tls = TfmdLists(all_texts, TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
```

We specify `dl_type=LMDataLoader` for when we will convert this [`TfmdLists`](/data.core.html#TfmdLists) to [`DataLoaders`](/data.core.html#DataLoaders): we will use an [`LMDataLoader`](/text.data.html#LMDataLoader) since we have a language modeling problem, not the usual fastai [`TfmdDL`](/data.core.html#TfmdDL).

In a [`TfmdLists`](/data.core.html#TfmdLists) you can access to the elements of the training or validation set quite easily:

```python
tls.train[0],tls.valid[0]
```




    (tensor([220, 198, 796,  ..., 198, 220, 198]),
     tensor([220, 198, 796,  ..., 198, 220, 198]))



They both look the same but that just begins they begin and end the same way, we can see the shape are different:

```python
tls.tfms(tls.train.items[0]).shape, tls.tfms(tls.valid.items[0]).shape
```




    (torch.Size([4576]), torch.Size([1485]))



And we can have a look at both decodes using [`show_at`](/data.core.html#show_at):

```python

```

```python

```

The fastai library expects the data to be assembled in a [`DataLoaders`](/data.core.html#DataLoaders) object (something that has a training and validation dataloader). We can get one by using the `dataloaders` method. We just have to specify a batch size and a sequence length. Since the GPT2 model was trained with sequences of size 1024, we use this sequence length (it's a stateless model, so it will change the perplexity if we use less):

```python
bs,sl = 8,1024
dls = tls.dataloaders(bs=bs, seq_len=sl)
```

Note that you may have to reduce the batch size depending on your GPU RAM.

In fastai, as soo as we have a [`DataLoaders`](/data.core.html#DataLoaders), we can use `show_batch` to have a look at the data (here texts for inputs, and the same text shifted by one token to the right for validation):

```python
dls.show_batch(max_n=2)
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
      <td>\n = Ten Commandments in Catholic theology = \n \n The Ten Commandments are a series of religious and moral &lt;unk&gt; that are recognized as a moral foundation in several of the Abrahamic religions, including Catholicism. As described in the Old Testament books Exodus and &lt;unk&gt;, the Commandments form part of a covenant offered by God to the Israelites to free them from the spiritual slavery of sin. According to the Catechism of the Catholic Church — the official &lt;unk&gt; of the Catholic Church's Christian beliefs — the Commandments are considered essential for spiritual good health and growth, and serve as the basis for Catholic social justice. A review of the Commandments is one of the most common types of examination of conscience used by Catholics before receiving the sacrament of &lt;unk&gt;. \n The Commandments appear in the earliest Church writings ; the Catechism states that they have "</td>
      <td>\n = Ten Commandments in Catholic theology = \n \n The Ten Commandments are a series of religious and moral &lt;unk&gt; that are recognized as a moral foundation in several of the Abrahamic religions, including Catholicism. As described in the Old Testament books Exodus and &lt;unk&gt;, the Commandments form part of a covenant offered by God to the Israelites to free them from the spiritual slavery of sin. According to the Catechism of the Catholic Church — the official &lt;unk&gt; of the Catholic Church's Christian beliefs — the Commandments are considered essential for spiritual good health and growth, and serve as the basis for Catholic social justice. A review of the Commandments is one of the most common types of examination of conscience used by Catholics before receiving the sacrament of &lt;unk&gt;. \n The Commandments appear in the earliest Church writings ; the Catechism states that they have " occupied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@ 8 million ( US $ 7 @.@ 0 million ) in 35 days. The film completed a 50 @-@ day run in 302 centres on 18 September 2009. By then, the film had collected ₹ 650 million ( US $ 9 @.@ 7 million ) and stood strong. \n The film completed its 100 @-@ day run in 223 centres and grossed over ₹ 1 @.@ 25 billion ( US $ 19 million ) without satellite and audio rights. By then it had surpassed &lt;unk&gt;'s Sivaji ( 2007 ), which grossed ₹ 650 million ( US $ 9 @.@ 7 million ) in Tamil Nadu, and stood second to &lt;unk&gt; ( 2008 ), which reached ₹ 2 billion ( US $ 30 million ). The film completed a 175 @-@ day run in 3 centres and, by then, collected a share of ₹ 580 million ( US $ 8</td>
      <td>8 million ( US $ 7 @.@ 0 million ) in 35 days. The film completed a 50 @-@ day run in 302 centres on 18 September 2009. By then, the film had collected ₹ 650 million ( US $ 9 @.@ 7 million ) and stood strong. \n The film completed its 100 @-@ day run in 223 centres and grossed over ₹ 1 @.@ 25 billion ( US $ 19 million ) without satellite and audio rights. By then it had surpassed &lt;unk&gt;'s Sivaji ( 2007 ), which grossed ₹ 650 million ( US $ 9 @.@ 7 million ) in Tamil Nadu, and stood second to &lt;unk&gt; ( 2008 ), which reached ₹ 2 billion ( US $ 30 million ). The film completed a 175 @-@ day run in 3 centres and, by then, collected a share of ₹ 580 million ( US $ 8</td>
    </tr>
  </tbody>
</table>


Another way to gather the data is to preprocess the texts once and for all and only use the transform to decode the tensors to texts:

```python
def tokenize(text):
    toks = tokenizer.tokenize(text)
    return tensor(tokenizer.convert_tokens_to_ids(toks))

tokenized = [tokenize(t) for t in progress_bar(all_texts)]
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='662' class='' max='662' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [662/662 00:12<00:00]
</div>



Now we change the previous [`Tokenizer`](/text.core.html#Tokenizer) like this:

```python
class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x): 
        return x if isinstance(x, Tensor) else tokenize(x)
        
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))
```

In the <code>encodes</code> method, we still account for the case where we get something that's not already tokenized, just in case we were to build a dataset with new texts using this transform.

```python
tls = TfmdLists(tokenized, TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
dls = tls.dataloaders(bs=bs, seq_len=sl)
```

And we can check it still works properly for showing purposes:

```python
dls.show_batch(max_n=2)
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
      <td>\n = New Year's Eve ( Up All Night ) = \n \n " New Year's Eve " is the twelfth episode of the first season of the American comedy television series Up All Night. The episode originally aired on NBC in the United States on January 12, 2012. It was written by Erica &lt;unk&gt; and was directed by Beth McCarthy @-@ Miller. The episode also featured a guest appearance from Jason Lee as Chris and Reagan's neighbor and Ava's boyfriend, Kevin. \n During Reagan ( Christina Applegate ) and Chris's ( Will &lt;unk&gt; ) first New Year's Eve game night, Reagan's competitiveness comes out causing Chris to become embarrassed. Meanwhile, Missy ( Jennifer Hall ) brings an unexpected date along to the party and, Kevin ( Jason Lee ) starts to feel as though Ava ( Maya Rudolph ) may be &lt;unk&gt; of him. \n " New Year's</td>
      <td>\n = New Year's Eve ( Up All Night ) = \n \n " New Year's Eve " is the twelfth episode of the first season of the American comedy television series Up All Night. The episode originally aired on NBC in the United States on January 12, 2012. It was written by Erica &lt;unk&gt; and was directed by Beth McCarthy @-@ Miller. The episode also featured a guest appearance from Jason Lee as Chris and Reagan's neighbor and Ava's boyfriend, Kevin. \n During Reagan ( Christina Applegate ) and Chris's ( Will &lt;unk&gt; ) first New Year's Eve game night, Reagan's competitiveness comes out causing Chris to become embarrassed. Meanwhile, Missy ( Jennifer Hall ) brings an unexpected date along to the party and, Kevin ( Jason Lee ) starts to feel as though Ava ( Maya Rudolph ) may be &lt;unk&gt; of him. \n " New Year's Eve</td>
    </tr>
    <tr>
      <th>1</th>
      <td>its peak intensity on August 28 as a Category 2 hurricane with maximum sustained winds of 100 mph ( 160 km / h ). At the same time, a reconnaissance aircraft reported a minimum barometric pressure of 991 mbar ( hPa ; 29 @.@ 27 inHg ) in the storm's eye as Edith made its closest pass to Bermuda. The hurricane began to gradually weaken after it passed east of the island, before becoming extratropical on August 31. The cyclone would later make a clockwise loop before dissipating completely late on September 3. Although Edith remained at sea, it was suspected that the hurricane may have caused the loss of the pleasure yacht &lt;unk&gt; IV, after it separated from its &lt;unk&gt;. \n \n = = = Tropical Storm Five = = = \n \n A weak disturbance was first observed near Grand Cayman on August 23, gaining tropical storm</td>
      <td>peak intensity on August 28 as a Category 2 hurricane with maximum sustained winds of 100 mph ( 160 km / h ). At the same time, a reconnaissance aircraft reported a minimum barometric pressure of 991 mbar ( hPa ; 29 @.@ 27 inHg ) in the storm's eye as Edith made its closest pass to Bermuda. The hurricane began to gradually weaken after it passed east of the island, before becoming extratropical on August 31. The cyclone would later make a clockwise loop before dissipating completely late on September 3. Although Edith remained at sea, it was suspected that the hurricane may have caused the loss of the pleasure yacht &lt;unk&gt; IV, after it separated from its &lt;unk&gt;. \n \n = = = Tropical Storm Five = = = \n \n A weak disturbance was first observed near Grand Cayman on August 23, gaining tropical storm strength</td>
    </tr>
  </tbody>
</table>


### Fine-tuning the model

The HuggingFace model will return a tuple in outputs, with the actual predictions and some additional activations (should we want to use them is some regularization scheme). To work inside the fastai training loop, we will need to drop those using a [`Callback`](/callback.core.html#Callback): we use those to alter the behavior of the training loop.

Here we need to write the event `after_pred` and replace `self.learn.pred` (which contains the predictions that will be passed to the loss function) by just its first element. In callbacks, there is a shortcut that lets you access any of the underlying [`Learner`](/learner.html#Learner) attribute so we can write `self.pred[0]` instead of `self.learn.pred[0]`. That shorcut only works for read access, not write, so we have to write `self.learn.pred` on the right side (otherwise we would set a `pred` attribute in the [`Callback`](/callback.core.html#Callback)).

```python
class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]
```

Of course we could make this a bit more complex and add some penalty to the loss using the other part of the tuple of predictions, like the [`RNNRegularizer`](/callback.rnn.html#RNNRegularizer).

Now, we are ready to create our [`Learner`](/learner.html#Learner), which is a fastai object grouping data, model and loss function and handles model training or inference. Since we are in a language model setting, we pass perplexity as a metric, and we need to use the callback we just defined. Lastly, we use mixed precision to save every bit of memory we can (and if you have a modern GPU, it will also make training faster):

```python
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()
```

We can check how good the model is without any fine-tuning step (spoiler alert, it's pretty good!)

```python
learn.validate()
```








    (#2) [3.2425637245178223,25.599266052246094]



This lists the validation loss and metrics (so 26.6 as perplexity is kind of amazing).

Now that we have a [`Learner`](/learner.html#Learner) we can use all the fastai training loop capabilities: learning rate finder, training with 1cycle etc... 

```python
learn.lr_find()
```








    SuggestedLRs(lr_min=0.00831763744354248, lr_steep=0.0691830962896347)




![png](output_62_2.png)


The learning rate finder curve suggests picking something between 1e-4 and 1e-3.

```python
learn.fit_one_cycle(1, 1e-4)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>perplexity</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.646031</td>
      <td>3.244000</td>
      <td>25.636072</td>
      <td>02:44</td>
    </tr>
  </tbody>
</table>


Now with just one epoch of fine-tuning and not much regularization, our model did not really improve since it was already amazing. To have a look at some generated texts, let's take a prompt that looks like a wikipedia article:

```python
df_valid.head(1)
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>\n = Tropical Storm &lt;unk&gt; ( 2008 ) = \n \n Tropical Storm &lt;unk&gt; was the tenth tropical storm of the 2008 Atlantic hurricane season . &lt;unk&gt; developed out of a strong tropical wave which moved off the African coast on August 31 . The wave quickly became organized and was declared Tropical Depression Ten while located 170 mi ( 270 km ) to the south @-@ southeast of the Cape Verde Islands on September 2 . The depression was quickly upgraded to Tropical Storm &lt;unk&gt; around noon the same day . Over the next several days , &lt;unk&gt; moved in a general west @-@ northwest direction and reached its peak...</td>
    </tr>
  </tbody>
</table>
</div>



Article seems to begin with new line and the title between = signs, so we will mimic that:

```python
prompt = "\n = Unicorn = \n \n A unicorn is a magical creature with a rainbow tail and a horn"
```

The prompt needs to be tokenized and numericalized, so we use the same function as before to do this, before we use the `generate` method of the model.

```python
prompt_ids = tokenizer.encode(prompt)
inp = tensor(prompt_ids)[None].cuda()
inp.shape
```




    torch.Size([1, 21])



```python
preds = learn.model.generate(inp, max_length=40, num_beams=5, temperature=1.5)
```

    Setting `pad_token_id` to 50256 (first `eos_token_id`) to generate sequence


```python
tokenizer.decode(preds[0])
```




    '\n = Unicorn = \n \n A unicorn is a magical creature with a rainbow tail and a horn on its head.\n\nA unicorn can fly at speeds of up to 100 miles per hour'


