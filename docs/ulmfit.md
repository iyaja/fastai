# ULMFiT



```python
from fastai.text.all import *
from nbdev.showdoc import show_doc
```

## Finetune a pretrained Language Model

First we get our data and tokenize it.

```python
path = untar_data(URLs.IMDB)
```

```python
texts = get_files(path, extensions=['.txt'], folders=['unsup', 'train', 'test'])
len(texts)
```




    100000



Then we put it in a [`Datasets`](/data.core.html#Datasets). For a language model, we don't have targets, so there is only one transform to numericalize the texts. Note that [`tokenize_df`](/text.core.html#tokenize_df) returns the count of the words in the corpus to make it easy to create a vocabulary.

```python
def read_file(f): return L(f.read().split(' '))
```

```python
splits = RandomSplitter(valid_pct=0.1)(texts)
tfms = [Tokenizer.from_folder(path), Numericalize()]
dsets = Datasets(texts, [tfms], splits=splits, dl_type=LMDataLoader)
```

Then we use that [`Datasets`](/data.core.html#Datasets) to create a [`DataLoaders`](/data.core.html#DataLoaders). Here the class of [`TfmdDL`](/data.core.html#TfmdDL) we need to use is [`LMDataLoader`](/text.data.html#LMDataLoader) which will concatenate all the texts in a source (with a shuffle at each epoch for the training set), split it in `bs` chunks then read continuously through it.

```python
bs,sl=256,80
dbunch_lm = dsets.dataloaders(bs=bs, seq_len=sl, val_bs=bs)
```

```python
dbunch_lm.show_batch()
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
      <td>xxbos i saw this before ' bubba ho - tep ' at the fantasia film festival in montreal . everything about it is either tipping the hat to ( or completely ripping off ) tim burton . i enjoyed it nonetheless , even if it is extremely derivative . what most impressed me was the quality of the visuals given the obvious shoe - string budget . the set design and the props were inventive and original , although the</td>
      <td>i saw this before ' bubba ho - tep ' at the fantasia film festival in montreal . everything about it is either tipping the hat to ( or completely ripping off ) tim burton . i enjoyed it nonetheless , even if it is extremely derivative . what most impressed me was the quality of the visuals given the obvious shoe - string budget . the set design and the props were inventive and original , although the script</td>
    </tr>
    <tr>
      <th>1</th>
      <td>climax 25 minutes of sloppy wrap - up with a character and her dad that we do n't give a crap about anyway … xxunk line , xxup save xxup yourself … xxunk xxup from xxup this xxup movie xxrep 3 ! xxbos i never get tired of watching this movie . i am a die - hard chick - flick fan : fluff all the way , all that meaningless dime - a - dozen stuff . xxmaj but</td>
      <td>25 minutes of sloppy wrap - up with a character and her dad that we do n't give a crap about anyway … xxunk line , xxup save xxup yourself … xxunk xxup from xxup this xxup movie xxrep 3 ! xxbos i never get tired of watching this movie . i am a die - hard chick - flick fan : fluff all the way , all that meaningless dime - a - dozen stuff . xxmaj but this</td>
    </tr>
    <tr>
      <th>2</th>
      <td>not hard for me to imagine a dedicated xxmaj mr . xxmaj xxunk teaching kids in xxmaj sunday xxmaj school about the good book . xxmaj nor is it hard to understand why they might picture xxunk 's guards in double - breasted suits like the gangsters in the news of their youth , or relating any number of other scenes to what was familiar to them . \n\n xxmaj connelly was not trying to convert viewers to religion …</td>
      <td>hard for me to imagine a dedicated xxmaj mr . xxmaj xxunk teaching kids in xxmaj sunday xxmaj school about the good book . xxmaj nor is it hard to understand why they might picture xxunk 's guards in double - breasted suits like the gangsters in the news of their youth , or relating any number of other scenes to what was familiar to them . \n\n xxmaj connelly was not trying to convert viewers to religion … he</td>
    </tr>
    <tr>
      <th>3</th>
      <td>get better , but it never did . xxmaj from the start to the end , it was one big cliché , extremely predictable with not one surprise in the entire film . xxmaj from the over the top ridiculous boyfriend of xxmaj dawn and the wedding in the pie shop , the wife of the doctor being in the delivery room . i even found the scene where the husband finds the money and wants her to tell him</td>
      <td>better , but it never did . xxmaj from the start to the end , it was one big cliché , extremely predictable with not one surprise in the entire film . xxmaj from the over the top ridiculous boyfriend of xxmaj dawn and the wedding in the pie shop , the wife of the doctor being in the delivery room . i even found the scene where the husband finds the money and wants her to tell him it</td>
    </tr>
    <tr>
      <th>4</th>
      <td>up and sings a happy tune , but his mother comes in and tells him to shut up again and gives him a dope slap that leaves a dent in his forehead . i mention this commercial , because it was considered funny , and i did n't hear any objections to it while i was there . xxmaj there is a lot more bloodshed and physical cruelty on screen in " the xxmaj great xxmaj yokai xxmaj war "</td>
      <td>and sings a happy tune , but his mother comes in and tells him to shut up again and gives him a dope slap that leaves a dent in his forehead . i mention this commercial , because it was considered funny , and i did n't hear any objections to it while i was there . xxmaj there is a lot more bloodshed and physical cruelty on screen in " the xxmaj great xxmaj yokai xxmaj war " than</td>
    </tr>
    <tr>
      <th>5</th>
      <td>this film portray our countries xxmaj special xxmaj forces . xxmaj gomer xxmaj pile could have probably survived longer than the " spec xxmaj ops " soldiers in this film . xxmaj for crying out loud they should have called them the xxmaj special xxmaj education xxmaj forces instead . xxmaj if you are going to write a script where you send in an elite team to deal with an outbreak of zombies , at least have the soldiers be</td>
      <td>film portray our countries xxmaj special xxmaj forces . xxmaj gomer xxmaj pile could have probably survived longer than the " spec xxmaj ops " soldiers in this film . xxmaj for crying out loud they should have called them the xxmaj special xxmaj education xxmaj forces instead . xxmaj if you are going to write a script where you send in an elite team to deal with an outbreak of zombies , at least have the soldiers be smarter</td>
    </tr>
    <tr>
      <th>6</th>
      <td>of high school ( they have to show at least 10 doors in the high school labeled " debate xxmaj club , " " german xxmaj club , " etc . ) and they tend to make fun of things like teen pregnancy and teen sex , which really has xxup nothing to do with making fun of horror films . xxmaj to say the least , i probably laughed once or twice through the entire 90 minutes , and</td>
      <td>high school ( they have to show at least 10 doors in the high school labeled " debate xxmaj club , " " german xxmaj club , " etc . ) and they tend to make fun of things like teen pregnancy and teen sex , which really has xxup nothing to do with making fun of horror films . xxmaj to say the least , i probably laughed once or twice through the entire 90 minutes , and that</td>
    </tr>
    <tr>
      <th>7</th>
      <td>the xxmaj secret xxmaj xxunk . \n\n xxmaj however , xxmaj i 'm a little perplexed about how people have perceived her diary and of her as a person , seeing her as a little saint or having a message of hope for the world . i do n't think that was the original intention of her diary . xxmaj she wrote it mainly for herself , even though she did make some rigorous rewrites before the occupants of the</td>
      <td>xxmaj secret xxmaj xxunk . \n\n xxmaj however , xxmaj i 'm a little perplexed about how people have perceived her diary and of her as a person , seeing her as a little saint or having a message of hope for the world . i do n't think that was the original intention of her diary . xxmaj she wrote it mainly for herself , even though she did make some rigorous rewrites before the occupants of the xxmaj</td>
    </tr>
    <tr>
      <th>8</th>
      <td>lied . xxmaj other facts are brought to light that , finally , result in xxmaj dillon 's release . xxmaj the killer is never found , though the movie gives us a thorough xxunk as a plausible perp . \n\n xxmaj this is a weeper from beginning to end . xxmaj nothing seems to go right for the couple . xxmaj oh , there are a few happy moment , maybe a party where everyone is glad to be</td>
      <td>. xxmaj other facts are brought to light that , finally , result in xxmaj dillon 's release . xxmaj the killer is never found , though the movie gives us a thorough xxunk as a plausible perp . \n\n xxmaj this is a weeper from beginning to end . xxmaj nothing seems to go right for the couple . xxmaj oh , there are a few happy moment , maybe a party where everyone is glad to be together</td>
    </tr>
  </tbody>
</table>


Then we have a convenience method to directly grab a [`Learner`](/learner.html#Learner) from it, using the [`AWD_LSTM`](/text.models.awdlstm.html#AWD_LSTM) architecture.

```python
opt_func = partial(Adam, wd=0.1)
learn = language_model_learner(dbunch_lm, AWD_LSTM, opt_func=opt_func, metrics=[accuracy, Perplexity()], path=path)
learn = learn.to_fp16(clip=0.1)
```

```python
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7,0.8))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>perplexity</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4.426135</td>
      <td>3.984901</td>
      <td>0.292371</td>
      <td>53.779987</td>
      <td>07:00</td>
    </tr>
  </tbody>
</table>


```python
learn.save('stage1')
```

```python
learn.load('stage1');
```

```python
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3, moms=(0.8,0.7,0.8))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>perplexity</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4.163227</td>
      <td>3.870354</td>
      <td>0.306840</td>
      <td>47.959347</td>
      <td>07:24</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4.055693</td>
      <td>3.790802</td>
      <td>0.316436</td>
      <td>44.291908</td>
      <td>07:41</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.979279</td>
      <td>3.729021</td>
      <td>0.323357</td>
      <td>41.638317</td>
      <td>07:22</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.919654</td>
      <td>3.688891</td>
      <td>0.327770</td>
      <td>40.000469</td>
      <td>07:22</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.889432</td>
      <td>3.660633</td>
      <td>0.330762</td>
      <td>38.885933</td>
      <td>07:22</td>
    </tr>
    <tr>
      <td>5</td>
      <td>3.842923</td>
      <td>3.637397</td>
      <td>0.333315</td>
      <td>37.992798</td>
      <td>07:26</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.813823</td>
      <td>3.619074</td>
      <td>0.335308</td>
      <td>37.303013</td>
      <td>07:25</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.793213</td>
      <td>3.608010</td>
      <td>0.336566</td>
      <td>36.892574</td>
      <td>07:20</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3.766456</td>
      <td>3.602140</td>
      <td>0.337257</td>
      <td>36.676647</td>
      <td>07:22</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3.759768</td>
      <td>3.600955</td>
      <td>0.337450</td>
      <td>36.633202</td>
      <td>07:23</td>
    </tr>
  </tbody>
</table>


Once we have fine-tuned the pretrained language model to this corpus, we save the encoder since we will use it for the classifier.

```python
learn.save_encoder('finetuned1')
```

## Use it to train a classifier

```python
texts = get_files(path, extensions=['.txt'], folders=['train', 'test'])
```

```python
splits = GrandparentSplitter(valid_name='test')(texts)
```

For classification, we need to use two set of transforms: one to numericalize the texts and the other to encode the labels as categories.

```python
x_tfms = [Tokenizer.from_folder(path), Numericalize(vocab=dbunch_lm.vocab)]
dsets = Datasets(texts, [x_tfms, [parent_label, Categorize()]], splits=splits, dl_type=SortedDL)
```

```python
bs = 64
```

```python
dls = dsets.dataloaders(before_batch=pad_input_chunk, bs=bs)
```

```python
dls.show_batch(max_n=2)
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
      <td>xxbos * * attention xxmaj spoilers * * \n\n xxmaj first of all , let me say that xxmaj rob xxmaj roy is one of the best films of the 90 's . xxmaj it was an amazing achievement for all those involved , especially the acting of xxmaj liam xxmaj neeson , xxmaj jessica xxmaj lange , xxmaj john xxmaj hurt , xxmaj brian xxmaj cox , and xxmaj tim xxmaj roth . xxmaj michael xxmaj canton xxmaj jones painted a wonderful portrait of the honor and dishonor that men can represent in themselves . xxmaj but alas … \n\n it constantly , and unfairly gets compared to " braveheart " . xxmaj these are two entirely different films , probably only similar in the fact that they are both about xxmaj scots in historical xxmaj scotland . xxmaj yet , this comparison frequently bothers me because it seems</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxbos xxmaj by now you 've probably heard a bit about the new xxmaj disney dub of xxmaj miyazaki 's classic film , xxmaj laputa : xxmaj castle xxmaj in xxmaj the xxmaj sky . xxmaj during late summer of 1998 , xxmaj disney released " kiki 's xxmaj delivery xxmaj service " on video which included a preview of the xxmaj laputa dub saying it was due out in " 1 xxrep 3 9 " . xxmaj it 's obviously way past that year now , but the dub has been finally completed . xxmaj and it 's not " laputa : xxmaj castle xxmaj in xxmaj the xxmaj sky " , just " castle xxmaj in xxmaj the xxmaj sky " for the dub , since xxmaj laputa is not such a nice word in xxmaj spanish ( even though they use the word xxmaj laputa many times</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>


Then we once again have a convenience function to create a classifier from this [`DataLoaders`](/data.core.html#DataLoaders) with the [`AWD_LSTM`](/text.models.awdlstm.html#AWD_LSTM) architecture.

```python
opt_func = partial(Adam, wd=0.1)
learn = text_classifier_learner(dls, AWD_LSTM, metrics=[accuracy], path=path, drop_mult=0.5, opt_func=opt_func)
```

We load our pretrained encoder.

```python
learn = learn.load_encoder('finetuned1')
learn = learn.to_fp16(clip=0.1)
```

Then we can train with gradual unfreezing and differential learning rates.

```python
lr = 1e-1 * bs/128
```

```python
learn.fit_one_cycle(1, lr, moms=(0.8,0.7,0.8), wd=0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.328318</td>
      <td>0.200650</td>
      <td>0.922120</td>
      <td>01:08</td>
    </tr>
  </tbody>
</table>


```python
learn.freeze_to(-2)
lr /= 2
learn.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7,0.8), wd=0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.208120</td>
      <td>0.166004</td>
      <td>0.937440</td>
      <td>01:15</td>
    </tr>
  </tbody>
</table>


```python
learn.freeze_to(-3)
lr /= 2
learn.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7,0.8), wd=0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.162498</td>
      <td>0.154959</td>
      <td>0.942400</td>
      <td>01:35</td>
    </tr>
  </tbody>
</table>


```python
learn.unfreeze()
lr /= 5
learn.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=(0.8,0.7,0.8), wd=0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.133800</td>
      <td>0.163456</td>
      <td>0.940560</td>
      <td>01:34</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.095326</td>
      <td>0.154301</td>
      <td>0.945120</td>
      <td>01:34</td>
    </tr>
  </tbody>
</table>

