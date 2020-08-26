# Transfer learning in text
> How to fine-tune a language model and train a classifier


```python
from fastai.text.all import *
```

In this tutorial, we will see how we can train a model to classify text (here based on their sentiment). First we will see how to do this quickly in a few lines of code, then how to get state-of-the art results using the approach of the [ULMFit paper](https://arxiv.org/abs/1801.06146).

We will use the IMDb dataset from the paper [Learning Word Vectors for Sentiment Analysis]((https://ai.stanford.edu/~amaas/data/sentiment/)), containing a few thousand movie reviews.

## Train a text classifier from a pretrained model

We will try to train a classifier using a pretrained model, a bit like we do in the [vision tutorial](http://docs.fast.ai/tutorial.vision). To get our data ready, we will first use the high-level API:

## Using the high-level API

We can download the data and decompress it with the following command:

```python
path = untar_data(URLs.IMDB)
path.ls()
```




    (#5) [Path('/home/sgugger/.fastai/data/imdb/unsup'),Path('/home/sgugger/.fastai/data/imdb/models'),Path('/home/sgugger/.fastai/data/imdb/train'),Path('/home/sgugger/.fastai/data/imdb/test'),Path('/home/sgugger/.fastai/data/imdb/README')]



```python
(path/'train').ls()
```




    (#4) [Path('/home/sgugger/.fastai/data/imdb/train/pos'),Path('/home/sgugger/.fastai/data/imdb/train/unsupBow.feat'),Path('/home/sgugger/.fastai/data/imdb/train/labeledBow.feat'),Path('/home/sgugger/.fastai/data/imdb/train/neg')]



The data follows an ImageNet-style organization, in the train folder, we have two subfolders, `pos` and `neg` (for positive reviews and negative reviews). We can gather it by using the [`TextDataLoaders.from_folder`](/text.data.html#TextDataLoaders.from_folder) method. The only thing we need to specify is the name of the validation folder, which is "test" (and not the default "valid").

```python
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
```

We can then have a look at the data with the `show_batch` method:

```python
dls.show_batch()
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
      <td>xxbos xxmaj match 1 : xxmaj tag xxmaj team xxmaj table xxmaj match xxmaj bubba xxmaj ray and xxmaj spike xxmaj dudley vs xxmaj eddie xxmaj guerrero and xxmaj chris xxmaj benoit xxmaj bubba xxmaj ray and xxmaj spike xxmaj dudley started things off with a xxmaj tag xxmaj team xxmaj table xxmaj match against xxmaj eddie xxmaj guerrero and xxmaj chris xxmaj benoit . xxmaj according to the rules of the match , both opponents have to go through tables in order to get the win . xxmaj benoit and xxmaj guerrero heated up early on by taking turns hammering first xxmaj spike and then xxmaj bubba xxmaj ray . a xxmaj german xxunk by xxmaj benoit to xxmaj bubba took the wind out of the xxmaj dudley brother . xxmaj spike tried to help his brother , but the referee restrained him while xxmaj benoit and xxmaj guerrero</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxbos xxmaj warning : xxmaj does contain spoilers . \n\n xxmaj open xxmaj your xxmaj eyes \n\n xxmaj if you have not seen this film and plan on doing so , just stop reading here and take my word for it . xxmaj you have to see this film . i have seen it four times so far and i still have n't made up my mind as to what exactly happened in the film . xxmaj that is all i am going to say because if you have not seen this film , then stop reading right now . \n\n xxmaj if you are still reading then i am going to pose some questions to you and maybe if anyone has any answers you can email me and let me know what you think . \n\n i remember my xxmaj grade 11 xxmaj english teacher quite well . xxmaj</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxbos i thought that xxup rotj was clearly the best out of the three xxmaj star xxmaj wars movies . i find it surprising that xxup rotj is considered the weakest installment in the xxmaj trilogy by many who have voted . xxmaj to me it seemed like xxup rotj was the best because it had the most profound plot , the most suspense , surprises , most xxunk the ending ) and definitely the most episodic movie . i personally like the xxmaj empire xxmaj strikes xxmaj back a lot also but i think it is slightly less good than than xxup rotj since it was slower - moving , was not as episodic , and i just did not feel as much suspense or emotion as i did with the third movie . \n\n xxmaj it also seems like to me that after reading these surprising reviews that</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>xxbos xxup the xxup shop xxup around xxup the xxup corner is one of the sweetest and most feel - good romantic comedies ever made . xxmaj there 's just no getting around that , and it 's hard to actually put one 's feeling for this film into words . xxmaj it 's not one of those films that tries too hard , nor does it come up with the oddest possible scenarios to get the two protagonists together in the end . xxmaj in fact , all its charm is innate , contained within the characters and the setting and the plot … which is highly believable to boot . xxmaj it 's easy to think that such a love story , as beautiful as any other ever told , * could * happen to you … a feeling you do n't often get from other romantic comedies</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xxbos xxmaj the premise of this movie has been tickling my imagination for quite some time now . xxmaj we 've all heard or read about it in some kind of con - text . xxmaj what would you do if you were all alone in the world ? xxmaj what would you do if the entire world suddenly disappeared in front of your eyes ? xxmaj in fact , the last part is actually what happens to xxmaj dave and xxmaj andrew , two room - mates living in a run - down house in the middle of a freeway system . xxmaj andrew is a nervous wreck to say the least and xxmaj dave is considered being one of the biggest losers of society . xxmaj that alone is the main reason to why these two guys get so well along , because they simply only have each</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xxbos xxrep 3 * xxup spoilers xxrep 3 * xxrep 3 * xxup spoilers xxrep 3 * xxmaj continued … \n\n xxmaj from here on in the whole movie collapses in on itself . xxmaj first we meet a rogue program with the indication we 're gon na get ghosts and vampires and werewolves and the like . xxmaj we get a guy with a retarded accent talking endless garbage , two ' ghosts ' that serve no real purpose and have no character what - so - ever and a bunch of henchmen . xxmaj someone 's told me they 're vampires ( straight out of xxmaj blade 2 ) , but they 're so undefined i did n't realise . \n\n xxmaj the funny accented guy with a ridiculous name suffers the same problem as the xxmaj oracle , only for far longer and far far worse .</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>6</th>
      <td>xxbos xxmaj i 've rented and watched this movie for the 1st time on xxup dvd without reading any reviews about it . xxmaj so , after 15 minutes of watching xxmaj i 've noticed that something is wrong with this movie ; it 's xxup terrible ! i mean , in the trailers it looked scary and serious ! \n\n i think that xxmaj eli xxmaj roth ( mr . xxmaj director ) thought that if all the characters in this film were stupid , the movie would be funny … ( so stupid , it 's funny … ? xxup wrong ! ) xxmaj he should watch and learn from better horror - comedies such xxunk xxmaj night " , " the xxmaj lost xxmaj boys " and " the xxmaj return xxmaj of the xxmaj living xxmaj dead " ! xxmaj those are funny ! \n\n "</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>7</th>
      <td>xxbos xxup myra xxup breckinridge is one of those rare films that established its place in film history immediately . xxmaj praise for the film was absolutely nonexistent , even from the people involved in making it . xxmaj this film was loathed from day one . xxmaj while every now and then one will come across some maverick who will praise the film on philosophical grounds ( aggressive feminism or the courage to tackle the issue of xxunk ) , the film has not developed a cult following like some notorious flops do . xxmaj it 's not hailed as a misunderstood masterpiece like xxup scarface , or trotted out to be ridiculed as a camp classic like xxup showgirls . \n\n xxmaj undoubtedly the reason is that the film , though outrageously awful , is not lovable , or even likable . xxup myra xxup breckinridge is just</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>8</th>
      <td>xxbos xxmaj after reading the previous comments , xxmaj i 'm just glad that i was n't the only person left confused , especially by the last 20 minutes . xxmaj john xxmaj carradine is shown twice walking down into a grave and pulling the lid shut after him . i anxiously awaited some kind of explanation for this odd behavior … naturally i assumed he had something to do with the evil goings - on at the house , but since he got killed off by the first rising corpse ( hereafter referred to as xxmaj zombie # 1 ) , these scenes made absolutely no sense . xxmaj please , if someone out there knows why xxmaj carradine kept climbing down into graves -- let the rest of us in on it ! ! \n\n xxmaj all the action is confined to the last 20 minutes so xxmaj</td>
      <td>neg</td>
    </tr>
  </tbody>
</table>


We can see that the library automatically processed all the texts to split then in *tokens*, adding some special tokens like:

- `xxbos` to indicate the beginning of a text
- `xxmaj` to indicate the next word was capitalized

Then, we can define a [`Learner`](/learner.html#Learner) suitable for text classification in one line:

```python
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
```

We use the [AWD LSTM](https://arxiv.org/abs/1708.02182) architecture, `drop_mult` is a parameter that controls the magnitude of all dropouts in that model, and we use [`accuracy`](/metrics.html#accuracy) to track down how well we are doing. We can then fine-tune our pretrained model:

```python
learn.fine_tune(4, 1e-2)
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
      <td>0.587251</td>
      <td>0.386230</td>
      <td>0.828960</td>
      <td>01:35</td>
    </tr>
  </tbody>
</table>



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
      <td>0.307347</td>
      <td>0.263843</td>
      <td>0.892800</td>
      <td>03:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.215867</td>
      <td>0.226208</td>
      <td>0.911800</td>
      <td>02:55</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.155399</td>
      <td>0.231144</td>
      <td>0.913960</td>
      <td>03:12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.129277</td>
      <td>0.200941</td>
      <td>0.925920</td>
      <td>03:01</td>
    </tr>
  </tbody>
</table>


```python
learn.fine_tune(4, 1e-2)
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
      <td>0.594912</td>
      <td>0.407416</td>
      <td>0.823640</td>
      <td>01:35</td>
    </tr>
  </tbody>
</table>



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
      <td>0.268259</td>
      <td>0.316242</td>
      <td>0.876000</td>
      <td>03:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.184861</td>
      <td>0.246242</td>
      <td>0.898080</td>
      <td>03:10</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.136392</td>
      <td>0.220086</td>
      <td>0.918200</td>
      <td>03:16</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.106423</td>
      <td>0.191092</td>
      <td>0.931360</td>
      <td>03:15</td>
    </tr>
  </tbody>
</table>


Not too bad! To see how well our model is doing, we can use the `show_results` method:

```python
learn.show_results()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
      <th>category_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos xxmaj there 's a sign on xxmaj the xxmaj lost xxmaj highway that says : \n\n * major xxup spoilers xxup ahead * \n\n ( but you already knew that , did n't you ? ) \n\n xxmaj since there 's a great deal of people that apparently did not get the point of this movie , xxmaj i 'd like to contribute my interpretation of why the plot makes perfect sense . xxmaj as others have pointed out , one single viewing of this movie is not sufficient . xxmaj if you have the xxup dvd of xxup md , you can " cheat " by looking at xxmaj david xxmaj lynch 's " top 10 xxmaj hints to xxmaj unlocking xxup md " ( but only upon second or third viewing , please . ) ;) \n\n xxmaj first of all , xxmaj mulholland xxmaj drive is</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxbos ( some spoilers included : ) \n\n xxmaj although , many commentators have called this film surreal , the term fits poorly here . xxmaj to quote from xxmaj encyclopedia xxmaj xxunk 's , surreal means : \n\n " fantastic or incongruous imagery " : xxmaj one need n't explain to the unimaginative how many ways a plucky ten - year - old boy at large and seeking his fortune in the driver 's seat of a red xxmaj mustang could be fantastic : those curious might read xxmaj james xxmaj kincaid ; but if you asked said lad how he were incongruous behind the wheel of a sports car , he 'd surely protest , " no way ! " xxmaj what fantasies and incongruities the film offers mostly appear within the first fifteen minutes . xxmaj thereafter we get more iterations of the same , in an</td>
      <td>pos</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxbos xxmaj hearkening back to those " good xxmaj old xxmaj days " of 1971 , we can vividly recall when we were treated with a whole xxmaj season of xxmaj charles xxmaj chaplin at the xxmaj cinema . xxmaj that 's what the promotional guy called it when we saw him on somebody 's old talk show . ( we ca n't recall just whose it was ; either xxup merv xxup griffin or xxup woody xxup woodbury , one or the other ! ) xxmaj the guest talked about xxmaj sir xxmaj charles ' career and how his films had been out of circulation ever since the 1952 exclusion of the former " little xxmaj tramp ' from xxmaj los xxmaj xxunk xxmaj xxunk on the grounds of his being an " undesirable xxmaj alien " . ( no xxmaj schultz , he 's xxup not from another</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>xxbos " buffalo xxmaj bill , xxmaj hero of the xxmaj far xxmaj west " director xxmaj mario xxmaj costa 's unsavory xxmaj spaghetti western " the xxmaj beast " with xxmaj klaus xxmaj kinski could only have been produced in xxmaj europe . xxmaj hollywood would never dared to have made a western about a sexual predator on the prowl as the protagonist of a movie . xxmaj never mind that xxmaj kinski is ideally suited to the role of ' crazy ' xxmaj johnny . xxmaj he plays an individual entirely without sympathy who is ironically dressed from head to toe in a white suit , pants , and hat . xxmaj this low - budget oater has nothing appetizing about it . xxmaj the typically breathtaking xxmaj spanish scenery around xxmaj almeria is nowhere in evidence . xxmaj instead , xxmaj costa and his director of photography</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>4</th>
      <td>xxbos xxmaj if you 've seen the trailer for this movie , you pretty much know what to expect , because what you see here is what you get . xxmaj and even if you have n't seen the previews , it wo n't take you long to pick up on what you 're in for-- specifically , a good time and plenty of xxunk from this clever satire of ` reality xxup tv ' shows and ` buddy xxmaj cop ' movies , ` showtime , ' directed by xxmaj tom xxmaj dey , starring xxmaj robert xxmaj de xxmaj niro and xxmaj eddie xxmaj murphy . \n\n\t xxmaj mitch xxmaj preston ( de xxmaj niro ) is a detective with the xxup l.a.p.d . , and he 's good at what he does ; but working a case one night , things suddenly go south when another cop</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xxbos * xxmaj some spoilers * \n\n xxmaj this movie is sometimes subtitled " life xxmaj everlasting . " xxmaj that 's often taken as reference to the final scene , but more accurately describes how dead and buried this once - estimable series is after this sloppy and illogical send - off . \n\n xxmaj there 's a " hey kids , let 's put on a show air " about this telemovie , which can be endearing in spots . xxmaj some fans will feel like insiders as they enjoy picking out all the various cameo appearances . xxmaj co - writer , co - producer xxmaj tom xxmaj fontana and his pals pack the goings - on with friends and favorites from other shows , as well as real xxmaj baltimore personages . \n\n xxmaj that 's on top of the returns of virtually all the members</td>
      <td>neg</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>6</th>
      <td>xxbos ( caution : several spoilers ) \n\n xxmaj someday , somewhere , there 's going to be a post - apocalyptic movie made that does n't stink . xxmaj unfortunately , xxup the xxup postman is not that movie , though i have to give it credit for trying . \n\n xxmaj kevin xxmaj costner plays somebody credited only as " the xxmaj postman . " xxmaj he 's not actually a postman , just a wanderer with a mule in the wasteland of a western xxmaj america devastated by some unspecified catastrophe . xxmaj he trades with isolated villages by performing xxmaj shakespeare . xxmaj suddenly a pack of bandits called the xxmaj holnists , the self - declared warlords of the xxmaj west , descend upon a village that xxmaj costner 's visiting , and their evil leader xxmaj gen . xxmaj bethlehem ( will xxmaj patton</td>
      <td>neg</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>7</th>
      <td>xxbos xxmaj in a style reminiscent of the best of xxmaj david xxmaj lean , this romantic love story sweeps across the screen with epic proportions equal to the vast desert regions against which it is set . xxmaj it 's a film which purports that one does not choose love , but rather that it 's love that does the choosing , regardless of who , where or when ; and furthermore , that it 's a matter of the heart often contingent upon prevailing conditions and circumstances . xxmaj and thus is the situation in ` the xxmaj english xxmaj patient , ' directed by xxmaj anthony xxmaj minghella , the story of two people who discover passion and true love in the most inopportune of places and times , proving that when it is predestined , love will find a way . \n\n xxmaj it 's xxup</td>
      <td>pos</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>8</th>
      <td>xxbos xxmaj no one is going to mistake xxup the xxup squall for a good movie , but it sure is a memorable one . xxmaj once you 've taken in xxmaj myrna xxmaj loy 's performance as xxmaj nubi the hot - blooded gypsy girl you 're not likely to forget the experience . xxmaj when this film was made the exotically beautiful xxmaj miss xxmaj loy was still being cast as foreign vixens , often xxmaj asian and usually sinister . xxmaj she 's certainly an eyeful here . xxmaj it appears that her skin was darkened and her hair was curled . xxmaj in most scenes she 's barefoot and wearing little more than a skirt and a loose - fitting peasant blouse , while in one scene she wears nothing but a patterned towel . i suppose xxmaj i 'm focusing on xxmaj miss xxmaj loy</td>
      <td>neg</td>
      <td>neg</td>
    </tr>
  </tbody>
</table>


And we can predict on new texts quite easily:

```python
learn.predict("I really liked that movie!")
```








    ('pos', tensor(1), tensor([0.0092, 0.9908]))



Here we can see the model has considered the review to be positive. The second part of the result is the index of "pos" in our data vocabulary and the last part is the probabilities attributed to each class (99.1% for "pos" and 0.9% for "neg"). 

Now it's your turn! Write your own mini movie review, or copy one from the Internet, and we can see what this model thinks about it. 

### Using the data block API

We can also use the data block API to get our data in a [`DataLoaders`](/data.core.html#DataLoaders). This is a bit more advanced, so fell free to skip this part if you are not comfortable with learning new APIs just yet.

A datablock is built by giving the fastai library a bunch of information:

- the types used, through an argument called `blocks`: here we have images and categories, so we pass [`TextBlock`](/text.data.html#TextBlock) and [`CategoryBlock`](/data.block.html#CategoryBlock). To inform the library our texts are files in a folder, we use the `from_folder` class method.
- how to get the raw items, here our function [`get_text_files`](/data.transforms.html#get_text_files).
- how to label those items, here with the parent folder.
- how to split those items, here with the grandparent folder.

```python
imdb = DataBlock(blocks=(TextBlock.from_folder(path), CategoryBlock),
                 get_items=get_text_files,
                 get_y=parent_label,
                 splitter=GrandparentSplitter(valid_name='test'))
```

This only gives a blueprint on how to assemble the data. To actually create it, we need to use the `dataloaders` method:

```python
dls = imdb.dataloaders(path)
```

## The ULMFiT approach

The pretrained model we used in the previous section is called a language model. It was pretrained on Wikipedia on the task of guessing the next word, after reading all the words before. We got great results by directly fine-tuning this language model to a movie review classifier, but with one extra step, we can do even better: the Wikipedia English is slightly different from the IMDb English. So instead of jumping directly to the classifier, we could fine-tune our pretrained language model to the IMDb corpus and *then* use that as the base for our classifier.

One reason, of course, is that it is helpful to understand the foundations of the models that you are using. But there is another very practical reason, which is that you get even better results if you fine tune the (sequence-based) language model prior to fine tuning the classification model. For instance, in the IMDb sentiment analysis task, the dataset includes 50,000 additional movie reviews that do not have any positive or negative labels attached in the unsup folder. We can use all of these reviews to fine tune the pretrained language model — this will result in a language model that is particularly good at predicting the next word of a movie review. In contrast, the pretrained model was trained only on Wikipedia articles.

The whole process is summarized by this picture:

![ULMFit process](/images/ulmfit.png)

### Fine-tuning a language model on IMDb

We can get our texts in a [`DataLoaders`](/data.core.html#DataLoaders) suitable for language modeling very easily:

```python
dls_lm = TextDataLoaders.from_folder(path, is_lm=True, valid_pct=0.1)
```

We need to pass something for `valid_pct` otherwise this method will try to split the data by using the grandparent folder names. By passing `valid_pct=0.1`, we tell it to get a random 10% of those reviews for the validation set.

We can have a look at our data using `show_batch`. Here the task is to guess the next word, so we can see the targets have all shifted one word to the right.

```python
dls_lm.show_batch(max_n=5)
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
      <td>xxbos xxmaj about thirty minutes into the film , i thought this was one of the weakest " xxunk ever because it had the usual beginning ( a murder happening , then xxmaj columbo coming , inspecting everything and interrogating the main suspect ) squared ! xxmaj it was boring because i thought i knew everything already . \n\n xxmaj but then there was a surprising twist that turned this episode into</td>
      <td>xxmaj about thirty minutes into the film , i thought this was one of the weakest " xxunk ever because it had the usual beginning ( a murder happening , then xxmaj columbo coming , inspecting everything and interrogating the main suspect ) squared ! xxmaj it was boring because i thought i knew everything already . \n\n xxmaj but then there was a surprising twist that turned this episode into a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>yeon . xxmaj these two girls were magical on the screen . i will certainly be looking into their other films . xxmaj xxunk xxmaj jeong - ah is xxunk cheerful and hauntingly evil as the stepmother . xxmaj finally , xxmaj xxunk - su xxmaj kim gives an excellent performance as the weary , broken father . \n\n i truly love this film . xxmaj if you have yet to see</td>
      <td>. xxmaj these two girls were magical on the screen . i will certainly be looking into their other films . xxmaj xxunk xxmaj jeong - ah is xxunk cheerful and hauntingly evil as the stepmother . xxmaj finally , xxmaj xxunk - su xxmaj kim gives an excellent performance as the weary , broken father . \n\n i truly love this film . xxmaj if you have yet to see '</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tends to be tedious whenever there are n't any hideous monsters on display . xxmaj luckily the gutsy killings and eerie set designs ( by no less than xxmaj bill xxmaj paxton ! ) compensate for a lot ! a nine - headed expedition is send ( at hyper speed ) to the unexplored regions of space to find out what happened to a previously vanished spaceship and its crew . xxmaj</td>
      <td>to be tedious whenever there are n't any hideous monsters on display . xxmaj luckily the gutsy killings and eerie set designs ( by no less than xxmaj bill xxmaj paxton ! ) compensate for a lot ! a nine - headed expedition is send ( at hyper speed ) to the unexplored regions of space to find out what happened to a previously vanished spaceship and its crew . xxmaj bad</td>
    </tr>
    <tr>
      <th>3</th>
      <td>movie just sort of meanders around and nothing happens ( i do n't mean in terms of plot - no plot is fine , but no action ? xxmaj come on . ) xxmaj in hindsight , i should have expected this - after all , how much can really happen between 4 teens and a bear ? xxmaj so although special effects , acting , etc are more or less on</td>
      <td>just sort of meanders around and nothing happens ( i do n't mean in terms of plot - no plot is fine , but no action ? xxmaj come on . ) xxmaj in hindsight , i should have expected this - after all , how much can really happen between 4 teens and a bear ? xxmaj so although special effects , acting , etc are more or less on par</td>
    </tr>
    <tr>
      <th>4</th>
      <td>greetings again from the darkness . xxmaj writer / xxmaj director ( and xxmaj wes xxmaj anderson collaborator ) xxmaj noah xxmaj baumbach presents a semi - autobiographical therapy session where he unleashes the anguish and turmoil that has carried over from his childhood . xxmaj the result is an amazing insight into what many people go through in a desperate attempt to try and make their family work . \n\n xxmaj</td>
      <td>again from the darkness . xxmaj writer / xxmaj director ( and xxmaj wes xxmaj anderson collaborator ) xxmaj noah xxmaj baumbach presents a semi - autobiographical therapy session where he unleashes the anguish and turmoil that has carried over from his childhood . xxmaj the result is an amazing insight into what many people go through in a desperate attempt to try and make their family work . \n\n xxmaj the</td>
    </tr>
  </tbody>
</table>


Then we have a convenience method to directly grab a [`Learner`](/learner.html#Learner) from it, using the [`AWD_LSTM`](/text.models.awdlstm.html#AWD_LSTM) architecture like before. We use accuracy and perplexity as metrics (the later is the exponential of the loss) and we set a default weight decay of 0.1. `to_fp16` puts the [`Learner`](/learner.html#Learner) in mixed precision, which is going to help speed up training on GPUs that have Tensor Cores.

```python
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], path=path, wd=0.1).to_fp16()
```

By default, a pretrained [`Learner`](/learner.html#Learner) is in a frozen state, meaning that only the head of the model will train while the body stays frozen. We show you what is behind the fine_tune method here and use a fit_one_cycle method to fit the model:

```python
learn.fit_one_cycle(1, 1e-2)
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
      <td>4.120048</td>
      <td>3.912788</td>
      <td>0.299565</td>
      <td>50.038246</td>
      <td>11:39</td>
    </tr>
  </tbody>
</table>


This model takes a while to train, so it's a good opportunity to talk about saving intermediary results. 

You can easily save the state of your model like so:

```python
learn.save('1epoch')
```

It will create a file in `learn.path/models/` named "1epoch.pth". If you want to load your model on another machine after creating your [`Learner`](/learner.html#Learner) the same way, or resume training later, you can load the content of this file with:

```python
learn = learn.load('1epoch')
```

We can them fine-tune the model after unfreezing:

```python
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3)
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
      <td>3.893486</td>
      <td>3.772820</td>
      <td>0.317104</td>
      <td>43.502548</td>
      <td>12:37</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.820479</td>
      <td>3.717197</td>
      <td>0.323790</td>
      <td>41.148880</td>
      <td>12:30</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.735622</td>
      <td>3.659760</td>
      <td>0.330321</td>
      <td>38.851997</td>
      <td>12:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.677086</td>
      <td>3.624794</td>
      <td>0.333960</td>
      <td>37.516987</td>
      <td>12:12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.636646</td>
      <td>3.601300</td>
      <td>0.337017</td>
      <td>36.645859</td>
      <td>12:05</td>
    </tr>
    <tr>
      <td>5</td>
      <td>3.553636</td>
      <td>3.584241</td>
      <td>0.339355</td>
      <td>36.026001</td>
      <td>12:04</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.507634</td>
      <td>3.571892</td>
      <td>0.341353</td>
      <td>35.583862</td>
      <td>12:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.444101</td>
      <td>3.565988</td>
      <td>0.342194</td>
      <td>35.374371</td>
      <td>12:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3.398597</td>
      <td>3.566283</td>
      <td>0.342647</td>
      <td>35.384815</td>
      <td>12:11</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3.375563</td>
      <td>3.568166</td>
      <td>0.342528</td>
      <td>35.451500</td>
      <td>12:05</td>
    </tr>
  </tbody>
</table>


Once this is done, we save all of our model except the final layer that converts activations to probabilities of picking each token in our vocabulary. The model not including the final layer is called the *encoder*. We can save it with `save_encoder`:

```python
learn.save_encoder('finetuned')
```

> Jargon:Encoder: The model not including the task-specific final layer(s). It means much the same thing as *body* when applied to vision CNNs, but tends to be more used for NLP and generative models.

Before using this to fine-tune a classifier on the reviews, we can use our model to generate random reviews: since it's trained to guess what the next word of the sentence is, we can use it to write new reviews:

```python
TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]
```









```python
print("\n".join(preds))
```

    i liked this movie because of its story and characters . The story line was very strong , very good for a sci - fi film . The main character , Alucard , was very well developed and brought the whole story
    i liked this movie because i like the idea of the premise of the movie , the ( very ) convenient virus ( which , when you have to kill a few people , the " evil " machine has to be used to protect


### Training a text classifier

We can gather our data for text classification almost exactly like before:

```python
dls_clas = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test', text_vocab=dls_lm.vocab)
```

The main difference is that we have to use the exact same vocabulary as when we were fine-tuning our language model, or the weights learned won't make any sense. We pass that vocabulary with `text_vocab`.

Then we can define our text classifier like before:

```python
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
```

The difference is that before training it, we load the previous encoder:

```python
learn = learn.load_encoder('finetuned')
```

The last step is to train with discriminative learning rates and *gradual unfreezing*. In computer vision, we often unfreeze the model all at once, but for NLP classifiers, we find that unfreezing a few layers at a time makes a real difference.

```python
learn.fit_one_cycle(1, 2e-2)
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
      <td>0.347427</td>
      <td>0.184480</td>
      <td>0.929320</td>
      <td>00:33</td>
    </tr>
  </tbody>
</table>


In just one epoch we get the same result as our training in the first section, not too bad! We can pass `-2` to `freeze_to` to freeze all except the last two parameter groups:

```python
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
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
      <td>0.247763</td>
      <td>0.171683</td>
      <td>0.934640</td>
      <td>00:37</td>
    </tr>
  </tbody>
</table>


Then we can unfreeze a bit more, and continue training:

```python
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
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
      <td>0.193377</td>
      <td>0.156696</td>
      <td>0.941200</td>
      <td>00:45</td>
    </tr>
  </tbody>
</table>


And finally, the whole model!

```python
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
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
      <td>0.172888</td>
      <td>0.153770</td>
      <td>0.943120</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.161492</td>
      <td>0.155567</td>
      <td>0.942640</td>
      <td>00:57</td>
    </tr>
  </tbody>
</table>

