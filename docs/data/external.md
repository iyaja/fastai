# External data
> Helper functions to download the fastai datasets


 A complete list of datasets that are available by default inside the library are: 

**Main datasets**:
1.    **ADULT_SAMPLE**: A small of the [adults dataset](https://archive.ics.uci.edu/ml/datasets/Adult) to  predict whether income exceeds $50K/yr based on census data. 
-    **BIWI_SAMPLE**: A [BIWI kinect headpose database](https://www.kaggle.com/kmader/biwi-kinect-head-pose-database). The dataset contains over 15K images of 20 people (6 females and 14 males - 4 people were recorded twice). For each frame, a depth image, the corresponding rgb image (both 640x480 pixels), and the annotation is provided. The head pose range covers about +-75 degrees yaw and +-60 degrees pitch. 
1.    **CIFAR**: The famous [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset which consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.      
1.    **COCO_SAMPLE**: A sample of the [coco dataset](http://cocodataset.org/#home) for object detection. 
1.    **COCO_TINY**: A tiny version of the [coco dataset](http://cocodataset.org/#home) for object detection.
-    **HUMAN_NUMBERS**: A synthetic dataset consisting of human number counts in text such as one, two, three, four.. Useful for experimenting with Language Models.
-    **IMDB**: The full [IMDB sentiment analysis dataset](https://ai.stanford.edu/~amaas/data/sentiment/).          

-    **IMDB_SAMPLE**: A sample of the full [IMDB sentiment analysis dataset](https://ai.stanford.edu/~amaas/data/sentiment/). 
-    **ML_SAMPLE**: A movielens sample dataset for recommendation engines to recommend movies to users.            
-    **ML_100k**: The movielens 100k dataset for recommendation engines to recommend movies to users.             
-    **MNIST_SAMPLE**: A sample of the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consisting of handwritten digits.        
-    **MNIST_TINY**: A tiny version of the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consisting of handwritten digits.                   
-    **MNIST_VAR_SIZE_TINY**:  
-    **PLANET_SAMPLE**: A sample of the planets dataset from the Kaggle competition [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space).
-    **PLANET_TINY**: A tiny version  of the planets dataset from the Kaggle competition [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) for faster experimentation and prototyping.
-    **IMAGENETTE**: A smaller version of the [imagenet dataset](http://www.image-net.org/) pronounced just like 'Imagenet', except with a corny inauthentic French accent. 
-    **IMAGENETTE_160**: The 160px version of the Imagenette dataset.      
-    **IMAGENETTE_320**: The 320px version of the Imagenette dataset. 
-    **IMAGEWOOF**: Imagewoof is a subset of 10 classes from Imagenet that aren't so easy to classify, since they're all dog breeds.
-    **IMAGEWOOF_160**: 160px version of the ImageWoof dataset.        
-    **IMAGEWOOF_320**: 320px version of the ImageWoof dataset.
-    **IMAGEWANG**: Imagewang contains Imagenette and Imagewoof combined, but with some twists that make it into a tricky semi-supervised unbalanced classification problem
-    **IMAGEWANG_160**: 160px version of Imagewang.        
-    **IMAGEWANG_320**: 320px version of Imagewang. 

**Kaggle competition datasets**:
1. **DOGS**: Image dataset consisting of dogs and cats images from [Dogs vs Cats kaggle competition](https://www.kaggle.com/c/dogs-vs-cats). 

**Image Classification datasets**:
1.    **CALTECH_101**: Pictures of objects belonging to 101 categories. About 40 to 800 images per category. Most categories have about 50 images. Collected in September 2003 by Fei-Fei Li, Marco Andreetto, and Marc 'Aurelio Ranzato.
1.    CARS: The [Cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) contains 16,185 images of 196 classes of cars.   
1.    **CIFAR_100**: The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class.   
1.    **CUB_200_2011**: Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations
1.    **FLOWERS**: 17 category [flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/) by gathering images from various websites.
1.    **FOOD**:         
1.    **MNIST**: [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consisting of handwritten digits.      
1.    **PETS**: A 37 category [pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) with roughly 200 images for each class.

**NLP datasets**:
1.    **AG_NEWS**: The AG News corpus consists of news articles from the AG’s corpus of news articles on the web pertaining to the 4 largest classes. The dataset contains 30,000 training and 1,900 testing examples for each class.
1.    **AMAZON_REVIEWS**: This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014.
1.    **AMAZON_REVIEWS_POLARITY**: Amazon reviews dataset for sentiment analysis.
1.    **DBPEDIA**: The DBpedia ontology dataset contains 560,000 training samples and 70,000 testing samples for each of 14 nonoverlapping classes from DBpedia. 
1.    **MT_ENG_FRA**: Machine translation dataset from English to French.
1.    **SOGOU_NEWS**: [The Sogou-SRR](http://www.thuir.cn/data-srr/) (Search Result Relevance) dataset was constructed to support researches on search engine relevance estimation and ranking tasks.
1.    **WIKITEXT**: The [WikiText language modeling dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.  
1.    **WIKITEXT_TINY**: A tiny version of the WIKITEXT dataset.
1.    **YAHOO_ANSWERS**: YAHOO's question answers dataset.
1.    **YELP_REVIEWS**: The [Yelp dataset](https://www.yelp.com/dataset) is a subset of YELP businesses, reviews, and user data for use in personal, educational, and academic purposes
1.    **YELP_REVIEWS_POLARITY**: For sentiment classification on YELP reviews.


**Image localization datasets**:
1.    **BIWI_HEAD_POSE**: A [BIWI kinect headpose database](https://www.kaggle.com/kmader/biwi-kinect-head-pose-database). The dataset contains over 15K images of 20 people (6 females and 14 males - 4 people were recorded twice). For each frame, a depth image, the corresponding rgb image (both 640x480 pixels), and the annotation is provided. The head pose range covers about +-75 degrees yaw and +-60 degrees pitch. 
1.    **CAMVID**: Consists of driving labelled dataset for segmentation type models.
1.    **CAMVID_TINY**: A tiny camvid dataset for segmentation type models.
1.    **LSUN_BEDROOMS**: [Large-scale Image Dataset](https://arxiv.org/abs/1506.03365) using Deep Learning with Humans in the Loop
1.    **PASCAL_2007**: [Pascal 2007 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) to recognize objects from a number of visual object classes in realistic scenes.
1.    **PASCAL_2012**: [Pascal 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) to recognize objects from a number of visual object classes in realistic scenes.

**Audio classification**:
1. **MACAQUES**: [7285 macaque coo calls](https://datadryad.org/stash/dataset/doi:10.5061/dryad.7f4p9) across 8 individuals from [Distributed acoustic cues for caller identity in macaque vocalization](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4806230).
2. **ZEBRA_FINCH**: [3405 zebra finch calls](https://ndownloader.figshare.com/articles/11905533/versions/1) classified [across 11 call types](https://link.springer.com/article/10.1007/s10071-015-0933-6). Additional labels include name of individual making the vocalization and its age.

**Medical Imaging datasets**:
1. **SIIM_SMALL**: A smaller version of the [SIIM dataset](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview) where the objective is to classify pneumothorax from a set of chest radiographic images.

**Pretrained models**:
1.    **OPENAI_TRANSFORMER**: The GPT2 Transformer pretrained weights.
1.    **WT103_FWD**: The WikiText-103 forward language model weights.
1.    **WT103_BWD**: The WikiText-103 backward language model weights.

To download any of the datasets or pretrained weights, simply run [`untar_data`](/data.external.html#untar_data) by passing any dataset name mentioned above like so: 

```python 
path = untar_data(URLs.PETS)
path.ls()
> > (#7393) [Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/keeshond_34.jpg'),Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/Siamese_178.jpg'),Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/german_shorthaired_94.jpg'),Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/Abyssinian_92.jpg'),Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/basset_hound_111.jpg'),Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/Russian_Blue_194.jpg'),Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/staffordshire_bull_terrier_91.jpg'),Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/Persian_69.jpg'),Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/english_setter_33.jpg'),Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/Russian_Blue_155.jpg')...]
```

To download model pretrained weights:```python path = untar_data(URLs.PETS)
path.ls()

>> (#2) [Path('/home/ubuntu/.fastai/data/wt103-bwd/itos_wt103.pkl'),Path('/home/ubuntu/.fastai/data/wt103-bwd/lstm_bwd.pth')]
```


<h2 id="Config" class="doc_header"><code>class</code> <code>Config</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/external.py#L10" class="source_link" style="float:right">[source]</a></h2>

> <code>Config</code>()

Setup config at `~/.fastai` unless it exists already.


If a config file doesn't exist already, it is always created at `~/.fastai/config.yml` location by default whenever an instance of the [`Config`](/data.external.html#Config) class is created. Here is a quick example to explain: 

```python
config_file = Path("~/.fastai/config.yml").expanduser()
if config_file.exists(): os.remove(config_file)
assert not config_file.exists()

config = Config()
assert config_file.exists()
```

The config is now available as `config.d`:

```python
config.d
```




    {'archive_path': '/home/jhoward/.fastai/archive',
     'data_path': '/home/jhoward/.fastai/data',
     'model_path': '/home/jhoward/.fastai/models',
     'storage_path': '/tmp',
     'version': 2}



As can be seen, this is a basic config file that consists of `data_path`, `model_path`, `storage_path` and `archive_path`. 
All future downloads occur at the paths defined in the config file based on the type of download. For example, all future fastai datasets are downloaded to the `data_path` while all pretrained model weights are download to `model_path` unless the default download location is updated.

Please note that it is possible to update the default path locations in the config file. Let's first create a backup of the config file, then update the config to show the changes and re update the new config with the backup file. 

```python
if config_file.exists(): shutil.move(config_file, config_bak)
config['archive_path'] = Path(".")
config.save()
```

```python
config = Config()
config.d
```




    {'archive_path': '.',
     'data_archive_path': '/home/jhoward/.fastai/data',
     'data_path': '/home/jhoward/.fastai/data',
     'model_path': '/home/jhoward/.fastai/models',
     'storage_path': '/tmp',
     'version': 2}



The `archive_path` has been updated to `"."`. Now let's remove any updates we made to Config file that we made for the purpose of this example. 

```python
if config_bak.exists(): shutil.move(config_bak, config_file)
config = Config()
config.d
```




    {'archive_path': '/home/jhoward/.fastai/archive',
     'data_archive_path': '/home/jhoward/.fastai/data',
     'data_path': '/home/jhoward/.fastai/data',
     'model_path': '/home/jhoward/.fastai/models',
     'storage_path': '/tmp',
     'version': 2}




<h2 id="URLs" class="doc_header"><code>class</code> <code>URLs</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/external.py#L59" class="source_link" style="float:right">[source]</a></h2>

> <code>URLs</code>()

Global constants for dataset and model URLs.


The default local path is at `~/.fastai/archive/` but this can be updated by passing a different `c_key`. Note: `c_key` should be one of `'archive_path', 'data_archive_path', 'data_path', 'model_path', 'storage_path'`.

```python
url = URLs.PETS
local_path = URLs.path(url)
test_eq(local_path.parent, Config()['archive']); 
local_path
```




    Path('/home/jhoward/.fastai/archive/oxford-iiit-pet.tgz')



```python
local_path = URLs.path(url, c_key='model')
test_eq(local_path.parent, Config()['model'])
local_path
```




    Path('/home/jhoward/.fastai/models/oxford-iiit-pet.tgz')



## Downloading


<h4 id="download_url" class="doc_header"><code>download_url</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/external.py#L154" class="source_link" style="float:right">[source]</a></h4>

> <code>download_url</code>(**`url`**, **`dest`**, **`overwrite`**=*`False`*, **`pbar`**=*`None`*, **`show_progress`**=*`True`*, **`chunk_size`**=*`1048576`*, **`timeout`**=*`4`*, **`retries`**=*`5`*)

Download `url` to `dest` unless it exists and not `overwrite`


The [`download_url`](/data.external.html#download_url) is a very handy function inside fastai! This function can be used to download any file from the internet to a location passed by `dest` argument of the function. It should not be confused, that this function can only be used to download fastai-files. That couldn't be further away from the truth. As an example, let's download the pets dataset from the actual source file: 

```python
fname = Path("./dog.jpg")
if fname.exists(): os.remove(fname)
url = "https://i.insider.com/569fdd9ac08a80bd448b7138?width=1100&format=jpeg&auto=webp"
download_url(url, fname)
assert fname.exists()
```





Let's confirm that the file was indeed downloaded correctly.

```python
from PIL import Image
```

```python
im = Image.open(fname)
plt.imshow(im);
```


![png](output_28_0.png)


As can be seen, the file has been downloaded to the local path provided in `dest` argument. Calling the function again doesn't trigger a download since the file is already there. This can be confirmed by checking that the last modified time of the file that is downloaded doesn't get updated. 

```python
if fname.exists(): last_modified_time = os.path.getmtime(fname)
download_url(url, fname)
test_eq(os.path.getmtime(fname), last_modified_time)
if fname.exists(): os.remove(fname)
```

We can also use the [`download_url`](/data.external.html#download_url) function to download the pet's dataset straight from the source by simply passing `https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz` in `url`. 


<h4 id="download_data" class="doc_header"><code>download_data</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/external.py#L188" class="source_link" style="float:right">[source]</a></h4>

> <code>download_data</code>(**`url`**, **`fname`**=*`None`*, **`c_key`**=*`'archive'`*, **`force_download`**=*`False`*)

Download `url` to `fname`.


The [`download_data`](/data.external.html#download_data) is a convenience function and a wrapper outside [`download_url`](/data.external.html#download_url) to download fastai files to the appropriate local path based on the `c_key`. 

If `fname` is None, it will default to the archive folder you have in your config file (or data, model if you specify a different `c_key`) followed by the last part of the url: for instance [`URLs.MNIST_SAMPLE`](/data.external.html#URLs.MNIST_SAMPLE) is `http://files.fast.ai/data/examples/mnist_sample.tgz` and the default value for `fname` will be `~/.fastai/archive/mnist_sample.tgz`.

If `force_download=True`, the file is alwayd downloaded. Otherwise, it's only when the file doesn't exists that the download is triggered.

```python
_get_check(URLs.PASCAL_2007),_get_check(URLs.PASCAL_2012)
```




    ([1637796771, '433b4706eb7c42bd74e7f784e3fdf244'],
     [2618908000, 'd90e29e54a4c76c0c6fba8355dcbaca5'])



### Extract


<h4 id="file_extract" class="doc_header"><code>file_extract</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/external.py#L215" class="source_link" style="float:right">[source]</a></h4>

> <code>file_extract</code>(**`fname`**, **`dest`**=*`None`*)

Extract `fname` to `dest` using `tarfile` or `zipfile`.


[`file_extract`](/data.external.html#file_extract) is used by default in [`untar_data`](/data.external.html#untar_data) to decompress the downloaded file. 


<h4 id="newest_folder" class="doc_header"><code>newest_folder</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/external.py#L231" class="source_link" style="float:right">[source]</a></h4>

> <code>newest_folder</code>(**`path`**)

Return newest folder on path



<h4 id="rename_extracted" class="doc_header"><code>rename_extracted</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/external.py#L237" class="source_link" style="float:right">[source]</a></h4>

> <code>rename_extracted</code>(**`dest`**)

Rename file if different from dest


let's rename the untar/unzip data if dest name is different from fname


<h4 id="untar_data" class="doc_header"><code>untar_data</code><a href="https://github.com/fastai/fastai/tree/master/fastai/data/external.py#L243" class="source_link" style="float:right">[source]</a></h4>

> <code>untar_data</code>(**`url`**, **`fname`**=*`None`*, **`dest`**=*`None`*, **`c_key`**=*`'data'`*, **`force_download`**=*`False`*, **`extract_func`**=*`file_extract`*)

Download `url` to `fname` if `dest` doesn't exist, and un-tgz or unzip to folder `dest`.


[`untar_data`](/data.external.html#untar_data) is a very powerful convenience function to download files from `url` to `dest`. The `url` can be a default `url` from the [`URLs`](/data.external.html#URLs) class or a custom url. If `dest` is not passed, files are downloaded at the `default_dest` which defaults to `~/.fastai/data/`.

This convenience function extracts the downloaded files to `dest` by default. In order, to simply download the files without extracting, pass the `noop` function as `extract_func`. 

Note, it is also possible to pass a custom `extract_func` to [`untar_data`](/data.external.html#untar_data) if the filetype doesn't end with `.tgz` or `.zip`. The `gzip` and `zip` files are supported by default and there is no need to pass custom `extract_func` for these type of files. 

Internally, if files are not available at `fname` location already which defaults to `~/.fastai/archive/`, the files get downloaded at `~/.fastai/archive` and are then extracted at `dest` location. If no `dest` is passed the `default_dest` to download the files is `~/.fastai/data`. If files are already available at the `fname` location but not available then a symbolic link is created for each file from `fname` location to `dest`.

Also, if `force_download` is set to `True`, files are re downloaded even if they exist. 

```python
from tempfile import TemporaryDirectory
```

```python
test_eq(untar_data(URLs.MNIST_SAMPLE), config.data/'mnist_sample')

with TemporaryDirectory() as d:
    d = Path(d)
    dest = untar_data(URLs.MNIST_TINY, fname='mnist_tiny.tgz', dest=d, force_download=True)
    assert Path('mnist_tiny.tgz').exists()
    assert (d/'mnist_tiny').exists()
    os.unlink('mnist_tiny.tgz')

#Test c_key
tst_model = config.model/'mnist_sample'
test_eq(untar_data(URLs.MNIST_SAMPLE, c_key='model'), tst_model)
assert not tst_model.with_suffix('.tgz').exists() #Archive wasn't downloaded in the models path
assert (config.archive/'mnist_sample.tgz').exists() #Archive was downloaded there
shutil.rmtree(tst_model)
```





Sometimes the extracted folder does not have the same name as the downloaded file.

```python
with TemporaryDirectory() as d:
    d = Path(d)
    untar_data(URLs.MNIST_TINY, fname='mnist_tiny.tgz', dest=d, force_download=True)
    Path('mnist_tiny.tgz').rename('nims_tini.tgz')
    p = Path('nims_tini.tgz')
    dest = Path('nims_tini')
    assert p.exists()
    file_extract(p, dest.parent)
    rename_extracted(dest)
    p.unlink()
    shutil.rmtree(dest)
```




