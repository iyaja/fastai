# Vision utils
> Some utils function to quickly download a bunch of images, check them and pre-resize them


```python

```

```python

```

```python
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    url = "https://www.fast.ai/images/jh-head.jpg"
    _download_image_inner(d, (125,url))
    assert (d/'00000125.jpg').is_file()
```


<h4 id="download_images" class="doc_header"><code>download_images</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/utils.py#L19" class="source_link" style="float:right">[source]</a></h4>

> <code>download_images</code>(**`dest`**, **`url_file`**=*`None`*, **`urls`**=*`None`*, **`max_pics`**=*`1000`*, **`n_workers`**=*`8`*, **`timeout`**=*`4`*)

Download images listed in text file `url_file` to path `dest`, at most `max_pics`


```python
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    url_file = d/'urls.txt'
    url_file.write("\n".join([f"https://www.fast.ai/images/{n}" for n in "jh-head.jpg thomas.JPG sg-head".split()]))
    download_images(d, url_file)
    for i in [0,2]: assert (d/f'0000000{i}.jpg').is_file()
    assert (d/f'00000001.JPG').is_file()
```






<h4 id="resize_to" class="doc_header"><code>resize_to</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/utils.py#L27" class="source_link" style="float:right">[source]</a></h4>

> <code>resize_to</code>(**`img`**, **`targ_sz`**, **`use_min`**=*`False`*)

Size to resize to, to hit `targ_sz` at same aspect ratio, in PIL coords (i.e w*h)


```python
class _FakeImg():
    def __init__(self, size): self.size=size
img = _FakeImg((200,500))

test_eq(resize_to(img, 400), [160,400])
test_eq(resize_to(img, 400, use_min=True), [400,1000])
```


<h4 id="verify_image" class="doc_header"><code>verify_image</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/utils.py#L35" class="source_link" style="float:right">[source]</a></h4>

> <code>verify_image</code>(**`fn`**)

Confirm that `fn` can be opened



<h4 id="verify_images" class="doc_header"><code>verify_images</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/utils.py#L45" class="source_link" style="float:right">[source]</a></h4>

> <code>verify_images</code>(**`fns`**)

Find images in `fns` that can't be opened



<h4 id="resize_image" class="doc_header"><code>resize_image</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/utils.py#L50" class="source_link" style="float:right">[source]</a></h4>

> <code>resize_image</code>(**`file`**, **`dest`**, **`max_size`**=*`None`*, **`n_channels`**=*`3`*, **`ext`**=*`None`*, **`img_format`**=*`None`*, **`resample`**=*`2`*, **`resume`**=*`False`*, **\*\*`kwargs`**)

Resize file to dest to max_size


```python
file = Path('images/puppy.jpg')
dest = Path('.')
resize_image(file, max_size=400, dest=dest)
im = Image.open(dest/file.name)
test_eq(im.shape[1],400)
(dest/file.name).unlink()
```


<h4 id="resize_images" class="doc_header"><code>resize_images</code><a href="https://github.com/fastai/fastai/tree/master/fastai/vision/utils.py#L69" class="source_link" style="float:right">[source]</a></h4>

> <code>resize_images</code>(**`path`**, **`max_workers`**=*`4`*, **`max_size`**=*`None`*, **`recurse`**=*`False`*, **`dest`**=*`Path('.')`*, **`n_channels`**=*`3`*, **`ext`**=*`None`*, **`img_format`**=*`None`*, **`resample`**=*`2`*, **`resume`**=*`None`*, **\*\*`kwargs`**)

Resize files on path recursevely to dest to max_size


```python
with tempfile.TemporaryDirectory() as d:
    dest = Path(d)/'resized_images'
    resize_images('images', max_size=100, dest=dest)
```




