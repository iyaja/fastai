# Tutorial - Binary classification of chest X-rays
> In this tutorial we will build a classifier that distinguishes between chest X-rays with pneumothorax and chest X-rays without pneumothorax. The image data is loaded directly from the DICOM source files, so no prior DICOM data handling is needed.


```python
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *

import pydicom

import pandas as pd
```

To use `fastai.medical.imaging` you'll need to:

```bash
conda install pyarrow
pip install pydicom kornia opencv-python scikit-image
```

## Download and import of X-ray DICOM files

First, we will use the [`untar_data`](/data.external.html#untar_data) function to download the _siim_small_ folder containing a subset (250 DICOM files, \~30MB) of the [SIIM-ACR Pneumothorax Segmentation](https://doi.org/10.1007/s10278-019-00299-9) \[1\] dataset.
The downloaded _siim_small_ folder will be stored in your _\~/.fastai/data/_ directory. The variable `pneumothorax-source` will store the absolute path to the _siim_small_ folder as soon as the download is complete.

```python
pneumothorax_source = untar_data(URLs.SIIM_SMALL)
```

The _siim_small_ folder has the following directory/file structure:

![siim_folder_structure.jpg](/images/siim_folder_structure.jpeg)

### Plotting the DICOM data

To analyze our dataset, we load the paths to the DICOM files with the [`get_dicom_files`](/medical.imaging.html#get_dicom_files) function. When calling the function, we append _train/_ to the `pneumothorax_source` path to choose the folder where the DICOM files are located. We store the path to each DICOM file in the `items` list.

```python
items = get_dicom_files(pneumothorax_source/f"train/")
```

Next, we split the `items` list into a train `trn` and validation `val` list using the [`RandomSplitter`](/data.transforms.html#RandomSplitter) function:

```python
trn,val = RandomSplitter()(items)
```

To plot an X-ray, we can select an entry in the `items` list and load the DICOM file with `dcmread`. Then, we can plot it with the function `show`.

```python
patient = 3
xray_sample = dcmread(items[patient])
xray_sample.show()
```


![png](output_14_0.png)


Next, we need to load the labels for the dataset. We import the _labels.csv_ file using pandas and print the first five entries. The **file** column shows the relative path to the _.dcm_ file and the **label** column indicates whether the chest x-ray has a pneumothorax or not.

```python
df = pd.read_csv(pneumothorax_source/f"labels.csv")
df.head()
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
      <th>file</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>train/No Pneumothorax/000000.dcm</td>
      <td>No Pneumothorax</td>
    </tr>
    <tr>
      <td>1</td>
      <td>train/Pneumothorax/000001.dcm</td>
      <td>Pneumothorax</td>
    </tr>
    <tr>
      <td>2</td>
      <td>train/No Pneumothorax/000002.dcm</td>
      <td>No Pneumothorax</td>
    </tr>
    <tr>
      <td>3</td>
      <td>train/Pneumothorax/000003.dcm</td>
      <td>Pneumothorax</td>
    </tr>
    <tr>
      <td>4</td>
      <td>train/Pneumothorax/000004.dcm</td>
      <td>Pneumothorax</td>
    </tr>
  </tbody>
</table>
</div>



Now, we use the [`DataBlock`](/data.block.html#DataBlock) class to prepare the DICOM data for training.

```python
pneumothorax = DataBlock(blocks=(ImageBlock(cls=PILDicom), CategoryBlock),
                   get_x=lambda x:pneumothorax_source/f"{x[0]}",
                   get_y=lambda x:x[1],
                   batch_tfms=aug_transforms(size=224))
```

Additionally, we plot a first batch with the specified transformations:

```python
dls = pneumothorax.dataloaders(df.values)
dls.show_batch(max_n=16)
```


![png](output_20_0.png)


## Training

We can then use the [`cnn_learner`](/vision.learner.html#cnn_learner) function and initiate the training.

```python
learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.fit_one_cycle(1)
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
      <td>1.250138</td>
      <td>1.026524</td>
      <td>0.560000</td>
      <td>00:03</td>
    </tr>
  </tbody>
</table>


```python
learn.predict(pneumothorax_source/f"train/Pneumothorax/000004.dcm")
```








    ('Pneumothorax', tensor(1), tensor([0.2858, 0.7142]))



```python
tta = learn.tta(use_max=True)
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
  <progress value='4' class='' max='4', style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4/4 00:02<00:00]
</div>



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
  <progress value='1' class='' max='1', style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1/1 00:00<00:00]
</div>



```python
learn.show_results(max_n=16)
```






![png](output_26_1.png)


```python
interp = Interpretation.from_learner(learn)
```





```python
interp.plot_top_losses(2)
```


![png](output_28_0.png)


_**Citations:**_

\[1\] _Filice R et al. Crowdsourcing pneumothorax annotations using machine learning annotations on the NIH chest X-ray dataset.  J Digit Imaging (2019). https://doi.org/10.1007/s10278-019-00299-9_
