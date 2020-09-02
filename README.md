<h2 align="center">mosaic-closest</h2>
<p align="center"><b>Create a Mosaic of images that builds up an image</b></p>
<br>

<h2>Table of Contents</h2>
<!-- TOC -->

- [Introduction](#introduction)
- [Usage](#usage)
- [Contributors](#contributors)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [License](#license)

<!-- /TOC -->

## Introduction

[2020.09.02] Further research in image matching: [Image Matching from Handcrafted to Deep Features: A Survey](https://link.springer.com/article/10.1007/s11263-020-01359-2).

Create a Mosaic of images that builds up an image.

My approach utilises two metrics to evaluate the **similarities in RGB colourspace** between two images:

1. Channel-wise luminances. We compare the luminances in three RGB channels of the images in the dataset against those of the chosen patch from the original input. Our goal is to find the image with minimum luminance differences.
2. Euclidean distances in Channel-wise histograms. We compare the Euclidean distances of histograms in three RGB channels of the images in the dataset against those of the chosen patch from the original input. Our goal is to find the image with minimum Euclidean distances.

We combine both metrics and decide on the image only if it satisfies both metrics. For the images in dataset to be replace the patch, we resize them to 32×32.

For detailed information, here is my report: [Creating A Mosaic of Several Tiny Images](https://github.com/JinhangZhu/mosaic-closest/blob/master/report/Creating%20A%20Mosaic%20of%20Several%20Tiny%20Images.pdf). Check it out!

<div align="center">
<img src="https://i.loli.net/2019/10/11/MLte568RyZ7j1JN.png" height="200px" alt="original" >
<img src="https://i.loli.net/2019/10/11/clMqgAhaR5TeibJ.png" height="200px" alt="mosaic" >
</div>

## Usage

- Put a new colored or grayscale image in the directory of the .ipynb file.

- Name the image `original.jpg`

- Place resource images in a folder named 'images' and the folder should be in the same directory.

**NB**: If NoneType error exists, make the threshold higher to make it less strict.

**NB**: If this.ipynb gets stuck while running using Jupyter Notebook. Please use command window to run mosaic.py.

COMMAND LINE: 

```python
python mosaic.py
```

## Contributors

<a href="https://github.com/jinhangzhu/mosaic-closest/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=jinhangzhu/mosaic-closest" />
</a>


## Maintainers

[Jinhang Zhu](https://github.com/JinhangZhu)

## Thanks

- https://stackoverflow.com/questions/34264710/what-is-the-point-of-floatinf-in-python

- https://docs.python.org/2/library/multiprocessing.html#multiprocessing-programming

- https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html

- https://github.com/pythonml/mosaic/blob/master/main.py

- https://stackoverflow.com/questions/38598118/difference-between-plt-show-and-cv2-imshow

- https://stackoverflow.com/questions/15393216/create-multidimensional-zeros-python

- https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

- https://stackoverflow.com/questions/47313732/jupyter-notebook-never-finishes-processing-using-multiprocessing-python-3

## License

- [MIT](https://opensource.org/licenses/MIT)
