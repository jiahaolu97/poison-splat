# Download DTU Dataset

Thanks to authors of [2.5DGS](https://github.com/hugoycj/2.5d-gaussian-splatting) [<sup>1,2</sup>](#reference) who provided the data downloading guidance.


Download preprocessed [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) data[<sup>3</sup>](#reference) provided by [NeuS](https://www.dropbox.com/sh/w0y8bbdmxzik3uk/AAAaZffBiJevxQzRskoOYcyja?e=1&dl=0)


The data is organized as follows:
```
<model_id>
|-- cameras_xxx.npz    # camera parameters
|-- image
    |-- 000000.png        # target image for each view
    |-- 000001.png
    ...
|-- mask
    |-- 000000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 000001.png
    ...
```


## Reference

<div id="reference"></div>

- [1] Ye C, Nie Y, Chang J, et al. GauStudio: A Modular Framework for 3D Gaussian Splatting and Beyond[J]. arXiv preprint arXiv:2403.19632, 2024.

- [2] Huang B, Yu Z, Chen A, et al. 2d gaussian splatting for geometrically accurate radiance fields[C]//ACM SIGGRAPH 2024 Conference Papers. 2024: 1-11.

- [3] Jensen R, Dahl A, Vogiatzis G, et al. Large scale multi-view stereopsis evaluation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2014: 406-413.