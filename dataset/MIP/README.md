# Download Mip-Nerf 360 Dataset

Please download the Mip-NeRF 360 dataset processed by colmap from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) [<sup>1</sup>](#reference), and after unzipping "Dataset Pt. 1" and "Dataset Pt. 2", combine the scenes. Finally, the current directory should contain the following folders:

```
poison-splat
|---dataset
    |---MIP
        |---bicycle
        |   |---images
        |   |   |---<image 0>
        |   |   |---<image 1>
        |   |   |---...
        |   |---images_2
        |   |---images_4
        |   |---images_8
        |   |---sparse
        |       |---0
        |           |---cameras.bin
        |           |---images.bin
        |           |---points3D.bin
        |---bonsai
        |---...
```



## Reference

<div id="reference"></div>

- [1] Barron J T, Mildenhall B, Verbin D, et al. Mip-nerf 360: Unbounded anti-aliased neural radiance fields[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 5470-5479.