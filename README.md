### SAPE
### [Project page](https://amirhertz.github.io/sape/)  &nbsp; &nbsp; [Paper](https://arxiv.org/pdf/2104.09125.pdf)

Official implementation for the paper "SAPE: Spatially-Adaptive Progressive Encoding for Neural Optimization".


### Environment
Create an anaconda environment and install [Pytorch](https://pytorch.org/). 
Install other dependencies:

```
conda env update --file environment.yml
```

### Tasks
Running examples:
```
python tasks_func_1d.py
python tasks_image_2d.py <path to an image file>
python tasks_silhouette_2d.py <path to a silhouette image file>
python tasks_occupancy_3d.py <path to a mesh file>
```

See ./assets directory for possible input files.

Models and other outputs (images, optimization animation, etc.) will be saved under ./assets/checkpoints/<task_name>/<file_name>


### Citation

If you find this code useful for your research, please cite our paper.

```bibtex
@inproceedings{hertz2021sape,
  title={SAPE: Spatially-Adaptive Progressive Encoding for Neural Optimization},
  author={Hertz, Amir and Perel, Or and Giryes, Raja and Sorkine-Hornung, Olga and Cohen-Or, Daniel},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```