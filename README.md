# IHUdeLiver10 2D-3D registration

## Description

This repository contains code to automatically configure the projective geometry to generate
Digitally Rendered Radiographs (DRRs) from a CT scan.

This code is released with the [IHUdeLiver10 dataset](https://doi.org/10.57745/EUBXGH) (to be released) to render DRRs from CT scans in an standardized way.

The library used for DRR rendering is [DeepDRR](https://github.com/arcadelab/deepdrr), which depends on CUDA.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Installation

The package dependencies are listed in [requirements.txt](https://github.com/coolteemf/IHUdeLiver10-2D_3D-deformable-registration/blob/main/requirements.txt).

## Usage

The code is split between a [main](https://github.com/coolteemf/IHUdeLiver10-2D_3D-deformable-registration/blob/main/ihudeliver10_2d3d/main.py) and a [utils](https://github.com/coolteemf/IHUdeLiver10-2D_3D-deformable-registration/blob/main/ihudeliver10_2d3d/utils.py) file.

The main file takes command line arguments to generate the projection parameter files and optionally render DRRs to better understand the effect of projection parameters.

Example usage from the terminal:
```
python ./ihudeliver10_2d3d/main.py --volume_path "path/to/volume" --cuda_path "/usr/local/cuda/bin" --max_disp 150 --center x y z --roi_size x_size y_size z_size
```

## Contributing

Suggestions are welcome.\
If you encounter an issue or a bug, please feel free to raise an issue.

## Acknowledgements
This work was funded by the ANR (ANR-20-CE19-0015).\
The IHUdeliver10 was acquired by Juan Verde, MD., at the Strasbourg IHU, who provided
invaluable support during the development process.\
We thank the authors of the [DeepDRR](https://github.com/arcadelab/deepdrr) framework, which was instrumental to the development of this code.