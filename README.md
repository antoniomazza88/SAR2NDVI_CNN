# SAR2NDVI_CNN
[A CNN-Based Fusion Method for Feature Extraction from Sentinel Data](http://www.mdpi.com/2072-4292/10/2/236) 

A CNN is trained to perform the estimation of the NDVI, using coupled Sentinel-1 and Sentinel-2 time-series.


# Team Members

* [Giuseppe Scarpa](giscarpa@unina.it); 
* [Massimiliano Gargiulo](massimiliano.gargiulo@unina.it); 
* [Antonio Mazza](antonio.mazza@unina.it), contact person; 
* [Raffaele Gaetano](raffaele.gaetano@cirad.fr). 

# License 

Copyright (c) 2018 [Image Processing Research Group of University Federico II of Naples, 'GRIP-UNINA'](http://www.grip.unina.it/).

All rights reserved. This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the terms of the license, as specified in the document `LICENSE.txt` (included in this directory).

# Prerequisites

This code is written for **Python2.7** and uses _Theano_ and _Lasagne_ libraries. The list of all requirements is in `requirements.txt`.

The command to install the requirements is:

```
cat requirements.txt | xargs -n 1 -L 1 pip2 install
```
Optional requirements for using gpu:

* cuda = 8
* cudnn = 5

# Usage

Firstly, you have to download the dataset from [LINK](http://www.grip.unina.it/).

The 8 proposed techniques to estimate the NDVI are:

```
SAR, SARp, OPTI, OPTII, SOPTI, SOPTII, SOPTIp, SOPTIIp. (1)
```
In the paper, these techniques correspond respectively to:

```
SAR, SAR+, Optical/C, Optical, Optical-SAR/C, Optical-SAR, Optical-SAR+/C, Optical-SAR+.
```

To train and/or test the CNN you have to use the `Train.py` and/or `Test.py`, in __TRAINING__ and __TEST__ directory respectively. 

In these files you have to set the technique_name that you can choose from (1): 

   ```
   kwargs['identifier'] = 'technique_name'
   ```

# Citing

If you use this CNN-based approach in your research or wish to refer to the baseline results, please use the following __BibTeX__ entry.
```
@article{scarpa2018cnn,
  title={A CNN-Based Fusion Method for Feature Extraction from Sentinel Data},
  author={Scarpa, Giuseppe and Gargiulo, Massimiliano and Mazza, Antonio and Gaetano, Raffaele},
  journal={Remote Sensing},
  volume={10},
  number={2},
  pages={236},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
