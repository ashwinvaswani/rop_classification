# rop_classification

This repository contains a docker image for rop classification (No rop, pre-plus and plus) from vessel segmented images.

### Pulling image from command line:
```
$ docker pull docker.pkg.github.com/qtim-lab/rop_classification/rop:v0
```

### Using as base image in dockerfile:
```
FROM docker.pkg.github.com/qtim-lab/rop_classification/rop:v0
```

### How to use the docker image:
Loading into the container will automatically set the working directory to src/ which contains all the code. 

However, the dataset (segmented vessels) must be mounted while running the container. 

Below is an example:
```
nvidia-docker run --rm -it -v {Absolute path to segmented images in local system}:/segmented rop_classification:v0
```

Authors:
1. Ashwin Vaswani
2. Katharina Hoebel
3. Praveer Singh
