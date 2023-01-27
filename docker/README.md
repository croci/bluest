# BLUEST docker files

### Description

This folder contains the docker files for the **BLUEST** development environment that contains all **BLUEST** dependencies (including those for the examples), but not **BLUEST** itself.

### Using the image from DockerHub

There is a pre-built docker image containing all BLUEST dependencies (including those for the examples) built upon the [legacy FEniCS docker images](https://bitbucket.org/fenics-project/docker/src/master/). This image can be downloaded and run as follows:

```bash
> docker run -ti -v $(pwd):/home/fenics/shared croci/bluest:latest
```

If you prefer using singularity you can also convert and run the docker image by typing:

```bash
> singularity build bluest.img croci/bluest:latest
> singularity exec --cleanenv ./bluest.img /bin/bash -l
```

Once the container is running, install **BLUEST** by calling:

```bash
> pip install --user git+https://github.com/croci/bluest.git
```

### Building the image yourself

Required modifications can be incorporated into the [Dockerfile](/docker/Dockerfile). The modified image can then be built by calling:

```bash
> docker build -t IMAGE_NAME:IMAGE_TAG .
```

Here IMAGE_NAME and IMAGE_TAG can be replaced with whatever (e.g., modified_bluest:latest). The built image can be run via, e.g.,

```bash
> docker run -ti -v $(pwd):/home/fenics/shared IMAGE_NAME:IMAGE_TAG
```

### License

See the [LICENSE](/docker/LICENSE.md) file for license rights and limitations.
