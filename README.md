## Set environement variables

To compile with `traccc` we first need a shared library. We already have a version compiled at `nersc`. Simply set:

```bash
export DATADIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/data
export INSTALLDIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/software/dev/install
export PATH=$INSTALLDIR/bin:$PATH
```

Then build like normal. 

## Build traccc

```
cmake -S traccc/ -B $BUILDDIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRACCC_BUILD_CUDA=ON \
    -DTRACCC_BUILD_EXAMPLES=ON \
    -DTRACCC_USE_ROOT=FALSE \
    -DCMAKE_INSTALL_PREFIX=$INSTALLDIR
```

Then build and install. Finally, source `$INSTALLDIR` and add to `$PATH`. 