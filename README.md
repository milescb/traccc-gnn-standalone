## Set environement variables

To compile with `traccc` we first need a shared library. We already have a version compiled at `nersc`. Simply set:

```bash
export DATADIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/data
export INSTALLDIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/software/dev/install
export PATH=$INSTALLDIR/bin:$PATH
```

Then build like normal:

```bash
mkdir build && cd build
cmake ../
make
```

The code can be run with the built executabe `./TracccGpuStandalone`. 

## Build traccc

If not at `nesrc`, configure `traccc` with the following command:

```
cmake -S traccc/ -B $BUILDDIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRACCC_BUILD_CUDA=ON \
    -DTRACCC_BUILD_EXAMPLES=ON \
    -DTRACCC_USE_ROOT=FALSE \
    -DCMAKE_INSTALL_PREFIX=$INSTALLDIR
```

Then build and install. Finally, source `$INSTALLDIR` and add to `$PATH` as above. 

## Getting ITk geometry files

To access the ITk geometry files, you must have a cern account and be able to access the `eos` area. First, sign up for the `atlas-tdaq-phase2-EFTracking-developers` e-group; then, the geometry files can be found at ` /eos/project/a/atlas-eftracking/GPU/ITk_data/ATLAS-P2-RUN4-03-00-00/`.