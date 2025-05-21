## Set environement variables

To compile with `traccc` we first need a shared library. We already have a version compiled at `nersc`. Simply set:

```bash
export DATADIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/data
export INSTALLDIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/software/prod/ver_03202024_traccc_v0.20.0/install
export PATH=$INSTALLDIR/bin:$PATH
```

Then build like normal. 