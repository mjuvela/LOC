## LOC

LOC is a line radiative transfer program written using  Python and OpenCL. 

For more details, see 
* http://www.interstellarmedium.org/radiative_transfer/loc/

The "Input files" and "Output files" sections below refer mostly to the 3D / octree version (`LOC_OT.py`); the spherically symmetric 1D version (`LOC1D.py`) is documented in the last section of this file.

---

## Input files

### Model cloud description

#### Cartesian 3D clouds

The 3D model is described by a separate binary file that defines (in the case of the Cartesian grid) the model dimensions and specifies for each cell seven quantities: the number density, the kinetic temperature, the amount of microturbulence (sigma), three components of the macroscopic velocity (vx, vy, and vz), and the fractional abundance. The file starts with the dimensions NX, NY, and NZ (three 4 byte integers), followed by the data for the NX×NY×NZ cells, each consisting of the aforementioned seven numbers (4 byte floats; the data for a single cell being consecutive elements in the file).

Of the three spatial coordinates, it is the X coordinate that should change fastest in the file. This corresponds to the "C-order" of elements. One must remember that in Python arrays the last index runs fastest. In other words, if one creates the cloud as a Python array before writing it to disk, that array must be indexed as `CLOUD[k, j, i, ifield]`, where k corresponds to the LOC Z axis and i to the LOC X axis.

The values in the file can be in physical units. For example, the density would be in units cm⁻³ and the microturbulence and velocities in km/s. Usually the density would be the density of H or of H₂, the abundance being the abundance of the examined species relative to that density. However, the raw values read from the file may be rescaled using appropriate keywords in the parameter file.

#### Octree clouds

The octree cloud file starts with five integer (int32) values: NX, NY, NZ, OTL, CELLS. Here (NX, NY, NZ) are the dimensions of the root grid, OTL is the number of hierarchy levels in the octree structure, and CELLS is the total number of cells in the model. The rest of the file contains a section for each of the parameter density, kinetic temperature, turbulent velocity dispersion, velocity in x-direction, velocity in y-direction, velocity in z-direction, and fractional abundance. Each of these sections starts with int32 number of the number of cells on root level, followed by parameter values on root level. The same is repeated for all hierarchy levels before moving onto the next parameter. The file structure is outlined below.

```
NX, NY, NZ, OTL, CELLS
cells_0, rho_0,   cells_1, rho_1, cells_2, ..., cells_N, rho_N,
cells_0, Tkin_0,  cells_1, Tkin_1, ...
cells_0, sigma_0, cells_1, sigma_1, ...
cells_0, vx_0,    cells_1, vx_1, ...
cells_0, vy_0,    cells_1, vy_1, ...
cells_0, vz_0,    cells_1, vz_1, ...
cells_0, abu_0,   cells_1, abu_1, ..., cells_N, abu_N
```

In the above `cells_*` are the number of cells on each hierarchy level (int32 scalars) and `rho_*`, `Tkin_*`, `vx_*`, `vy_*`, `vz_*`, `abu_*` are float32 vectors containing the values for the corresponding parameter for all cells on that hierarchy level.

The actual grid hierarchy is contained in the `rho_*` vectors. When a cell is refined (divided into eight subcells) the density value `rho_x[i]` is replaced by the binary float32 representation of `-(float *)&ind`. Here `ind` is the index of the first subcell in the vector `rho_(x+1)`, i.e. the density vector of the next hierarchy level. For the other quantities (Tkin, sigma, etc.) the values in the parent cells are usually undefined and are not used by LOC.

### Initialisation file

The following list describes the keywords used in the initialisation file (ini file). This is the file that is given as a command line parameter for the LOC programme. Each line in the parameter file starts with a keyword, possibly followed by one or more parameters separated by spaces. Comment lines or inline comments are indicated with a `#` character.

### Keywords (alphabetical order)

| Keyword | Type | Default | Description |
|---|---|---|---|
| `abundance` | float | — | Scaling of the abundance values read from the model file (synonym for `fraction`). |
| `alpha` | float | 1.0 | For 1D models only. Parameter affecting the radial discretisation: the default linear discretisation rᵢ, where i is the shell index. Radii will be placed at rᵢ^(1/(1+alpha)). Values alpha < 0 make the 1D grid denser towards the centre; values alpha > 0 make it denser towards the outer parts. Must be > -1. |
| `angle` | float | — | Apparent size of (root) grid cell in units of arcsec. For 1D models, this corresponds to the radius of the spherical model. |
| `bandwidth` | float | — | Total bandwidth used in the simulations and for the output spectra, in units of km/s. |
| `cabfile` | filename | — | Optional file giving the fractional abundance of each collisional partner (abundances relative to the density values used to define the model). In 3D models (LOC_OT.py), the file starts with NX, NY, NZ, and the number of collisional partners (four 4 byte integers). The rest lists the abundances for every model cell. In 1D models (LOC1D.py) the file content is just the CELLS×PARTNERS abundance values (float32). |
| `channels` | int | — | Number of velocity channels (in the simulation and the output spectra). |
| `clip` | float | 0.0 | Skip calculations when density is below this threshold. |
| `cloud` | filename | — | Name of the model file (with the density etc. values for every model cell). |
| `clsolve` | int | 1 | Solve equilibrium equations on device instead of on host. |
| `colden` | int | 0 | If > 0, instructs LOC to save information of the column density. In 1D models column densities are saved to a binary file; in 3D models results are saved to FITS files for the same pixels as for which the spectra are saved. |
| `coldensave` | int | 0 | Save total and molecular column density, mass-weighted LOS Tkin. |
| `cooling` | — | 0 | If given, the total cooling rate of the current species is calculated and saved to a file (one value per cell). |
| `coolfile` | filename | `'brute.cool'` | Filename for the cooling rate output. |
| `cores` | int | -1 | Ask LOC to only use the specified number of cores. Depends on the OpenCL implementation. |
| `crtemit` | filename | — | Specifies the name for a file with additional continuum emission in each cell. The binary file should start with two int32 numbers (number of cells and number of transitions), followed by the main array `emission[cells, transitions]` of float32 numbers (continuum emission in units of photons/s/Hz/H₂). |
| `crttau` | filename | — | Specifies the name for a file with additional continuum opacity in each cell. The binary file should start with two int32 numbers (number of cells and number of transitions), followed by the main array `tau[cells, transitions]` of float32 numbers (optical depth per pc). |
| `density` | float | — | Factor by which the density values read from the model file are multiplied before calculations (synonym for `kdensity`). |
| `device` | string | — | Select the OpenCL device for the calculations. The string can be `'c'` or `'g'` to select the first available CPU or GPU, respectively. Otherwise, the string can be any unique string that appears as part of the device name (use `clinfo` to list available devices). |
| `direction` | float, float, string | — | Specifies the direction of the observer as angles (in degrees) from the positive Z axis and rotation from positive X towards Y. The optional third parameter is an additional string included in the name of the output files. |
| `distance` | float | — | Distance between the model and the observer, in units of pc. |
| `dnlimit` | float | -1.0 | If the NI relative change exceeds this value, set new = 0.5×old + 0.5×new. |
| `damping` | float | -1.0 | With ALI, dampen iterations: `damping×old + (1-damping)×new`. |
| `doublecool` | int | 0 | Use double precision for cooling calculations. |
| `escape` | int | 0 | Flag 0/1: whether to estimate and save escape probabilities. |
| `fits` | int, float, float | — | Makes LOC_OT.py write results into FITS files instead of custom binary files. This includes files for the spectra, the optical depths, and the column densities. The optional parameters are the centre coordinates in degrees (RA, DEC) used in the written FITS files. If there is only one argument, it is interpreted as a flag (1 = use FITS, 0 = do not). |
| `FITS_DE` | float | 0.0 | FITS map centre DEC (degrees). |
| `FITS_RA` | float | 0.0 | FITS map centre RA (degrees). |
| `fraction` | float | — | Scaling applied to the abundance values read from the model file (synonym for `abundance`). |
| `fwhm` | float, ... | — | Beam FWHM [arcsec] for each spectrum (LOC_OT only), in the same order as `spectra` arguments. |
| `gpu` | int | 0 | For any value > 0, LOC tries to select a GPU instead of a CPU device. See also keyword `platform`. |
| `grid` | float | — | Specifies the step between spectra in the output maps (~pixel size). The unit is arcsec. |
| `hfsfile` | filename | — | Gives the name of the file that specifies the relative velocities and intensities of hyperfine components, when these are assumed to be in LTE. |
| `infallindex` | float | -1.0 | Save map of infall index (density-threshold-based collapse indicator). |
| `isotropic` | float | — | Specifies the temperature (in Kelvin) for isotropic background radiation, assumed to have the spectrum of a blackbody (synonym for `Tbg`). |
| `idevice` | int | 0 | Selected device within the platform (for given device type, running index over all devices of the requested type). |
| `iterations` | int | — | Number of iterations (consisting of the simulation of the radiation field and the updating of the level populations). |
| `kabundance` | float | 1.0 | Scaling applied to the abundance values read from the model file. |
| `kdensity` | float | 1.0 | Factor by which the density values read from the model file are multiplied before calculations (synonym for `density`). |
| `ktemperature` | float | 1.0 | Scalar scaling applied to the kinetic temperature values read from the model description file (synonym for `temperature`). |
| `kvelocity` | float | 1.0 | Scaling applied to the velocity values (three components of macroscopic velocity) that are read from the model description file (synonym for `velocity`). |
| `ksigma` | float | 1.0 | Scaling to be applied to the microturbulence value read from the file for the model description (synonym for `sigma`). |
| `KILL_EMISSION` | int | 999999 | Write spectra ignoring emission from cells >= KILL_EMISSION (1D models only). |
| `load` | filename | — | At the beginning of the run, load previously saved level populations from the given binary file. |
| `local` | int | -1 | Will change the default local work group size with the size given as the argument (see OpenCL documentation; defaults are probably 1 for CPU and 32 for GPU). |
| `levels` | int | — | Specifies the number of lowest energy levels included in the computations (less or equal to the number of levels for which the molecule file contains data). |
| `losspectrum` | int | 0 | Contribution of LOS steps to one spectrum: 1 = escaped, 2 = emitted (no foreground absorptions). |
| `lowmem` | int | 1 | Turns on some optimisations to minimise the memory usage, at the expense of some increase in the run time. |
| `mapview` | float, float, int, int, [float, float, float] | — | In LOC_OT.py an alternative way to specify the mapping parameters. Theta is the angle from the positive z-axis, and phi is the rotation around the z axis from the positive x axis, both in degrees. Parameters nx and ny are the number of pixels in the output map. The optional parameters cx, cy, and cz give a position in cell units for the centre of the map. |
| `MAP_INTERPOLATION` | int | -1 | Spatial interpolation in map making (0 = off, 1 = 3-point, 2 = weighted 3-point, 3 = weighted 4-point). |
| `maxbuf` | int | 40 | Maximum allocation of rays per root-grid ray (for OCTREE ≥ 4). |
| `max_sigma` | float | 0.0 | Maximum turbulent linewidth in the model. |
| `min_sigma` | float | 1e30 | Minimum turbulent linewidth in the model. |
| `minmaplevel` | int | -1 | Only hierarchy levels > minmaplevel used in map calculation. |
| `molecule` | filename | — | Specifies the name for the file containing the molecular data. The file can be in Lamda database format. |
| `multitran` | int | -1 | > 1 means multiple transitions calculated on a single kernel call. |
| `nray` | int | 64 | Applies only to 1D models (LOC1D.py). Number of rays (number of impact parameters) used during the calculation of the radiation field. |
| `nside` | int | 1 | Specifies the angular grid for the simulated rays, which follows the Healpix scheme. The total number of rays per model surface element is 12 × nside². |
| `offsets` | int | 1 | Number of spatial ray offsets per surface element. |
| `octree` | filename | — | In LOC_OT.py similar to the `cloud` keyword and specifies the name of the cloud file. An optional integer attached to the keyword indicates the ray-tracing implementation (1–5, 40). One should probably not use this keyword; `cloud` is preferred. |
| `ONESHOT` | int | 0 | For OCTREE=40, single kernel call to cover all rays (given side). |
| `overlap` | filename | — | Instructs LOC to take into account the possible overlap line profiles of some transitions (not assumed to be in LTE). The argument is the name of a file that lists these transitions. |
| `platform` | int, int | — | Specifies the OpenCL platform to be used in the calculations. The optional second integer parameter selects a device within the platform (running index over all devices within the platform that are of the requested type, CPU or GPU). By default all available platforms are tried in order. |
| `points` | int, int | — | Specifies the dimensions of the output spectrum files (the number of spectra along horizontal and vertical directions). |
| `plweight` | int | 1 | Include path-length weighting for OCTREE=4 (affects OT4 only). |
| `prefix` | string | — | Prefix for the output files. |
| `save` | filename | — | Gives the name for a binary file where the level populations are saved at the end of a run. |
| `sigma` | float | — | Scaling to be applied to the microturbulence value read from the file for the model description (synonym for `ksigma`). |
| `spectra` | int, int, ... | — | Specifies the transitions for which the spectra are to be computed and saved. The arguments are pairs "upper lower", two integers that specify the upper and lower energy level of the transition (0-offset numbering). |
| `speray` | int | -1 | For 1D models, specifies that spectra will be written for the number of offsets given here as the argument, instead of the default number of (equidistant) rays given by the keyword `nray` (synonym for `nray_spe`). |
| `stop` | float | 1.0e-5 | Iterations are automatically stopped when the largest relative change in level populations falls below this limit. Only levels up to the level specified with the keyword `uppermost` are tested. |
| `tau` | — | — | (See `tausave`.) |
| `tausave` | int | 0 | For int > 0, LOC1D.py saves optical depths to a file with suffix ".tau"; for LOC_OT.py, optical depths can be saved to FITS files. |
| `temperature` | float | — | Scalar scaling applied to the kinetic temperature values read from the model description file (synonym for `ktemperature`). |
| `thermaldv` | int | 1 | If > 0, add thermal broadening; otherwise use file values as such. |
| `transitions` | int, int, ... | — | List of transitions (using running index of the transitions) for which the excitation temperatures are to be saved. |
| `uppermost` | int | 999 | Specifies the uppermost energy level that is checked for level population changes (see keyword `stop`). |
| `verbose` | int | 1 | Verbosity level. |
| `velocity` | float | — | Scaling applied to the velocity values (three components if macroscopic velocity) that are read from the model description file (synonym for `kvelocity`). |
| `WITH_ALI` | int | 1 | Whether to use Accelerated Lambda Iteration (can be 0 for OCTREE ≥ 2). |
| `WITH_HALF` | int | 0 | Whether CLOUD is stored in half precision (vx, vy, vz, sigma). |
| `WITH_CRT` | int | 0 | Enable dust continuum (CRT) files. |
| `WITH_HFS` | — | — | (Set automatically when `hfsfile` is specified.) |
| `with_crt` | int | 0 | Include CRT files (dust continuum absorption and emission). |

### Molecular parameters

LOC reads directly files in the format of the Lamda database.

If the abundances of the collisional partners are assumed to be constant over the model volume, these abundances (with default value 1.0) can be added to the molecule file as a floating point number after the number of temperatures (same line, after the number of temperatures). If the abundances change from cell to cell, one needs to use the `cabfile` keyword of the ini files.

Possible hyperfine structure of the lines can be handled assuming LTE conditions between the hyperfine components (instead of the general line overlap in frequency space; see the `overlap` keyword). In this case, the molecule file should specify only the main energy levels, not the individual hyperfine components. Calculations require the use of the keyword `hfsfile` and a corresponding text file that lists the relative velocities and intensities of the components.

On the other hand, if the molecule file already describes transitions for which the profiles may overlap, one can use the keyword `overlap`. Its argument is also a text file, which this time lists the transitions for which the frequency overlap should be taken into account.

### Output files

#### Spectrum files

The spectra are saved to binary files. Currently the file names are similar to `prefix.molecule.01-00.bin`, where `prefix` is given in the ini-file, `molecule` is replaced by the actual name of the molecule (species), and the number gives the transition (based on the 0-offset numbering of energy levels).

The file contains:
- Three 4-byte integers: number of pixels along the two map axes (NX, NY), and the number of velocity channels (NCHN)
- Two 4-byte floating point numbers: velocity of the first channel and the channel width (km/s)
- A cube with dimensions (NX, NY, NCHN+2), consisting of 4-byte floating point numbers. Each vector of NCHN+2 elements starts with the x- and y-offsets, followed by the channel values

#### Excitation temperature

Excitation temperatures are saved to binary files, named combining the prefix (as given in the ini-file), the name of the molecule and the transition, and with an ending of `.tex`. The file:
- Starts with four 4-byte integers: the dimensions of the model cloud (NX, NY, NZ), and the number of excitation levels included in the calculation (LEVELS)
- Ends with a (NX, NY, NZ) cube of excitation temperature values for the chosen transition, saved as 4-byte floating point numbers

#### Optical depths

In the 1D case, the optical depths (saved using the keyword `tausave 1`) is a binary file. The beginning of the file is: number-of-offsets (int32), number-of-channels (int32), velocity-of-first-channel (float32, km/s), channel-width (float32, km/s). These are followed by number-of-offsets times number-of-channels values (float32) giving the optical depths.

In the 3D case, optical depths are saved by default to a binary file with the suffix `.tau`, the file containing just the peak optical depth of each spectrum. If one is saving data to FITS files (using `fits 1` in the ini file), the resulting 3D FITS file will contain the optical depths for all spectra and all channels.

#### Column densities

In the 1D case, when spectra are written, the program also saves a file (suffix `.colden`) that lists, among other things, column densities for the radially requidistant set of rays. The contents of this binary file are: num (int32), offset[num] (float32), N[num] (float32, total column density), Nmol[num] (float32, column density of the simulated species), and tau[num] (float32, peak optical depth).

In the 3D case, column density maps (when `colden 1`) are saved to FITS files, where the filename includes the string `NH2`. LOC_OT.py saves at the same time also similar maps for the column density of the examined species (file `…N…fits`), the mass-weighted line-of-sight kinetic temperature (file `…Tkin…fits`), and the mass-weighted line-of-sight mean velocity (file `…VLOS…fits`).

#### Cooling rates

The cooling rates are saved to a plain binary file that consists of one 4-byte value per cell. The values are the cooling rates in cgs units.

#### Examples of run times for a 3D model

Below are some examples of run times on different computer systems. The model has a 121³ 3D Cartesian grid. One is calculating CO spectra, taking into account the ten first energy levels and running the simulations with 128 velocity channels. The OpenCL platforms include Intel, NVidia, and AMD GPUs as well as CPUs used with POCL or Intel's own OpenCL implementations. Run times are given in seconds for a complete run that includes five iterations and the final writing of the spectrum files.

| Run time (seconds) | Device |
|---|---|
| 853.87 | CPU: Intel i7-8550U 1.80GHz, 4 cores, POCL |
| 855.89 | CPU: Intel i7-8550U 1.80GHz, 4 cores, Intel driver |
| 279.62 | GPU: Intel UHD Graphics 620, Intel driver |
| 249.54 | GPU: Intel UHD Graphics 620, Intel NEO driver |
| 46.39 | GPU: AMD Radeon VII (external GPU enclosure) |
| 351.70 | CPU: Intel i7-8700K CPU @ 3.70GHz, 6 cores, POCL |
| 31.24 | GPU: NVidia GeForce GTX 1080 (Laptop 2) |
| 64.68 | CPU: AMD Ryzen Threadripper 3960x, 24 cores, POCL |
| 30.19 | GPU: NVidia GeForce GTX 1080 Ti (Desktop) |
| 14.89 | GPU: NVidia GeForce RTX 3090 (Desktop) |

GPUs are generally more efficient than CPUs but also show a factor of ten differences between the cards. A modern multi-core CPU system does get close.

### Full example of a LOC run

#### Cloud model

We discretise the model onto a regular Cartesian grid of 64³ cells. The Python script `make_cloud.py` writes the cloud file:

```python
from numpy import *
from numpy.random import randn

NX, NY, NZ = 64, 64, 64
C = ones((NZ, NY, NX, 7), float32)

k, j, i = indices((NZ, NY, NX), float32)
c = 0.5*(NX-1.0)
i, j, k = (i-c)/c, (j-c)/c, (k-c)/c
r = sqrt(i*i+j*j+k*k)

C[:,:,:,0] = exp(-3.0**r*r)
C[:,:,:,1] *= 15.0
C[:,:,:,2] *= 1.0
C[:,:,:,3] = randn(NZ,NY,NX)
C[:,:,:,4] = randn(NZ,NY,NX)
C[:,:,:,5] = randn(NZ,NY,NX)
C[:,:,:,6] = 1.0

fp = open('my.cloud', 'wb')
asarray([NX, NY, NZ], int32).tofile(fp)
C.tofile(fp)
fp.close()
```

#### Ini file and LOC run

```
cloud          my.cloud
distance       100.0
angle          3.0
molecule       co.dat
density        1.0e4
temperature    1.0
fraction       1.0e-4
velocity       0.5
sigma          0.5
isotropic      2.73
levels         10
uppermost      3
iterations     5
nside          2
direction      90 0.001
points         64 64
grid           3.0
spectra        1 0  2 1
transitions    1 0  2 1
bandwidth      6.0
channels       128
prefix         test
load           test.save
save           test.save
stop          -1.0
gpu            1
```

Run from the command line:

```
LOC_OT.py   my.ini
```

This produces files including `test_CO_01-00.spe`, `test_CO_02-01.spe`, `test_CO_01-00.tex`, `test_CO_02-01.tex`, and `test.save`.

#### Plotting spectra

```python
import os, sys
from matplotlib.pylab import *

def LOC_read_spectra_3D(filename):
    fp = open(filename, 'rb')
    NRA, NDE, NCHN = fromfile(fp, int32, 3)
    V0, DV = fromfile(fp, float32, 2)
    SPE = fromfile(fp, float32).reshape(NRA, NDE, 2+NCHN)
    OFF = SPE[:,:,0:2].copy()
    SPE = SPE[:,:,2:]
    fp.close()
    return V0+arange(NCHN)*DV, SPE

def LOC_read_Tex_3D(filename):
    fp = open(filename, 'rb')
    NX, NY, NZ, dummy = fromfile(fp, int32, 4)
    TEX = fromfile(fp, float32).reshape(NZ, NY, NX)
    fp.close()
    return TEX
```

---

## LOC1D.py — spherically symmetric 1D models

`LOC1D.py` is the version of LOC for spherically symmetric models. The cloud is described by a set of concentric shells, and the radiation field is sampled along rays that are defined by their impact parameter with respect to the centre. The physical size of the model is the outer radius

```
GL = angle[arcsec] × ARCSEC_TO_RADIAN × distance[pc] × PARSEC   [cm]
```

Only radial (spherically symmetric) velocity fields are supported — there are no transverse velocities and no 3D turbulence field.

In practice, the code uses three different ray sets:

- **NRAY rays** (keyword `nray`) with impact parameters distributed according to `alpha` are used during the iterations. Each ray is characterised by its impact parameter, and the path length of each ray in each shell is precalculated. The escaping and incoming angle-averaged intensities (escape probabilities and SIJ integrals) are accumulated ray by ray, the SIJ values being corrected by the `VOLUME/APL` path-length factor (`PATH_CORRECTION`), where `APL` is the total path length of all rays within a shell.
- **NRAY_SPE equidistant rays** (keyword `speray`, default = `nray`) are used when the spectra are written. The first spectrum corresponds to the central line of sight (impact parameter = 0) and the last one to the impact parameter 0.9999×GL. The set of spectra is thus a position–velocity cut through the centre of the sphere.
- if escape probabilities are requested (keyword `escape`), a third kernel integrates the photon escape along `12×nside²` Healpix directions.

The kernels used are `kernel_update_1d_py.c` (no HFS), `kernel_update_1d_hfs_py.c` (`hfsfile`) and `kernel_update_1d_ol_py.c` (`overlap`). The level populations are by default solved on the device (the `SolveCL` kernel that is part of the same source file), or in Python with `clsolve 0`.

### Model cloud description (1D)

The 1D cloud file (`cloud` keyword) is a plain binary file, with the following contents:

```
CELLS                       (int32)  number of shells
RADIUS[CELLS]               (float32) outer radius of each shell, in increasing order
{ rho, Tkin, sigma, abu, Vrad }[CELLS]   (5 × float32 per shell)
```

| Quantity | Unit | Comment |
|---|---|---|
| `rho` | cm⁻³ | number density of H / H₂ (as defined by the molecule file) |
| `Tkin` | K | kinetic temperature |
| `sigma` | km/s | **non-thermal** Doppler width (the thermal part is added by LOC, see `thermaldv`) |
| `abu` | — | fractional abundance of the simulated species relative to `rho` |
| `Vrad` | km/s | radial velocity, positive outwards |

The file size is thus `4 + 4×CELLS + 20×CELLS` bytes. The radii are either normalised (the last element = 1.0, the physical size being set by the `angle` and `distance` keywords) or absolute values in cm (the last element then gives GL). The shell radii are normalised internally by the value of the last element. LOC prints a warning if the cloud file gives a radius (in cm) that differs by more than 1% from the value following from `angle` × `distance`.

The values read from the file are scaled and clipped as follows:

| Quantity | Scaled by | Clipped to |
|---|---|---|
| `rho` | `density` (`kdensity`) | [1.0e-5, 1.0e20] cm⁻³ |
| `Tkin` | `temperature` (`ktemperature`) | [2.0, 2900.0] K |
| `sigma` | `sigma` (`ksigma`) — only the non-thermal part | — |
| `abu` | `fraction` / `abundance` (`kabundance`) | [1.0e-20, 2.0] |
| `Vrad` | `velocity` (`kvelocity`) | — |

If `thermaldv 1` (default), thermal broadening is added in quadrature to the `sigma` value read from the file,

```
sigma_total = sqrt( sigma_file² + 2.0e-10 × k × Tkin / (amu × μ) )   [km/s]
```

where μ is the molecular weight given in the molecule file.

**Note:** the innermost shell must be hit by at least one ray, i.e. the smallest impact parameter, approximately `(0.5/NRAY)^(1/(1+alpha))` (in units of GL), has to be smaller than the outer radius of the innermost shell (in units of GL). Otherwise LOC1D.py stops with the message "INNERMOST CELL IS NOT HIT BY ANY RAYS". Either make the innermost shell thick enough, increase `nray`, or place the rays closer to the centre with `alpha` < 0.

### Ini-file keywords for LOC1D.py

The general syntax of the ini-file is the same as described above (keyword, arguments, `#` comments). Some of the keywords in the general table at the beginning of this document are used in a slightly different way, or are not used at all by LOC1D.py.

#### Keywords that must be given

These keywords have no default values in `ReadIni()` — LOC1D.py aborts if one of them is missing.

| Keyword | Type | Description |
|---|---|---|
| `cloud` | filename | Name of the 1D model file (format above). |
| `molecule` | filename | Molecule file in Lamda format. If the file is not found in the current directory, it is searched for in the (hard-wired) molecule directory defined in `LOC_aux.py`. |
| `levels` | int | Number of energy levels included in the calculation. The number of radiative transitions among these levels is derived automatically. |
| `channels` | int | Number of velocity channels. With HFS this is redefined to be the largest number of channels needed by any of the transitions (see below). |
| `bandwidth` | float | Bandwidth in km/s. The channel width is `bandwidth/channels` and remains unchanged also in the HFS case. |
| `distance` | float | Distance in pc. Always needed, also when the model size is taken from the cloud file (`angle` ≤ 0). |
| `angle` | float | Apparent **radius** of the model in arcsec. Give a negative value to take the physical radius from the cloud file (radii in cm); the angle is then derived from `distance`. |

#### Keywords specific to 1D models

| Keyword | Type | Default | Description |
|---|---|---|---|
| `nray` | int | 64 | Number of rays (impact parameters) used in the calculation of the radiation field. |
| `speray` | int | = `nray` | Number of spectra (equidistant impact parameters) that are written. May be larger than `nray`; synonym of the internal parameter `nray_spe`. |
| `alpha` | float | 1.0 | Sampling of the impact parameters: `IP = u^(1/(1+alpha))`, ray weight `DIRWEI = 2/(1+alpha)×IP^(1-alpha)`, where u runs over [0,1]. With alpha = 1 all rays have the same weight. alpha < 0 places more rays towards the centre, alpha > 0 towards the outer parts. Must be > -1. (In the current implementation this keyword affects only the ray sampling — the shell radii are used exactly as they are given in the cloud file.) A value < -10 selects an experimental mode with an equal number of rays per shell (requires `nray` divisible by the number of shells); it is not better than the normal weighting. |
| `KILL_EMISSION` | int | 999999 | Shells with index ≥ KILL_EMISSION contribute no line (or dust) emission to the output spectra. Shell numbering starts at 0 (innermost shell). |
| `tausave` | int | 0 | If > 0, the optical depth of every channel of every ray is saved in a second pass to a file with the suffix `.tau`. |
| `escape` | int | 0 | If > 0, the escape probability and the radial distribution of the emission are calculated and saved to a text file with the suffix `.esc`. |
| `nside` | int | 1 | Healpix NSIDE parameter for the directions used in the escape probability calculation (12×nside² directions). No other use in 1D models. |
| `cooling` | [filename] | off | If the keyword is present, the total cooling rate of the simulated species is calculated. Optionally the file name follows the keyword, otherwise the default `brute.cool` is used. Not implemented in combination with HFS. |
| `doublecool` | int | 0 | Use double precision accumulation in the cooling rate calculation. |
| `clsolve` | int | 1 | Solve the equilibrium equations on the OpenCL device (default) or in Python (`clsolve 0`). |
| `constant_tkin` | flag | off | Assume the same Tkin in all shells and precalculate the collisional part of the equilibrium matrix. Only used by the Python solver (`clsolve 0`). |
| `WITH_ALI` | int | 1 | Use ALI (escape-probability based) radiative coupling coefficients in the on-device solver. With 0, the full Einstein A coefficients are used instead. No effect on the Python solver, which always uses the ALI formulation. |
| `lowmem` | int | 1 | Keep the per-ray photon number arrays in global instead of local memory (needed with larger channel counts, slightly slower). |
| `local` (`LOCAL`) | int | 1 (CPU), 32 (GPU) | Override the default local work group size. |
| `hfsfile` | filename | — | Enables the hyperfine structure calculation assuming LTE between the HFS components. The file format is described below. |
| `overlap` | filename | — | Alternative to `hfsfile`: individual transitions whose profiles may overlap in frequency are followed separately (not assumed to be in LTE). The file format is described below. |
| `cabfile` | filename | — | Cell-by-cell abundances of the collisional partners: `CELLS × PARTNERS` float32 values, no header, cell index running slow and partner index fast. Without this keyword the abundances from the molecule file (default 1.0) are used. |
| `crtdust` | flag | off | Include dust continuum absorption and emission. The file names are fixed in LOC1D.py: `crt.opacity` and `crt.emission` (the `crttau` and `crtemit` keywords are **not** used by LOC1D.py). |
| `thermaldv` | int | 1 | Add thermal broadening to the `sigma` value read from the cloud file. |
| `pickle` | flag | on | Append the pickled INI dictionary to the `.spe`, `.tau`, `.colden` and `.tex` files. On by default, and used e.g. by `ConvolveSpectra1D.py` to get the model `angle`. |
| `load` / `save` | filename | — | Load / save level populations. Note that the number of levels in the save file must be the same as in the current run. |
| `iterations`, `stop`, `uppermost` | int, float, int | 1, 1e-5, 999 | Same meaning as in the general table. The iterations are stopped early only from the third iteration onwards. |
| `spectra` / `transitions` (`tex`) | int pairs | — | `spectra` gives the "upper lower" level pairs for which spectra are written, `transitions` (or `tex`) those for which excitation temperatures are written. |
| `prefix` | string | `prefix` | Prefix of all output files. |
| `isotropic` | float | 2.72548 | Temperature of the blackbody background radiation. |
| `gpu`, `platform` | int | 0, all | Device selection. In LOC1D.py only these two are effective: the device is always of the requested type (CPU by default) and the index within the platform is the first available one. The `device`, `sdevice`, `idevice` and `cores` keywords are not used. |

Keywords from the general table that LOC1D.py does **not** use: `points`, `grid`, `direction`, `mapview`, `offsets`, `fits`/`FITS_RA`/`FITS_DE` (there is no FITS output in 1D), `colden`/`coldensave` (column densities are always saved with the spectra, see below), `infallindex`, `losspectrum`, `multitran`, `minmaplevel`, `MAP_INTERPOLATION`, `octree`, `maxbuf`, `plweight`, `ONESHOT`, `WITH_HALF`, `clip`, `damping`, `dnlimit`, `min_sigma`, `max_sigma`, `verbose`, `crttau`, `crtemit`, `fwhm` (spectral convolution is done afterwards with `ConvolveSpectra1D.py`).

### Other input files

#### HFS structure file (`hfsfile`)

When the hyperfine components are assumed to be in LTE, the molecule file should describe only the main energy levels (no individual HFS components). The relative velocities and intensities of the components are listed in the text file given with the `hfsfile` keyword. The format is

```
# comments start with # in the first character of the line
# one block per transition:  upper  lower  number_of_components
# followed by exactly number_of_components lines:
#          velocity_shift_[km/s]   relative_weight
1 0 3
 -6.0   0.50
  0.0   1.00
  6.0   0.25
```

- The velocity shifts are given relative to an arbitrary zero point (only the separation of the components matters), in km/s.
- The weights are relative, they are normalised internally.
- The component lines must follow the transition line directly — blank lines and comments are allowed only between the blocks.
- Transitions that are not listed in the file are given a single component at offset 0.0 with weight 1.0.
- The number of levels in the molecule file, and hence the numbering of the levels in this file, must be consistent with the `levels` keyword.

The channel width `bandwidth/channels` is kept unchanged, but for every transition the band is widened to cover the full spread of the components:

```
channels(transition) = (bandwidth + vmax - vmin) / channel_width
```

The common number of channels is the largest one among all transitions (LOC prints "HFS revised => CHANNELS …"), and the `.spe`/`.tau` files of the individual transitions contain their own, smaller number of channels. The Gaussian profiles are now stored per transition and per shell (`GAU[TRANSITIONS, CELLS, CHANNELS]`), which requires more memory than a normal run. With HFS, cooling rates are not calculated and no `.colden` file is written.

#### Overlapping line file (`overlap`)

Instead of assuming LTE between the HFS components, one can follow the individual transitions and take their frequency overlap into account:

```
# one block per band of overlapping lines:
#   number_of_transitions_in_this_band
# followed by that many lines with  upper  lower  (0-offset level numbering)
2
1 0
2 1
```

For each band, the spectrum is written to files with names like `prefix.band0.spe`, with a number of channels equal to `channels` plus the frequency spread of the band expressed in channels. Transitions that do not belong to any band are treated as solitary lines and written as usual.

#### Collisional partner abundances (`cabfile`)

Plain binary file of `CELLS × PARTNERS` float32 values (no header), the abundances being relative to the density values used to define the model. If the file cannot be read with this shape, LOC1D.py stops with an error message.

#### Dust continuum files (`crtdust`)

Two binary files, written by CRT, both starting with the number of cells and the number of transitions (two int32 values), followed by `CELLS × TRANSITIONS` float32 values:

- `crt.opacity` — continuum optical depth per pc (converted internally to optical depth per GL),
- `crt.emission` — continuum emission in photons s⁻¹ Hz⁻¹ per H₂ (converted internally to photons s⁻¹ channel⁻¹ cm³, and multiplied by the density of each shell).

The number of cells and the number of transitions in these files must match those of the current calculation.

### Output files

All binary output files are little-endian, single precision, and — unless `pickle` is switched off — end with the pickled INI dictionary appended after the numerical data. When reading such a file, read exactly the number of values given by the header and ignore the rest of the file.

File names follow the pattern `prefix.MOLECULE_UU-LL.ext`, where `UU-LL` are the (0-offset) upper and lower levels of the transition, e.g. `test.CO_01-00.spe`.

#### Spectra (`.spe`)

Written for each transition listed with the `spectra` keyword:

```
NSPE        (int32)  number of spectra = number of impact parameters (speray)
NCHN        (int32)  number of velocity channels (for HFS: this transition's number)
V0, DV      (float32) first channel velocity and channel width [km/s]
SPE[NSPE, NCHN]  (float32)  brightness temperature [K] (Rayleigh–Jeans)
```

The velocity scale is centred on zero: `V0 = -0.5×(NCHN-1)×DV`. The i:th spectrum corresponds to the impact parameter `0.9999×GL×i/(NSPE-1)`, the first one being the line of sight through the centre. The background radiation is included (the spectra are observed intensities, not source functions).

#### Optical depths (`.tau`, keyword `tausave 1`)

The same format as the spectra files, but the values are the cumulative optical depths of each channel of each ray (a second pass over the rays is made after the iterations). In the HFS case, the `.tau` file contains the sum of the HFS component optical depths.

#### Column densities (`.colden`)

Written automatically whenever spectra are written (except in the HFS case):

```
N           (int32)  number of rays (NSPE)
OFF[N]      (float32) impact parameters [cm]
NTOT[N]     (float32) total (H2) column density [cm-2]
NMOL[N]     (float32) column density of the simulated species [cm-2]
TAU[N]      (float32) peak optical depth of the spectrum
```

#### Excitation temperatures (`.tex`, keyword `transitions` / `tex`)

```
CELLS       (int32)  number of shells
TEX[CELLS]  (float32) excitation temperature [K] for the transition
```

Note the single-integer header (the general 3D format has four integers).

#### Escape probabilities (`.esc`, keyword `escape 1`)

A plain text file written for each transition for which spectra are saved:

```
# radius,  j/volume,  j/impact_parameter, beta=escape_probability
  0.0000e+00      3.1169e-01    0.0000e+00     5.8055e-02
  1.1219e+16      2.8731e-01    1.0567e-01     3.5976e-01
  ...
```

The columns are the impact parameter (cm), the relative emissivity per unit volume (proportional to the population of the upper level of the transition at that position), the relative emissivity per unit impact parameter, and the escape probability, the latter being the ratio of photons that escape the model to the photons emitted, summed over the frequency channels and over the `12×nside²` directions.

#### Cooling rates (default `brute.cool`, keyword `cooling`)

A plain binary file of `CELLS` float32 values, the total cooling rate of the simulated species in each shell in cgs units (erg s⁻¹ cm⁻³). Not calculated in the HFS case.

#### Level populations (`save` / `load`)

```
CELLS, LEVELS   (2 × int32)
NI[CELLS, LEVELS]  (float32)  level population densities [cm-3]
```

Saved every 11:th iteration and after the last iteration. A file with a different number of levels than the current run cannot be reloaded (the level populations are then reset to LTE).

### Full example of a LOC1D.py run

#### Cloud model

```python
from numpy import *

CELLS = 100
R     = linspace(0.15, 1.0, CELLS).astype(float32)   # shell outer radii, normalised

C = zeros((CELLS, 5), float32)
C[:,0] = 1.0e4*(1.0 - R)**0.5     # density [cm-3]
C[:,1] = 15.0                     # Tkin [K]
C[:,2] = 0.2                      # non-thermal Doppler width [km/s]
C[:,3] = 1.0e-4                   # abundance
C[:,4] = 1.0                      # radial velocity [km/s]

fp = open('my.cloud1d', 'wb')
asarray([CELLS], int32).tofile(fp)
asarray(R, float32).tofile(fp)
C.tofile(fp)
fp.close()
```

#### Ini file and LOC run

```
molecule       co.dat
cloud          my.cloud1d
levels         5
uppermost      2
distance       100.0
angle          30.0
bandwidth      4.0
channels       64
nray           128
speray         64
iterations     5
stop           1.0e-5
spectra        1 0   2 1
transitions    1 0
tausave        1
prefix         test
load           test.save
save           test.save
```

Run from the command line:

```
LOC1D.py   test.ini
```

This produces files including `test.CO_01-00.spe`, `test.CO_02-01.spe`, `test.CO_01-00.tau`, `test.CO_02-01.tau`, `test.CO_01-00.colden`, `test.CO_01-00.tex`, and `test.save`.

To switch to the hyperfine structure calculation, add the line `hfsfile  co.hfs` to the ini file, `co.hfs` being a file in the format described above.

#### Reading 1D output

```python
from numpy import *

def LOC_read_spectra_1D(filename):
    fp = open(filename, 'rb')
    NSPE, NCHN = fromfile(fp, int32, 2)
    V0, DV     = fromfile(fp, float32, 2)
    SPE        = fromfile(fp, float32, NSPE*NCHN).reshape(NSPE, NCHN)
    fp.close()          # pickled INI at the end of the file is left unread
    return V0 + arange(NCHN)*DV, SPE, SPE[:, NCHN//2]   # channels, spectra, peak profile

def LOC_read_colden_1D(filename):
    fp = open(filename, 'rb')
    N      = fromfile(fp, int32, 1)[0]
    OFF    = fromfile(fp, float32, N)     # impact parameters [cm]
    NTOT   = fromfile(fp, float32, N)     # [cm-2]
    NMOL   = fromfile(fp, float32, N)     # [cm-2]
    TAU    = fromfile(fp, float32, N)     # peak optical depth
    fp.close()
    return OFF, NTOT, NMOL, TAU
```

The simulated spectra can afterwards be convolved with a Gaussian beam:

```
ConvolveSpectra1D.py   test.CO_01-00.spe   20.0
```

which writes `test.CO_01-00_convolved.spe`. The model `angle` is taken from the pickled information in the spectrum file, or can be given with the `-angle` option.
