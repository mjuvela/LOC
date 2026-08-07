## LOC

LOC is a line radiative transfer program written using  Python and OpenCL. 

For more details, see 
* http://www.interstellarmedium.org/radiative_transfer/loc/

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
