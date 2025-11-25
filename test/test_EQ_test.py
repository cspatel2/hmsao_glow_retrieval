# %%
import xarray as xr
import numpy as np
from glowpython import GlowModel, maxwellian
from datetime import datetime, timedelta
import geomagdata
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker
from test_glow import get_brightness_from_ver, get_brightness_from_dat, get_geomag_for_glow
from typing import Iterable
from glowpython.utils import interpolate_nan
from glowpython import plots
from tqdm import tqdm
from glob import glob
# %%
longitude = 20.41
latitude = 67.84
elevation = 420  # Approximate elevation
testdate = datetime(2025, 1, 22, 17, 42, 22, 255821)
gmdict = get_geomag_for_glow(testdate)
# %%
def maxwellian_enegry_flux(E: float, Eo: float, Qo: float):  # enegry distribution
    return (Qo/(2*np.pi*Eo**3)) * E * np.exp(-E/Eo)

def chi_squared(obs: float, mod: float) -> float:
    return np.nansum((obs - mod)**2/obs**2)

# def chi_squared(obs: float, std: float, mod: float) -> float:
#     return np.nansum((obs - mod)**2/std**2)

# def chi_squared(obs:Iterable, mod:Iterable) -> float:
#     std = np.std(obs)
#     return np.nansum((obs - mod)**2/std**2)


# %%
# TODO: get observed data
wl = '5577'
gdat = xr.open_dataset(glob(f'*{wl}.nc')[0])
obs = gdat[f'{wl}'].values
std = gdat[f'{wl}_err'].values
# %%
plot = False
# TODO: create a map of E and Q
# characteritic enegry Eo (eV)
Eo = np.logspace(2, 4, 20)
if plot:
    plt.figure()
    plt.plot(Eo)
    plt.title('Characteristic Energy (eV)')
    plt.yscale('log')
    plt.ylabel('Eo (eV)')
    plt.xlabel('index')

# total eengry flux (ergs/cm2/s)
Qo = np.logspace(-1, 2, 40)
if plot:
    plt.figure()
    plt.plot(Qo)
    plt.title('Total Energy Flux (ergs/cm2/s)')
    plt.yscale('log')
    plt.ylabel('Qo (ergs/cm2/s)')
    plt.xlabel('index')

# meshgrid of Qo(x) and Eo(y)
qgrid, egrid = np.meshgrid(Qo, Eo)
if plot:
    plt.figure()
    plt.imshow(qgrid, norm=colors.LogNorm())
    plt.colorbar()
    plt.xlabel('Qo index')
    plt.ylabel('Eo index')
    plt.title('Qo meshgrid')

    plt.figure()
    plt.imshow(egrid, norm=colors.LogNorm())
    plt.colorbar()
    plt.xlabel('Qo index')
    plt.ylabel('Eo index')
    plt.title('Eo meshgrid')


# %%
# TODO: for each E&Q, run the model to model to get brightness
def get_brightness_from_QE(E: float, Q: float, wl:str, plot:bool=False) -> float:
    maxds = maxwellian(time=testdate,
                    glat=latitude,
                    glon=longitude,
                    Nbins=300,
                    Q=Q,
                    Echar=E,
                    geomag_params=gmdict,
                    )
    subds = maxds.sel(wavelength=wl)
    if plot:
        rhmax = subds.ver.idxmax('alt_km').values
        print(f'Red Model Brightness: {get_brightness_from_ver(subds):.02f} R')
        print(f'Red Peak Altitude: {subds.ver.idxmax('alt_km').values:.2f} km')
        subds.ver.plot(y='alt_km', color='red')
        plt.axhline(y=rhmax, color='red', ls='--', lw=.5)
        plt.text(-.2, rhmax, f'{rhmax:.01f} km', color='red')
    return get_brightness_from_ver(subds,elevation)

#%%
grid= []
for i in tqdm(range(egrid.shape[0])):
    row = list(map(lambda E,Q:get_brightness_from_QE(E,Q,wl),egrid[i,:],qgrid[i,:]))
    grid.append(row)
    del row
#%%
modelgrid = xr.Dataset(
    data_vars = dict(
        model = (('Eo','Qo'),grid)
    ),
    coords=dict(
        Qo = (('Qo'),Qo),
        Eo = (('Eo'),Eo/1000)

    )
)

modelgrid.model.plot(xscale = 'log', yscale ='log',norm=colors.LogNorm())
# %%

# TODO: for each E&Q find the chi_squared value (this takes in model brightness and measured brightness)
measured_brightness = get_brightness_from_dat(gdat,testdate,0)
chigrid = []
for i in tqdm(range(egrid.shape[0])):
    row = list(map(lambda obs:chi_squared(obs,measured_brightness), modelgrid.model.values[i,:]))
    chigrid.append(row)


# TODO: that E&Q pair with the lowest chi_squared is the best fit
# %%
cgrid = xr.Dataset(
    data_vars = dict(
        chi = (('Eo','Qo'),chigrid)
    ),
    coords=dict(
        Qo = (('Qo'),Qo),
        Eo = (('Eo'),Eo/1000)

    )
)

# %%
cgrid.chi.plot(xscale = 'log', yscale ='log',norm=colors.LogNorm())
# %%
