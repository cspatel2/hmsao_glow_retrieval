# %%
import xarray as xr
import numpy as np
from glowpython import GlowModel, maxwellian,plots
from glowpython.utils import interpolate_nan
from datetime import datetime, timedelta
import geomagdata
import matplotlib.pyplot as plt
from typing import Dict
from glob import glob
import os
# %%

def get_brightness_from_ver(ds:xr.Dataset, alt_min:float=None, alt_max:float = None) -> float:
    """Performs LOS intergration of VER to calcuate brightness. 

    Args:
        ds (xr.Dataset): glow Dataset with of selected wavelength
        alt_min (float, optional): lower limit of intergration. Defaults to None.
        alt_max (float, optional): upper limit of intergration. Defaults to None.

    Returns:
        float: brightness (R)
    """    
    ds = ds.sel(alt_km = slice(alt_min, alt_max))
    ds = ds.dropna('alt_km')
    altcm = ds.alt_km.values * 10e5  # km -> cm
    ver = ds.ver.values # VER(z) in photons cm-3 s-1
    
    return 1e-6 * np.trapezoid(altcm,ver) # R

# def get_brightness_from_ver(ds:xr.Dataset, alt_min:float=None, alt_max:float = None) -> float:
#     """Performs LOS intergration of VER to calcuate brightness. 

#     Args:
#         ds (xr.Dataset): glow Dataset with of selected wavelength
#         alt_min (float, optional): lower limit of intergration. Defaults to None.
#         alt_max (float, optional): upper limit of intergration. Defaults to None.

#     Returns:
#         float: brightness (R)
#     """    
#     ds = ds.sel(alt_km = slice(alt_min, alt_max))
#     altcm = ds.alt_km.values * 10e5  # km -> cm
#     dz = np.diff(altcm)
#     dz = np.append(dz, dz[-1])
#     ver = ds.ver.values # VER(z) in photons cm-3 s-1
#     return 1e-6 * np.nansum(ver*dz) # R

def get_brightness_from_dat(ds:xr.Dataset,wl:str,dt:datetime, za:float) -> float:
    ds = ds.sel(tstamp = dt.timestamp(), method='nearest').sel(za=za, method='nearest')
    return ds[wl].values, ds[f'{wl}_err'].values

def get_geomag_for_glow(dt:datetime)->Dict:
    previousdate = dt - timedelta(days=1)
    geomag = geomagdata.get_indices([previousdate, dt], 81)
    gmdict = {'f107a': geomag.iloc[1]['f107s'],  # 81 day average
            'f107': geomag.iloc[1]['f107'],  # today
            'f107p': geomag.iloc[0]['f107'],  # previous day
            'Ap': geomag.iloc[1]['Ap'],
            }
    return gmdict


#%%
if __name__ == '__main__':
    # location, time, geomagnetic parameters
    # Kiruna, Sweden coordinates
    longitude = 20.41
    latitude = 67.84
    elevation = 420  # Approximate elevation
    testdate = datetime(2025, 1, 22, 19, 46)
    gmdict = get_geomag_for_glow(testdate)

    #set up glow model
    glow = GlowModel()
    glow.setup(testdate, latitude, longitude, geomag_params=gmdict)
    ds = glow()  # Evaluate the GLOW model and get the dataset

    #measured data
    w = 6300
    fdir = '/home/charmi/locsststor/proc/hmsao_1b/202501'
    fname = glob(os.path.join(fdir,f'*{w}.nc'))[0]
    rdat = xr.open_dataset(fname)

    w = 5577
    fname = glob(os.path.join(fdir,f'*{w}.nc'))[0]
    gdat = xr.open_dataset(fname)

    #modeled data
    gds = ds.sel(wavelength='5577')
    ghmax = gds.ver.idxmax('alt_km').values
    rds = ds.sel(wavelength='6300')
    rhmax = rds.ver.idxmax('alt_km').values

    plt.figure()
    
    rds.ver.plot(y='alt_km', color='red')
    print(f'Red Model Brightness: {get_brightness_from_ver(rds):.02f} R')
    print(f'Red Measured Brightness: {get_brightness_from_dat(rdat,'6300',testdate,0)[0]:.02f} R')
    print(f'Red Peak Altitude: {rds.ver.idxmax('alt_km').values:.2f} km')
    plt.axhline(y = rhmax, color ='red', ls = '--', lw = .5)
    plt.text(-.2,rhmax, f'{rhmax:.01f} km', color = 'red')

    gds.ver.plot(y='alt_km', color='green')
    print(f'Green Model Brightness: {get_brightness_from_ver(gds):.02f} R')
    print(f'Green Measured Brightness: {get_brightness_from_dat(gdat,'5577',testdate,0)[0]:.02f} R')
    print(f'Green Peak Altitude: {gds.ver.idxmax('alt_km').values:.2f} km')
    plt.axhline(y = ghmax, color ='green', ls = '--', lw = .5)
    plt.text(-.2,ghmax, f'{ghmax:.01f} km', color = 'green')


    plt.title(f'Glow Model\n {testdate.strftime("%Y-%m-%d %H:%M:%S")} ')
    yticks = plt.yticks()
    plt.show()

#%%%
# maxds = maxwellian(testdate,latitude,longitude,
#             10,
#             Q=1,
#             Echar=100,
#             geomag_params=gmdict,
#             )
#     #
# rmds = maxds.sel(wavelength = '6300')
# print(f'Red Model Brightness: {get_brightness_from_ver(rmds):.02f} R')
# ne = interpolate_nan(maxds["NeIn"].values, inplace=False)

# print(f'TEC: {np.trapezoid(ne, maxds.alt_km.values*1e5)*1e-12:.2f} TECU, hmf2: {maxds.attrs["hmf2"]:.1f} km')
# # %%
# plots.precip(maxds.precip)
# # %%
# plots.ver(maxds)
# # %%
# plots.density(maxds)
# # %%
# plots.temperature(maxds)
#%%
