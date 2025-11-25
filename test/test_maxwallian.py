# %%

#%%
import numpy as np
import xarray as xr
from glowpython import maxwellian
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob
from test_glow import get_geomag_for_glow
#%%

# %%
testdate = datetime(2025,1,27,19,46)
gmdict = get_geomag_for_glow(testdate)
sweden = {'lat': 67.84, 'lon': 20.41, 'elev': 420}  # Approximate location of Kiruna
wls = ['5577','6300']

iono = maxwellian(testdate,sweden['lat'],sweden['lon'],
                  Nbins= 250,
                  Q = 1,
                  Echar=100,
                  geomag_params=gmdict)

# %%
iono.sel(wavelength = wls[0]).ver.plot(y='alt_km', color='green')
iono.sel(wavelength = wls[1]).ver.plot(y='alt_km', color='red')

# %%
