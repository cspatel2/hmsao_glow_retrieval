#%%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from datetime import datetime, timezone, timedelta
from pathlib import Path
# %%
datadir = Path('/home/charmi/locsststor/proc/hmsao/l1c')
#%%

# %%
date = '20250320'
win = '6300'
fns = list(datadir.glob(f'*{date}*{win}*.nc'))
# %%
ds = xr.open_mfdataset(fns)
# %%
ds = ds.assign_coords(time = ('tstamp',[datetime.fromtimestamp(t, tz=timezone.utc) for t in ds.tstamp.values]))


# %%
plt.figure(figsize=(10,6))
ds.daybool.plot(x = 'time')
# %%
fig, ax = plt.subplots(figsize=(10,6))
ds['intensity'].sum('za').sel(wavelength = slice(629,631)).plot(x = 'time', ax =ax)
# cax = ax.twinx()
# ds.sza.plot(x = 'time', color='orange', ax =cax)
# %%
