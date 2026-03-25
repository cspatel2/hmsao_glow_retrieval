#%%
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import xarray as xr
import lzma
import pickle
from datetime import datetime, timezone
# %%
datadir = Path('model_neutral_qe')
subdirs = sorted([d for d in datadir.iterdir() if d.is_dir()])
# %%
fns = list(subdirs[2].glob('*.'))
# %%
fn = fns[3]

# %%
time = []
q = []
e = []
with lzma.open(fn, 'rb') as f:
    fitres = pickle.load(f)
    for vals in fitres:
        tstamp, pert = vals
        time.append(datetime.fromtimestamp(tstamp, tz=timezone.utc))
        q.append(pert.x[-2])
        e.append(pert.x[-1])
# %%
fig,ax = plt.subplots(figsize=(10,6))
color = 'black'
ax.plot(time, q, label='q', color=color)
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Q [ergs/s.cm2]', color=color)
cax = ax.twinx()
color = 'blue'
cax.plot(time, e, label='e', color=color)
cax.set_ylabel('Echar [eV] ', color=color)
cax.tick_params(axis='y', labelcolor=color)

# %%
fns = list(subdirs[1].glob('*vert*.nc'))
fns.sort()
fn = fns[2]

# %%
ds = xr.open_dataset(fn)
ds = ds.assign_coords({'time': ('tstamp', [datetime.fromtimestamp(t, tz=timezone.utc) for t in ds.tstamp.values])})
# %%
nifull  = ds['NeIn'].integrate('alt_km')*1e-7
nofull  = ds['NeOut'].integrate('alt_km')*1e-7
nin = ds['NeIn'].sel(alt_km = slice(None, 200)).integrate('alt_km')*1e-7
nout = ds['NeOut'].sel(alt_km = slice(None, 200)).integrate('alt_km')*1e-7
# %%
plt.figure(figsize=(10,6))
nofull.plot(x = 'time', label='NeOut Full')
nifull.plot(x = 'time', label='NeIn Full')
nin.plot(x = 'time', label='NeIn')
nout.plot(x = 'time' , label='NeOut')
plt.xlabel('Time (UTC)')
plt.ylabel('Ne [TECU]')

plt.legend()
# %%
ds
# %%
