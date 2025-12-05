#%%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import xarray as xr
# %%
datadir = Path('../data/POES_data')
fns = list(datadir.glob('*.nc'))
fns.sort()
# %%
dates =np.sort(np.unique( [fn.stem.split('_')[-2] for fn in fns]))
# %%
ds = xr.open_dataset(fns[0])
# %%
ds
# %%
