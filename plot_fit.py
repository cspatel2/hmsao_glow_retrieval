#%%
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import List, Optional, SupportsFloat as Numeric
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import scipy
from tqdm import tqdm
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pytz
from matplotlib import ticker
import matplotlib
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime, timedelta,timezone
from common_functions import get_date
from settings import Directories, ROOT_DIR
from plotting_functions import init

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
# %%
def compile_fit_intensity_stats(suffixes: List[str], base_suffix: Optional[str] = None, rootdir:str|Path = ROOT_DIR) -> dict[str, xr.Dataset]:
    if base_suffix is not None:
        msuffixes = [base_suffix] + suffixes
    else:
        msuffixes = [None] + suffixes
    
    if isinstance(rootdir, str):
        rootdir = Path(rootdir)
    dates =np.unique( list(map(get_date, (rootdir).glob('*/*.nc'))))
    print(f'Found {len(dates)} dates to process.')
    dates.sort()
    print(f'Dates: {dates}')
    stats = {}
    for date in tqdm(dates, dynamic_ncols=True): # type: ignore
        dslist = []
        for suffix in msuffixes:
            dirs = Directories(suffix=suffix, basedir=str(rootdir))
            with xr.open_dataset(dirs.model_dir / f'keofit_{date}.nc') as ds:
                for var in list(ds.data_vars):
                    if var not in ['6300', '5577']:
                        ds = ds.drop_vars(var)
                dslist.append(ds)


        dss = xr.concat(dslist, dim='suffix', compat='equals')
        for var in list(dss.data_vars):
            dss[var + '_mean'] = dss[var].mean(dim='suffix') 
            dss[var + '_std'] = dss[var].std(dim='suffix')
            dss[var + '_min'] = dss[var].min(dim='suffix')
            dss[var + '_max'] = dss[var].max(dim='suffix')
            dss[var + '_geomean'] = dss[var].std(dim='suffix')
            dss[var + '_geomean'].values = scipy.stats.mstats.gmean(dss[var], axis=0)

        stats[date] = dss
    return stats

# %%
def compile_counts_data(counts_dir: Path, dates: Optional[List[str]]) -> dict[str, xr.Dataset]:
    data = {}
    for date in tqdm(dates, dynamic_ncols=True): # type: ignore
        fns = list(counts_dir.glob(f'*{date}*.nc'))
        fns.sort()
        fn = fns[-1]
        nds = xr.load_dataset(fn)
        ds = nds.copy()
        daybool = [1 if z < 90+18 else 0 for z in ds.sza.values]
        ds = ds.assign_coords(daybool = ('tstamp', daybool))
        if 'daybool' in list(ds.coords): #pick out only night time data and drop everything else
            ds = ds.where(ds.daybool == 0, drop=True)
        # drop all the  negative za values
        ds = ds.where(ds.za > 0.5,   drop=True)
        ds = ds.assign_coords(za = np.abs(ds.za.values))
        tstamps = ds.tstamp.values
        start = dt.datetime.fromtimestamp(tstamps[0], tz= dt.timezone.utc)
        end = dt.datetime.fromtimestamp(tstamps[-1], tz= dt.timezone.utc)
        start += dt.timedelta(hours=1)
        end -= dt.timedelta(hours=1)
        ds = ds.sel(tstamp =slice(start.timestamp(), end.timestamp()))
        data[date] = ds
    return data
# %%
suffixes = init('nqe', rootdir='model_neutral_qe')
print(f'Found suffixes: {suffixes}')
compiled_stats = compile_fit_intensity_stats(suffixes[1:], base_suffix=suffixes[0], rootdir='model_neutral_qe')

dates = list(compiled_stats.keys())
counts_dir = Path('/home/charmi/locsststor/proc/hmsao/l2c')
complied_counts = compile_counts_data(counts_dir, dates)
# %%

# %%
def plot_fit_vs_observed_multidates(compiled_fits:  dict[str, xr.Dataset],compiled_counts:  dict[str, xr.Dataset], dates: List[str], za_idx:int, savefig: Optional[Path] = None):
    
    import matplotlib.units as munits
    converter = mdates.ConciseDateConverter()
    munits.registry[np.datetime64] = converter
    munits.registry[datetime.date] = converter
    munits.registry[datetime] = converter
    def get_hour_range_for_date(date, min_hour, max_hour, tz = timezone.utc):
        date = date.replace(tzinfo=tz)
        start_datetime = date + timedelta(hours=min_hour)
        # start_datetime = datetime.combine(date, datetime.min.time()) + timedelta(hours=min_hour)
        if max_hour <= 24:
            # end is next day
            end_datetime = date + timedelta(hours=max_hour)
        else:
            max_hour -= 24
            end_datetime = date.replace(day = date.day+1) 
            end_datetime += timedelta(hours=max_hour)
        return start_datetime, end_datetime

    dates.sort()

    lprops = {
        '5577': {'color': 'green', 'label': '5577 Fit', 'obs_color': 'lightgreen', 'obs_label': '5577 Observed'},
        '6300': {'color': 'red', 'label': '6300 Fit', 'obs_color': 'salmon', 'obs_label': '6300 Observed'},
    }

    for date in dates:
        fds = compiled_fits[date].copy() #fitted dataset
        ods = compiled_counts[date].copy() #observed dataset


        slit_width = 0.01 #,cm  ,#100UM
        foreoptic_fl= 2.5 #,cm  ,#25MM
        dza = np.mean([np.deg2rad(np.mean(np.diff(fds.za.data))),np.deg2rad(np.mean(np.diff(ods.za.data)))])
        aw = slit_width/foreoptic_fl * dza # cm^2 sr
        to_rayleighs = 1 / aw / (4*np.pi*1e6)  # photons/(s cm^2 sr) to Rayleighs

        tstamps = ods.tstamp.values
        time = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in tstamps]
        height = ods.za.values
        dheight = np.rad2deg(dza) 
        obs_5577 = ods['5577'].values.T[::-1, :] * to_rayleighs/1000 #kR
        stds_5577 = ods['5577_err'].values.T[::-1, :] * to_rayleighs/1000 #kR
        obs_6300 = ods['6300'].values.T[::-1, :] * to_rayleighs/1000 #kR
        stds_6300 = ods['6300_err'].values.T[::-1, :] * to_rayleighs/1000 #kR

        fit_5577 = fds['5577_mean'].values.T[::-1, :] * to_rayleighs/1000 
        fstd_5577 = fds['5577_std'].values.T[::-1, :] * to_rayleighs/1000   
        fit_6300 = fds['6300_mean'].values.T[::-1, :] * to_rayleighs/1000
        fstd_6300 = fds['6300_std'].values.T[::-1, :] * to_rayleighs/1000


        #%%
        fig,ax = plt.subplots(1,2, figsize=(12,6), sharex=True)
        axL = ax[0]
        axR = ax[1]

        obs_55 = axL.plot(time,obs_5577[za_idx,:], color
date = dates[0]
fds = compiled_stats[date].copy() #fitted dataset
ods = complied_counts[date].copy() #observed dataset


slit_width = 0.01 #,cm  ,#100UM
foreoptic_fl= 2.5 #,cm  ,#25MM
dza = np.mean([np.deg2rad(np.mean(np.diff(fds.za.data))),np.deg2rad(np.mean(np.diff(ods.za.data)))])
aw = slit_width/foreoptic_fl * dza # cm^2 sr
to_rayleighs = 1 / aw / (4*np.pi*1e6)  # photons/(s cm^2 sr) to Rayleighs

tstamps = ods.tstamp.values
time = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in tstamps]
# if (len(tstamps) == 0):
#     continue
height = ods.za.values
dheight = np.rad2deg(dza) 
obs_5577 = ods['5577'].values.T[::-1, :] * to_rayleighs/1000 #kR
stds_5577 = ods['5577_err'].values.T[::-1, :] * to_rayleighs/1000 #kR
obs_6300 = ods['6300'].values.T[::-1, :] * to_rayleighs/1000 #kR
stds_6300 = ods['6300_err'].values.T[::-1, :] * to_rayleighs/1000 #kR

fit_5577 = fds['5577_mean'].values.T[::-1, :] * to_rayleighs/1000 
fstd_5577 = fds['5577_std'].values.T[::-1, :] * to_rayleighs/1000   
fit_6300 = fds['6300_mean'].values.T[::-1, :] * to_rayleighs/1000
fstd_6300 = fds['6300_std'].values.T[::-1, :] * to_rayleighs/1000

#%%
za_idx = 20
fig,ax = plt.subplots(1,2, figsize=(12,6), sharex=True)
axL = ax[0]
axR = ax[1]

obs_55 = axL.plot(time,obs_5577[za_idx,:], color ='Green')
fill_55 = axL.fill_between(time, obs_5577[za_idx,:]-stds_5577[za_idx,:], obs_5577[za_idx,:]+stds_5577[za_idx,:], color='green', alpha=0.5)
fit_55 = axL.plot(time,fit_5577[za_idx,:],color = 'black')

obs_63 = axR.plot(time,obs_6300[za_idx,:], color ='Red')
fill_63 = axR.fill_between(time, obs_6300[za_idx,:]-stds_6300[za_idx,:], obs_6300[za_idx,:]+stds_6300[za_idx,:], color='red', alpha=0.5)
fit_63 = axR.plot(time,fit_6300[za_idx,:],color = 'black')
axR.legend([obs_63,fill_63 ], ['Combined Line Data'], loc='upper right')
plt.show()

# %%
