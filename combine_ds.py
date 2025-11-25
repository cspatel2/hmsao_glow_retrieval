#%%
from os import path
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter
from typing import Any, Dict, List, Optional, SupportsFloat as Numeric, Tuple
from typing import Iterable
from matplotlib import pyplot as plt
# %%
####### INPUTS ##############################################
datadir = Path('/home/charmi/locsststor/proc/hmsao/l2')
fns = list(datadir.glob('*/*.nc'))
savedir = Path('/home/charmi/locsststor/proc/hmsao/l2c')
#########################################################################
# %%
def get_dates_from_fn(fn:str|Path) -> str:
    if isinstance(fn, str):fn = Path(fn)
    return str(fn.stem.split('_')[-2])

def combine_ds_by_datavars(dslist: Iterable[xr.Dataset]) -> xr.Dataset:
    """Combine datasets by taking only unique data variables from each dataset.

    Args:
        dslist (Iterable[xr.Dataset]):  List of xarray Datasets to combine.

    Returns:
        xr.Dataset:  Combined xarray Dataset with unique data variables.
    """    
    
    # build variable occurrence counter
    counter = Counter()

    for ds in dslist:
        counter.update(ds.data_vars.keys())    
    # choose only variables appearing exactly once
    unique_vars = {v for v, c in counter.items() if c == 1}
    # Sextract only unique vars from each dataset
    datasets_unique = [
        ds[[v for v in ds.data_vars if v in unique_vars]]
        for ds in dslist
    ]
    # merge them
    combined = xr.merge(datasets_unique)
    return combined

#create dataset with the given time range
def rechunk_datasets_in_daterange(all_fns: list[Path], start_timeofday:Tuple[Numeric] = (12,0,0) , duration: Tuple[float] = (0,24,0,0)): # type: ignore
    """Combine datasets in given time ranges by rechunking them.

    Args:
        all_fns (list[Path]): List of file paths to xarray Datasets.
        start_timeofday (Tuple[float], optional): Start time of day as (hours, minutes, seconds). Defaults to (12.,0.,0.).
        duration (Tuple[float], optional): Duration as (days, hours, minutes, seconds). Defaults to (0.,24.,0.,0.).
    """
    savedir = fns[0].parent
    if 'l2c' not in str(savedir):
        savedir = savedir.parent.joinpath('l2c')
    savedir.mkdir(parents=True, exist_ok=True)
    fns.sort()

    nds= xr.open_dataset(fns[0])
    tstamp_attrs = nds.tstamp.attrs
    del nds

    file_ranges = []
    for fn in fns:
        ds = xr.open_dataset(fn)
        t0 = datetime.fromtimestamp(ds.tstamp.values[0], tz=timezone.utc)
        t1 = datetime.fromtimestamp(ds.tstamp.values[-1], tz=timezone.utc)
        file_ranges.append((fn, t0, t1))
        del ds

    global_start = np.min([fr[1] for fr in file_ranges]) #very start time of all files
    global_end = np.max([fr[2] for fr in file_ranges]) #very end time of all files

    start = global_start.replace(hour=start_timeofday[0], minute=start_timeofday[1], second=start_timeofday[2]) #type: ignore
    duration_td = timedelta(days=duration[0], hours =duration[1], minutes=duration[2], seconds=duration[3]) # type: ignore
    start -= duration_td  #ensure saving the first chuck that may start before global_start

    while start < global_end:
        end = start + duration_td
        # select files that overlap with this time range
        selected_fns = [fr[0] for fr in file_ranges if not (fr[2] < start or fr[1] > end)]
        if len(selected_fns) > 0:
            print(f'Time range: {start} to {end}, selected {len(selected_fns)} files.')
            # Load datasets
            datasets = [xr.open_dataset(f) for f in selected_fns]
            datasets = [ds.sel(tstamp=slice(start.timestamp(), end.timestamp())) for ds in datasets]
            combined = xr.concat(datasets, dim='tstamp')
            combined.tstamp.attrs = tstamp_attrs
            combined.attrs.update({'ROI' : 'all',
                                'FileCreationDate':datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT") })
            savefn = savedir.joinpath(f'hmsao-l2c_{start.strftime("%Y%m%dT%H%M%S")}_{end.strftime("%Y%m%dT%H%M%S")}.nc')
            combined.to_netcdf(savefn)
            # print(f'Saved combined dataset for {start} to {end} to {savedir.joinpath(f"hmsao-l2c_{start.strftime("%Y%m%dT%H%M%S")}_{end.strftime("%Y%m%dT%H%M%S")}.nc")}')
        else:
            print(f'Time range: {start} to {end}, no files selected.')
        start = end



# %%
savedir.mkdir(parents=True, exist_ok=True)
dates = sorted(np.unique([get_dates_from_fn(fn) for fn in fns]))
# %%
#combine all datasets with the same date into one dataset with all variables
for date in dates:
    fns = sorted(list(datadir.glob(f'*/*{date}*.nc')))
    # Load datasets
    datasets = [xr.open_dataset(f) for f in fns]
    combined = combine_ds_by_datavars(datasets)
    combined.attrs.update({'ROI' : 'all',
                        'FileCreationDate':datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT") })
    combined.to_netcdf(savedir.joinpath(f'{date}.nc'))
    print(f'Saved combined dataset for {date} to {savedir.joinpath(f"{date}.nc")}')

#rechunch dataset into noon - noon files
datadir = savedir
fns = list(datadir.glob('*.nc'))
rechunk_datasets_in_daterange(fns)
for fn in fns:
    fn.unlink()

# %%
