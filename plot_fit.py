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
from common_functions import get_date, LINESTYLE_DICT
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
def plot_fit_vs_observed_multidates_colN(compiled_fits:  dict[str, xr.Dataset],compiled_counts:  dict[str, xr.Dataset], dates: List[str], za_idx:int, savefig: Optional[Path] = None) -> None:
    """ plot fitted and observed data to show the goodness-of-fit for multiple dates in a column format.
    Each date get a row with N (species) columns
    here N=2 (6300A and 5577A)
    Args:
        compiled_fits (dict[str, xr.Dataset]):  dict of fitted datasets per date
        compiled_counts (dict[str, xr.Dataset]): dict of observed datasets per date
        dates (List[str]): list of dates to plot
        za_idx (int): index of zenith angle to plot
        savefig (Optional[Path], optional): directory name where the figures are saved. if None, plots will not be saved, only shown. Defaults to None.

    Returns:
        None
    """
    fontsize = 11    
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
    #line propoerties
    lprops = {
        '5577': {'color': 'green', 'linestyle': LINESTYLE_DICT['dotted'], 'lw': .95},
        '6300': {'color': 'red', 'linestyle': LINESTYLE_DICT['loosely dashed'], 'lw': 0.95},
    }

    species = list(lprops.keys())
    
    # ------------------------------------------------------------
    # FIGURE SETUP: N rows (one per date), 2 columns
    # ------------------------------------------------------------

    N = len(dates)
    fig, ax = plt.subplots(N, 2, figsize=(12, 3 * N),  gridspec_kw={'wspace': 0.01, 'hspace': 0.1, 'left': 0.1}) #, hspace = 0.15, wspace=0.025)

    # --------------------------------------------------------
    # time axis limits
    # --------------------------------------------------------
    tstart = []
    tend = []
    for date in dates:
        ds = compiled_fits[date]
        tstart.append(datetime.fromtimestamp(float(ds.tstamp.min()), tz=timezone.utc))
        tend.append(datetime.fromtimestamp(float(ds.tstamp.max()), tz=timezone.utc))
        del ds
    
    start_hours = []
    end_hours = []
    for i in range(N):
        start_dt = tstart[i]
        end_dt = tend[i]

        start_hours.append(start_dt.hour + start_dt.minute / 60.0 + start_dt.second / 3600.0)
        # Compute end hour relative to start day
        if end_dt.date() == start_dt.date():
            end_hours.append(end_dt.hour+ end_dt.minute / 60.0 + end_dt.second / 3600.0)
        else:
            # add 24 to indicate crossing midnight
            end_hours.append(24 + end_dt.hour +  end_dt.minute / 60.0 + end_dt.second / 3600.0 )
    min_hour = min(start_hours)
    max_hour = max(end_hours)

    # get xlim for each subplot
    xlims = []
    for i in range(N):
        date_dt = tstart[i].replace(hour=0, minute=0, second=0, microsecond=0)
        xlims.append(get_hour_range_for_date(date_dt, min_hour, max_hour))

    # ------------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------------
    ystarts, yends = [],[]
    for i,date in enumerate(dates):

        # --- Load data ---
        fds = compiled_fits[date].copy() #fitted dataset
        ods = compiled_counts[date].copy() #observed dataset
        
        # --- Conversion factor to Rayleighs ---
        slit_width = 0.01 #,cm  ,#100UM
        foreoptic_fl= 2.5 #,cm  ,#25MM
        dza = np.mean([np.deg2rad(np.mean(np.diff(fds.za.data))),np.deg2rad(np.mean(np.diff(ods.za.data)))])
        aw = slit_width/foreoptic_fl * dza # cm^2 sr
        to_rayleighs = 1 / aw / (4*np.pi*1e6)  # photons/(s cm^2 sr) to Rayleighs

        tstamps = ods.tstamp.values
        time = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in tstamps]
        height = ods.za.values
        dheight = np.rad2deg(dza) 
     
        # --- Per-species Plotting ---
        handles_total = {spec: [] for spec in species}
        labels_total = {spec: [] for spec in species}

        if len(dates) == 1:
            axL = ax[0]
            axR = ax[1]
        else:
            axL = ax[i, 0]
            axR = ax[i, 1]

        for spec in species:
            # --- Extract data ---
            obs = ods[spec].values.T[::-1, :] * to_rayleighs/1000 #kR
            stds = ods[spec + '_err'].values.T[::-1, :] * to_rayleighs/1000 #kR

            fit = fds[spec + '_mean'].values.T[::-1, :] * to_rayleighs/1000 
            fstd = fds[spec + '_std'].values.T[::-1, :] * to_rayleighs/1000   

            # --- Plot observed ---
            if spec == '5577':
                ax_curr = axR
            else:
                ax_curr = axL

            obs_line = ax_curr.plot(time, obs[za_idx,:], **lprops[spec])
            fill_obs = ax_curr.fill_between(time, obs[za_idx,:]-stds[za_idx,:], obs[za_idx,:]+stds[za_idx,:],\
                                             color=lprops[spec]['color'], alpha=0.2)

            # --- Plot fitted ---
            color = 'black'
            lw = 0.55

            fit_line = ax_curr.plot(time, fit[za_idx,:],color = color, linewidth= lw )
            # fill_fit = ax_curr.fill_between(time, fit[za_idx,:]-fstd[za_idx,:], fit[za_idx,:]+fstd[za_idx,:], \
            #                                 color=color, alpha=0.1)

            # --- Collect legend handles ---
            handles_total[spec].append((obs_line[0], fill_obs)) # fit and fill as one legend entry of observed
            handles_total[spec].append(fit_line[0]) # fit and fill as one legend entry of fitted

            # handles_total[spec].append((fit_line[0], fill_fit)) # fit and fill as one legend entry of fitted

            labels_total[spec].append(f'Observed ±1σ')
            # labels_total[spec].append(f'Modeled ±1σ')
            labels_total[spec].append(f'Modeled')



        # --- Y limits ---
        axL.set_xlim(xlims[i])
        axR.set_xlim(xlims[i])

        # --- Y labels ---
        axR.set_ylabel('[KeV]')
        axL.set_ylabel('[KeV]')

        # --- Titles only on first row ---
        if i == 0:
            axL.set_title('Red Line (6300 $\AA$) ', pad=28, fontsize=fontsize) # type: ignore
            axR.set_title('Green Line (5577 $\AA$)', pad=28, fontsize=fontsize) # type: ignore

        # --- Legend setup ---        
        if i == 0:
            for spec in species:
                handles,labels = handles_total[spec], labels_total[spec]
                if spec == '5577':
                    ax_curr = axR
                else:
                    ax_curr = axL
                ax_curr.legend(
                handles, labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.175),
                bbox_transform=ax_curr.transAxes,
                ncol=2,
                fontsize=fontsize,
                frameon=False,

            )
                    
    plt.show()
    if savefig is not None:
        savepath = Path(savefig)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        if len(dates) > 1:
            savefn = savepath/ f'fit_stats_{dates[0].split('T')[0]}-{dates[-1].split('T')[0]}.png'
        else:
            savefn = savepath/ f'fit_stats_{dates[0].split('T')[0]}.png'
        fig.savefig(savefn, dpi=300)
        


def plot_fit_vs_observed_multidates_col1(compiled_fits: dict[str, xr.Dataset],compiled_counts: dict[str, xr.Dataset], dates: List[str], za_idx:int, savefig: Optional[Path] = None,same_yscale: bool= False) -> None:
    fontsize = 10
    
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
    #line propoerties
    lprops = {
        '5577': {'color': 'green', 'linestyle': LINESTYLE_DICT['dashed'], 'lw': 0.85},
        '6300': {'color': 'red', 'linestyle': LINESTYLE_DICT['dashed'], 'lw': 0.85},
    }

    species = list(lprops.keys())
    
    # ------------------------------------------------------------
    # FIGURE SETUP: N rows (one per date), 2 columns
    # ------------------------------------------------------------

    N = len(dates)
    nrows = N
    ncols = 1
    width_per_col = 4
    height_per_row = 2
    fig, ax = plt.subplots(N, 1, figsize=(width_per_col * ncols,
        height_per_row * nrows), sharey=same_yscale, gridspec_kw={'wspace': 0.01, 'hspace': 0.4, 'left':0.1}) #, hspace = 0.15, wspace=0.025)
    

    # --------------------------------------------------------
    # time axis limits
    # --------------------------------------------------------
    tstart = []
    tend = []
    for date in dates:
        ds = compiled_fits[date]
        tstart.append(datetime.fromtimestamp(float(ds.tstamp.min()), tz=timezone.utc))
        tend.append(datetime.fromtimestamp(float(ds.tstamp.max()), tz=timezone.utc))
        del ds
    
    start_hours = []
    end_hours = []
    for i in range(N):
        start_dt = tstart[i]
        end_dt = tend[i]

        start_hours.append(start_dt.hour + start_dt.minute / 60.0 + start_dt.second / 3600.0)
        # Compute end hour relative to start day
        if end_dt.date() == start_dt.date():
            end_hours.append(end_dt.hour+ end_dt.minute / 60.0 + end_dt.second / 3600.0)
        else:
            # add 24 to indicate crossing midnight
            end_hours.append(24 + end_dt.hour +  end_dt.minute / 60.0 + end_dt.second / 3600.0 )
    min_hour = min(start_hours)
    max_hour = max(end_hours)

    # get xlim for each subplot
    xlims = []
    for i in range(N):
        date_dt = tstart[i].replace(hour=0, minute=0, second=0, microsecond=0)
        xlims.append(get_hour_range_for_date(date_dt, min_hour, max_hour))

    # ------------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------------
    for i,date in enumerate(dates):

        # --- Load data ---
        fds = compiled_fits[date].copy() #fitted dataset
        ods = compiled_counts[date].copy() #observed dataset
        
        # --- Conversion factor to Rayleighs ---
        slit_width = 0.01 #,cm  ,#100UM
        foreoptic_fl= 2.5 #,cm  ,#25MM
        dza = np.mean([np.deg2rad(np.mean(np.diff(fds.za.data))),np.deg2rad(np.mean(np.diff(ods.za.data)))])
        aw = slit_width/foreoptic_fl * dza # cm^2 sr
        to_rayleighs = 1 / aw / (4*np.pi*1e6)  # photons/(s cm^2 sr) to Rayleighs

        tstamps = ods.tstamp.values
        time = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in tstamps]
        height = ods.za.values
        dheight = np.rad2deg(dza) 
     
        # --- Per-species Plotting ---
        # handles_total = {spec: [] for spec in species}
        # labels_total = {spec: [] for spec in species}
        handles_total, labels_total = [], []

        ax_curr = ax[i]

        for spec in species:
            # --- Extract data ---
            obs = ods[spec].values.T[::-1, :] * to_rayleighs/1000 #kR
            stds = ods[spec + '_err'].values.T[::-1, :] * to_rayleighs/1000 #kR

            fit = fds[spec + '_mean'].values.T[::-1, :] * to_rayleighs/1000 
            fstd = fds[spec + '_std'].values.T[::-1, :] * to_rayleighs/1000   

            # --- Plot observed ---

            obs_line = ax_curr.plot(time, obs[za_idx,:], **lprops[spec])
            fill_obs = ax_curr.fill_between(time, obs[za_idx,:]-stds[za_idx,:], obs[za_idx,:]+stds[za_idx,:],\
                                             color=lprops[spec]['color'], alpha=0.2)

            # --- Plot fitted ---
            lw = 1

            fit_line = ax_curr.plot(time, fit[za_idx,:],color = lprops[spec]['color'], linewidth= lw,  )
            # fill_fit = ax_curr.fill_between(time, fit[za_idx,:]-fstd[za_idx,:], fit[za_idx,:]+fstd[za_idx,:], \
            #                                 color=color, alpha=0.1)
            # ax_curr.set_aspect('auto')

            # --- Collect legend handles ---
            handles_total.append((obs_line[0], fill_obs)) # fit and fill as one legend entry of observed
            handles_total.append(fit_line[0]) # fit and fill as one legend entry of fitted

            # handles_total[spec].append((fit_line[0], fill_fit)) # fit and fill as one legend entry of fitted

            labels_total.append(f'{spec} Observed ±1σ')
            # labels_total[spec].append(f'Modeled ±1σ')
            labels_total.append(f'{spec} Modeled')



        # --- Y limits ---
        ax_curr.set_xlim(xlims[i])
        ax_curr.set_xlim(xlims[i])

        # --- Y labels ---
        # ax_curr.set_ylabel('Emission Brightness [kR]')
        fig.text(0.0001, 0.5, 'Emission Brightness [kR]', va='center', rotation='vertical', fontsize=fontsize)

        # --- Titles only on first row ---
        # if i == 0:
            # axL.set_title('Red Line (6300 $\AA$) ', pad=28, fontsize=fontsize) # type: ignore
            # axR.set_title('Green Line (5577 $\AA$)', pad=28, fontsize=fontsize) # type: ignore

        # --- Legend setup ---        
        if i == 0:
            handles,labels = handles_total, labels_total
            ax_curr.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.45),
            bbox_transform=ax_curr.transAxes,
            ncol=len(handles)//2,
            fontsize=fontsize,
            frameon=False,
            )
                    
    plt.show()
    if savefig is not None:
        savepath = Path(savefig)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        if len(dates) > 1:
            savefn = savepath/ f'fit_stats_{dates[0].split('T')[0]}-{dates[-1].split('T')[0]}.png'
        else:
            savefn = savepath/ f'fit_stats_{dates[0].split('T')[0]}.png'
        fig.savefig(savefn, dpi=300)
# %%
#%%
suffixes = init('nqe', rootdir='model_neutral_qe')
print(f'Found suffixes: {suffixes}')
compiled_stats = compile_fit_intensity_stats(suffixes[1:], base_suffix=suffixes[0], rootdir='model_neutral_qe')

alldates = list(compiled_stats.keys())
counts_dir = Path('/home/charmi/locsststor/proc/hmsao/l2c')
complied_counts = compile_counts_data(counts_dir, alldates)
#%%
dates = alldates[:2]
plot_fit_vs_observed_multidates_colN(compiled_stats, complied_counts, dates, za_idx=20)#, savefig=Path('./plots/fit_vs_observed/'))

plot_fit_vs_observed_multidates_col1(compiled_stats, complied_counts, dates, za_idx=20)#, savefig=Path('./plots/fit_vs_observed/'))
#%%
