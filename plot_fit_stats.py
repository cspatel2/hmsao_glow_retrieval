# %%
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import lzma
import os
from pathlib import Path
import pickle
from typing import List, Optional, SupportsFloat as Numeric

from matplotlib import pyplot as plt, ticker
import matplotlib
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import natsort
import numpy as np
import pytz
import scipy
import tqdm
import xarray
from common_functions import LINESTYLE_DICT, fill_array_1d, get_date
from settings import ROOT_DIR, Directories
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from plotting_functions import init
# %%

def compile_fit_stats(suffixes: List[str], base_suffix: Optional[str] = None, rootdir:str|Path = ROOT_DIR) -> dict[str, xarray.Dataset]:
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
    for date in tqdm.tqdm(dates, dynamic_ncols=True):
        dss = []
        for suffix in msuffixes:
            dirs = Directories(suffix=suffix, basedir=str(rootdir))
            with lzma.open(dirs.model_dir / f'fitres_{date}.xz', 'rb') as f:
                fitres = pickle.load(f)
                tstamps = []
                scales = []
                for vals in fitres:
                    tstamp, pert = vals
                    tstamps.append(tstamp)
                    if pert is not None:
                        scales.append(
                            (pert.x[0], pert.x[1], pert.x[2], pert.x[3], pert.x[4], pert.x[5], pert.x[6]))
                    else:
                        scales.append(
                            (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                scales = np.array(scales)
                ds = xarray.DataArray(
                    scales,
                    dims=['tstamp', 'species'],
                    coords={
                        'tstamp': tstamps,
                        'species': ['O', 'O2', 'N2', 'NO', 'N4S', 'Q', 'Echar']
                    },
                )
                dss.append(ds)
        dss = xarray.concat(dss, dim='suffix', compat='equals')
        ds = xarray.Dataset({'density': dss})
        ds['minval'] = ds.density.min(dim='suffix')
        ds['maxval'] = ds.density.max(dim='suffix')
        ds['meanval'] = ds.density.mean(dim='suffix')
        ds['stdval'] = ds.density.std(dim='suffix')
        ds['geomean'] = ds.density.std(dim='suffix')
        ds['geomean'].values = scipy.stats.mstats.gmean(ds.density, axis=0)
        stats[date] = ds
    return stats

def plot_fit_stats_multidates(stats: dict[str, xarray.Dataset], dates: List[str], savefig:str|None = None) -> None:
    """Plot the density statistics for multiple dates.

    Args:
        stats (dict[str, xarray.Dataset]): The compiled statistics.
        dates (List[str]): The dates to plot.
        savefig (str | None): If provided, the path to save the figure. if None, the figure will not be saved. Default is None.
    """
    fontszie = 12
    plt.rcParams.update({'font.size': fontszie,
                          'axes.labelsize': fontszie,
                         'axes.titlesize': fontszie, 
                         'legend.fontsize': fontszie,
                         'figure.titlesize': fontszie})
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
    # Line properties
    lprops = {
        'O': {'color': 'blue', 'linestyle': LINESTYLE_DICT['dotted'], 'label': 'O+', 'lw': .95},
        'O2': {'color': 'red', 'linestyle': LINESTYLE_DICT['loosely dashed'], 'label': 'O$_2$', 'lw': 0.95},
        'N2': {'color': 'forestgreen', 'linestyle': LINESTYLE_DICT['dashdot'], 'label': 'N$_2$', 'lw': 0.95},
        'NO': {'color': 'purple', 'linestyle': LINESTYLE_DICT['densely dashdotted'], 'label': 'NO', 'lw': 0.95},
        'N4S': {'color': 'darkorange', 'linestyle': LINESTYLE_DICT['dashdotdotted'], 'label': 'N$(^4S)$', 'lw': 0.95},
        'Q': {'color': 'black', 'linestyle': LINESTYLE_DICT['densely dashdotted'], 'label': 'Q', 'lw': 0.95},
        'Echar': {'color': 'blueviolet', 'linestyle': LINESTYLE_DICT['densely dashdotted'], 'label': 'E$_{o}$', 'lw': 0.95},
    }

    species = ['O', 'O2', 'N2', 'NO', 'N4S', 'Q', 'Echar']
    
    # ------------------------------------------------------------
    # FIGURE SETUP: N rows (one per date), 2 columns
    # ------------------------------------------------------------
    N = len(dates)
    fig, ax = plt.subplots(N, 2, figsize=(12, 3 * N), layout='constrained') #, hspace = 0.15, wspace=0.025)

    # locator = mdates.AutoDateLocator()
    # formatter = mdates.ConciseDateFormatter(locator)

    # --------------------------------------------------------
    # time axis limits
    # --------------------------------------------------------
    tstart = []
    tend = []
    for date in dates:
        ds = stats[date]

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

    for i, date in enumerate(dates):

        ds = stats[date]
        times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ds.tstamp.values]


        baseval = ds.isel(suffix=0)
        handles_left = []
        labels_left = []
        handles_right = []
        labels_right = []

        # ------------------------------------------------------------
        # Per-species plotting
        # ------------------------------------------------------------

        if len(dates) == 1:
            axL = ax[0]
            axR = ax[1]
            cax = ax[1].twinx()
        else:
            axL = ax[i, 0]
            axR = ax[i, 1]
            cax = ax[i, 1].twinx()

        for sp in species:
            bss = baseval.sel(species=sp)
            dss = ds.sel(species=sp)

            meanval = dss['meanval'].values
            minval = dss['minval'].values
            maxval = dss['maxval'].values

            if sp == 'Q':
                conv = 1e3
                line_R, = axR.plot(times, meanval * conv, **lprops[sp])
                fill_R = axR.fill_between(times, minval*conv, maxval*conv,
                                 color=lprops[sp]['color'], alpha=0.2)
                handles_right.append((line_R,fill_R))
                labels_right.append(fr'[{lprops[sp]["label"]}]$\pm 1\sigma$')

            elif sp == 'Echar':
                conv = 1e-3
                line_R, = cax.plot(times, meanval * conv, **lprops[sp])
                fill_R = cax.fill_between(times, minval*conv, maxval*conv,
                                 color=lprops[sp]['color'], alpha=0.2)
                handles_right.append((line_R,fill_R))
                labels_right.append(fr'[{lprops[sp]["label"]}]$\pm 1\sigma$')
                
            else:
                line_L, = axL.plot(times, meanval, **lprops[sp])
                fill_L = axL.fill_between(times, minval, maxval,
                                 color=lprops[sp]['color'], alpha=0.2)
                handles_left.append((line_L,fill_L))
                labels_left.append(fr'[{lprops[sp]["label"]}]$\pm 1\sigma$')
            

        # ------------------------------------------------------------
        # AXIS LIMITS
        # ------------------------------------------------------------
        axL.set_xlim(xlims[i])
        axR.set_xlim(xlims[i])
        cax.set_xlim(xlims[i])
        # print(f'Date: {date}, xlim: {xlims[i]}')

        # ------------------------------------------------------------
        # Titles only on first row
        # ------------------------------------------------------------
        if i == 0:
            axL.set_title('Density Scale-Factors', pad=43,)
            axR.set_title('Precipitation Parameters', pad=43)

        # --------------------------------------------------------
        # AXIS LABELS
        # --------------------------------------------------------
        # axL.set_ylabel('[cm$^{-3}$]')
        axR.set_ylabel('Q [uW/m$^2$]')
        cax.set_ylabel('E$_{o}$ [keV]', color=lprops['Echar']['color'])
        cax.tick_params(axis='y', labelcolor=lprops['Echar']['color'])

        # axL.set_ylabel('[cm$^{-3}$]')
        axR.set_ylabel('Total Energy Flux Q [uW/m$^2$]')
        cax.set_ylabel('Characteristic Energy E$_{o}$ [KeV]', color=lprops['Echar']['color'])
        cax.tick_params(axis='y', labelcolor=lprops['Echar']['color'])


        # --------------------------------------------------------
        # LEGENDS ONLY IN FIRST ROW
        # --------------------------------------------------------
        # print(f'Handles left: {handles_left}, labels left: {labels_left}')
        if i == 0:
            # Left subplot legend
            axL.legend(
                handles_left, labels_left,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.3),
                bbox_transform=axL.transAxes,
                ncol=len(handles_left)//1.5,
                # fontsize=12,
                frameon=False,

            )

            # Right subplot legend
            axR.legend(
                handles_right, labels_right,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.175),
                bbox_transform=axR.transAxes,
                ncol=3,
                # fontsize=12,
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
if __name__ == '__main__':
    # Populate directories and compile stats
    suffixes = init('nqe', rootdir='model_neutral_qe')
    print(f'Found suffixes: {suffixes}')
    compiled_stats = compile_fit_stats(suffixes[1:], base_suffix=suffixes[0], rootdir='model_neutral_qe')
    #%%
    #plot
    stats = compiled_stats.copy()
    dates = list(stats.keys())[:3]
    plot_fit_stats_multidates(stats, dates, savefig='./')

# %%

# %%
