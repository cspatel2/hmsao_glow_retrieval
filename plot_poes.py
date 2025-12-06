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
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy
import xarray as xr
from datetime import datetime, timezone
from plotting_functions import init
from typing import List, Optional, SupportsFloat as Numeric

from common_functions import get_date
from tqdm import tqdm
import warnings
from settings import ROOT_DIR, Directories
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# # %%
#
def compile_poes_data(datadir: Path| str, dates: List[str]) -> dict[str,xr.Dataset]:
    """Compile POES data for a specific date into a xarray Dataset.

    Args:
        datadir (Path): Directory containing POES data files.
        date (str): Date string in 'YYYY-MM-DD' format.

    Returns:
        xr.Dataset: Compiled POES data for the specified date.
    """
    if isinstance(datadir, str):
        datadir = Path(datadir)
    compiled = {}
    for date in tqdm(dates):
        fns = list(Path('../data/POES_data').glob(f'*{date.split('T')[0]}*.nc'))
        dslist = [] # list for the same date, different dats
        for fn in fns:
            ds = xr.open_dataset(fn)
            pds =  xr.Dataset(
                            {
                                'Q': (('tstamp'),ds['ted_ele_eflux_atmo_total'].data, ds['ted_ele_eflux_atmo_total'].attrs | {'Var_name_in_ogds': 'ted_ele_eflux_atmo_total' }),
                                'Q_err': (('tstamp'),ds['ted_ele_eflux_atmo_total_err'].data, ds['ted_ele_eflux_atmo_total_err'].attrs | {'Var_name_in_ogds': 'ted_ele_eflux_atmo_total_err' }),
                            },
                            coords={
                                'tstamp': ('tstamp',ds['time'].data*1e-3, {'units':'seconds since 1970-01-01 00:00:00'}),
                                'time':('tstamp', [datetime.fromtimestamp(t, tz=timezone.utc) for t in ds['time'].data*1e-3], {'description':'UTC time of observation'}),
                                'lon': ('tstamp',ds['lon'].data, ds['lon'].attrs),
                                'lat': ('tstamp',ds['lat'].data, ds['lat'].attrs),
                            })
            dslist.append(pds)

        compiled[date] = xr.merge(dslist)

    return compiled

def compile_fit_data(suffixes: List[str], base_suffix: Optional[str] = None, rootdir:str|Path = ROOT_DIR) -> dict[str, xr.Dataset]:
    """Compile fit data for a specific date into a xarray Dataset.

    Args:
        datadir (Path): Directory containing fit data files.
        date (str): Date string in 'YYYY-MM-DD' format.

    Returns:
        xr.Dataset: Compiled fit data for the specified date.
    """
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
                fds = xr.Dataset(
                    {
                        'Q': (('tstamp'), ds['fit_params'].sel(param='Q').data, {'units':'erg/cm^2/s', 'description':'total Energy flux of precipitating electrons', 'Note':'1 erg/cm^2/s = 1 mW/m^2'}),
                        'E': (('tstamp'), ds['fit_params'].sel(param='Echar').data, {'units':'eV', 'description':'characteristic energy of precipitating electrons'}),
                    },
                    coords={
                        'tstamp': ('tstamp', ds['tstamp'].data, {'units':'seconds since 1970-01-01 00:00:00'}),
                        'time':('tstamp', [datetime.fromtimestamp(t, tz=timezone.utc) for t in ds['tstamp'].data], {'description':'UTC time of observation'}),
                        'lon': ('tstamp', ds['lon'].data, ds['lon'].attrs),
                        'lat': ('tstamp', ds['lat'].data, ds['lat'].attrs),
                    }
                )
                dslist.append(fds)
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

#%%
def plot_fit_vs_poes(compiled_poes: dict[str,xr.Dataset], compiled_fits: dict[str,xr.Dataset],dates:list[str], location: dict[str, float], same_yscale: bool = True, savefig: Optional[str] = None) -> None:
    fontsize = 12
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
        'Q': {'color': 'black', 'linestyle': LINESTYLE_DICT['dashed'], 'lw': .95},
        # 'E': {'color': 'blue', 'linestyle': LINESTYLE_DICT['dotted'], 'lw': 0.95},
    }

    species = list(lprops.keys())

    # ------------------------------------------------------------
    # FIGURE SETUP: N rows (one per date), 2 columns
    # ------------------------------------------------------------

    N = len(dates)
    nrows = N
    ncols = 1
    width_per_col = 6
    height_per_row = 3
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

        # --- load data ---
        pds = compiled_poes[date].copy()
        fds = compiled_fits[date].copy()
        # --- filter POES data for location ---
        tds = pds
        # lonmin = location['lon'] - 30
        # lonmax = location['lon'] + 30
        # tds = pds.where(pds.lon > lonmin, drop=True)
        # tds = tds.where(tds.lon < lonmax, drop=True)
        latmin = location['lat'] - 20
        latmax = location['lat'] + 20
        tds = tds.where(tds.lat > latmin, drop=True)
        tds = tds.where(tds.lat < latmax, drop=True)

        #per species plotting
        handles_total, labels_total = [], []
        ax_curr = ax[i]
        if len(species) > 1:
                #Enegry
                cax = ax_curr.twinx()

        ymins = []
        ymaxs = []
        for si, spec in enumerate(species):
            if si > 1: ax_curr = cax
            # --- extract data ---
            otime = tds['time'].values
            obs = tds[spec].values
            obs_std = tds[spec + '_err'].values

            ftime = fds['time'].values
            fit = fds[spec + '_mean'].values
            fit_std = fds[spec + '_std'].values

            ymins.append(np.nanmax(obs))
            ymaxs.append(np.nanmax(obs))
            ymins.append(np.nanmax(fit-fit_std))
            ymaxs.append(np.nanmax(fit+fit_std))


            # --- plot data ---
            obs_line = ax_curr.plot(otime, obs, color=lprops[spec]['color'], lw=1.5, ls = '--')
            # obs_fill = ax_curr.fill_between(otime, obs - obs_std, obs + obs_std, color=lprops[spec]['color'], alpha=0.2)

            fit_line = ax_curr.plot(ftime, fit, **lprops[spec])
            fit_fill = ax_curr.fill_between(ftime, fit - fit_std, fit + fit_std, color=lprops[spec]['color'], alpha=0.2)

            handles_total.append((obs_line[0]))#, obs_fill)) # fit and fill as one legend entry of observed
            handles_total.append((fit_line[0], fit_fill)) # fit and fill as one legend entry of fitted
            labels_total.append(f'POES')
            labels_total.append(f'HiT&MIS')
        
        # --- limits ---
        ax_curr.set_xlim(xlims[i])
        # ax_curr.set_xlim(xlims[i])

        ax_curr.set_ylim(-0.01, np.max(ymaxs)*0.6)

        # --- Y labels ---
        # ax_curr.set_ylabel('Emission Brightness [kR]')
        fig.text(0.0001, 0.5, 'Total Enegry FLux, Q [mW/m$^{2}$]', va='center', rotation='vertical', fontsize=fontsize)
        if len(species) > 1:
            fig.text(0.99, 0.5, 'Characteristic Energy, E$_{o}$ [eV]', va='center', rotation='vertical', fontsize=fontsize, color = lprops['E']['color'])
            cax.tick_params(axis='y', colors=lprops['E']['color'])

        # --- Legend setup ---   
        if len(species) > 1:
            legend_cols = len(handles_total)//2
        else:
            legend_cols = len(handles_total)      
        if i == 0:
            handles,labels = handles_total, labels_total
            ax_curr.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.25),
            bbox_transform=ax_curr.transAxes,
            ncol=legend_cols,
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
            




#%%

datadir = '../data/POES_data'
rootdir = 'model_neutral_qe'
suffixes = init('nqe', rootdir=rootdir)
print(f'Found suffixes: {suffixes}')
comp_fits = compile_fit_data(suffixes[1:], base_suffix=suffixes[0], rootdir=rootdir)
alldates = list(comp_fits.keys())
comp_poes = compile_poes_data(datadir, alldates)


# %%

LOCATION = {'lat': 67.84080387407106, 'lon':20.410219771333818}
dates = alldates[:2] 
plot_fit_vs_poes(comp_poes, comp_fits, dates, LOCATION)

        


# %%
