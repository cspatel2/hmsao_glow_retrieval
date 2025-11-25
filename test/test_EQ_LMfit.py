#%%
import numpy as np
import xarray as xr
from glowpython import maxwellian
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters,create_params, Model
from test_glow import get_geomag_for_glow, get_brightness_from_ver #, get_brightness_from_dat
from datetime import datetime
from glob import glob
import os
#%%

# %%
def get_brightness_ratio_from_dat(ds:xr.Dataset,wl:str,dt:datetime) -> float:
    ds = ds.sel(tstamp = dt.timestamp(), method='nearest')
    return ds[wl].values, ds[f'{wl}_err'].values




#residual function to minimize
def residual(params, wls, data, uncertainty):
    time = datetime.fromtimestamp(params['tstamp'].value)
    gmdict = get_geomag_for_glow(time)
    iono = maxwellian(time = time,
                    glat = params['lat'].value,
                    glon = params['lon'].value ,
                    Nbins= params['nbins'].value,
                    Q = params['Q'].value,
                    Echar=params['E'].value,
                    geomag_params=gmdict)
    model =[]
    for w in wls:
        lds = iono.sel(wavelength = w)
        model.append(get_brightness_from_ver(lds,420))
    return (np.array(data)-np.array(model))/np.array(uncertainty)
#%%
sweden = {'lat': 67.84, 'lon': 20.41, 'elev': 420}  # Approximate location of Kiruna
testdate = datetime(2025, 1, 27, 17, 42, 22, 255821)
# wls = ['5577','6300']
wls = ['6300']

# parameters of the function to vary 
params = create_params(tstamp = {'value':testdate.timestamp(),'vary':False},
                       lat = {'value':sweden['lat'],'vary':False},
                       lon = {'value':sweden['lon'],'vary':False},
                       nbins = {'value':350,'vary':False, 'min':10, 'max': 
                                350},
                       Q = {'vary':False, 'min':1e-4,'max':3},
                       E = {'vary':True, 'min':0,'max':1e4},
)
# measured data
data, unc = [],[]
for w in wls:
    yymmdd = testdate.strftime('%Y%m%d')
    fdir ='/home/charmi/locsststor/proc/hmsao_1b/202501'
    fname = glob(os.path.join(fdir,f'*{yymmdd}*{w}.nc'))
    ds = xr.open_dataset(fname[0])
    rdata,runc = get_brightness_za_from_dat(ds,w,testdate)

    # rdata,runc = get_brightness_from_dat(ds,w,testdate,0)

    

    data.append(rdata)
    unc.append(runc)

#%%
#minimise
out = minimize(fcn=residual,params=params, args = (wls,data[0],unc))
out

# %%
np.shape(data[0])
# %%
np.shape(runc)
# %%

