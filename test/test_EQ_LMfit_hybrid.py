#%%
import numpy as np
import xarray as xr
from glowpython import maxwellian
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters,create_params, Model
from test_glow import get_geomag_for_glow
from datetime import datetime
from glob import glob
import os

# %%
def get_brightness_ratio_from_dat(ds:xr.Dataset,wls:str,dt:datetime,zabinsize:float = 1) -> tuple[np.ndarray, np.ndarray] :
    assert len(wls) == 2,'Wls should be of the form [wl1,wl2] such that the brightness-ratio = B[wl1]/B[wl2]'
    ds1 = ds.sel(tstamp = dt.timestamp(), method='nearest')
    ds2 = ds.sel(tstamp = dt.timestamp(), method='nearest')
    
    wl = wls[0]
    b1,e1 = ds1[wl].values, ds1[f'{wl}_err'].values

    wl = wls[1]
    b2,e2 = ds2[wl].values, ds2[f'{wl}_err'].values

    b_ratio = b1/b2
    b_err = b_ratio *np.sqrt((e1/b1)**2 + (e2/b2)**2)

    return b_ratio, b_err

def get_brightness_from_ver(ds:xr.Dataset, alt_min:float=None, alt_max:float = None) -> float:
    """Performs LOS intergration of VER to calcuate brightness. 

    Args:
        ds (xr.Dataset): glow Dataset with of selected wavelength
        alt_min (float, optional): lower limit of intergration. Defaults to None.
        alt_max (float, optional): upper limit of intergration. Defaults to None.

    Returns:
        float: brightness (R)
    """    
    ds = ds.sel(alt_km = slice(alt_min, alt_max))
    ds = ds.dropna('alt_km')
    altcm = ds.alt_km.values * 10e5  # km -> cm
    ver = ds.ver.values # VER(z) in photons cm-3 s-1
    
    return 1e-6 * np.trapezoid(altcm,ver) # R



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
        model.append(get_brightness_from_ver(lds,200))
    model_ratio = model[0]/model[1]
    return (np.array(data)-np.array(model))/np.array(uncertainty)
#%%
sweden = {'lat': 67.84, 'lon': 20.41, 'elev': 420}  # Approximate location of Kiruna
testdate = datetime(2025, 1, 27, 17, 42, 22, 255821)
# wls = ['5577','6300']
wls = ['6300', '5577']

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

