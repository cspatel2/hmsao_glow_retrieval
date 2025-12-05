#%%
from pathlib import Path
from settings import ROOT_DIR
from typing import List, Optional, SupportsFloat as Numeric
import natsort
import os
#%%
def init(suffix: str, run=False, rootdir:str|Path = ROOT_DIR) -> List[str]:
    """Populate the directories with the required results.

    Returns:
        List[str]: A list of valid suffixes for the directories.
    """
    if isinstance(rootdir, str):
        rootdir = Path(rootdir)
    dirs = list(rootdir.glob(f'keomodel_{suffix}*'))
    suffixes = [d.name.split('_', 1)[-1] for d in dirs]
    suffixes = natsort.natsorted(suffixes)
    if run:
        for suff in suffixes:
            dirname = ROOT_DIR / f'keomodel_{suff}'
            if not dirname.exists():
                print(f'[ERROR] {dirname} does not exist. Skipping.')
                continue
            print(f'Running generate_vert for {suff}')
            os.system(f'python generate_vert.py {suff}')
            print(f'Running fit_den for {suff}')
            os.system(f'python fit_den.py {suff}')
            print(f'Running fit_loc for {suff}')
            os.system(f'python fit_loc.py {suff}')
            print(f'Running fit_tec for {suff}')
            os.system(f'python fit_tec.py {suff}')

    return suffixes