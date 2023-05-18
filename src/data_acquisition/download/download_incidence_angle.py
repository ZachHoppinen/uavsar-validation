from uavsar_pytools import UavsarImage
from pathlib import Path

data_dir = Path('/bsuhome/zacharykeskinen/scratch/data/uavsar')
ncs_dir = data_dir.joinpath('ncs')
tif_dir = data_dir.joinpath('tifs', 'Lowman, CO')
geom_dir = ncs_dir.joinpath('geometry')

UavsarImage(url = 'https://uavsar.asfdaac.alaska.edu/UA_lowman_23205_21021_006_210322_L090_CX_01/lowman_23205_21021_006_210322_L090_CX_01.inc', work_dir=str(geom_dir)).url_to_tiff()