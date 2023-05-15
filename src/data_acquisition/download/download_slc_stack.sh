CDIR=$(pwd)

cd ~
cd scratch/data/uavsar/
mkdir slcs
cd slcs

wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC.dop

wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s1_2x8.llh
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s1_2x8.lkv
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21002_004_210115_L090VV_01_BC_s1_1x1.slc
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21002_004_210115_L090VV_01_BC.ann
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21004_003_210120_L090VV_01_BC_s1_1x1.slc
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21004_003_210120_L090VV_01_BC.ann

wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s2_2x8.llh
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s2_2x8.lkv
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21002_004_210115_L090VV_01_BC_s2_1x1.slc
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21002_004_210115_L090VV_01_BC.ann
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21004_003_210120_L090VV_01_BC_s2_1x1.slc
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21004_003_210120_L090VV_01_BC.ann

wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s3_2x8.llh
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s3_2x8.lkv
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21002_004_210115_L090VV_01_BC_s3_1x1.slc
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21002_004_210115_L090VV_01_BC.ann
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21004_003_210120_L090VV_01_BC_s3_1x1.slc
wget -N https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_21004_003_210120_L090VV_01_BC.ann
cd $CDIR