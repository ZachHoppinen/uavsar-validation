CDIR=$(pwd)

cd ~
cd scratch/data/uavsar/
mkdir uavsar_geometry
cd uavsar_geometry

wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC.dop
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s1_2x8.llh
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s1_2x8.lkv

wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC.dop
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s2_2x8.llh
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s2_2x8.lkv

wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC.dop
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s3_2x8.llh
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_23205_01/lowman_23205_01_BC_s3_2x8.lkv

wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_05208_01/lowman_05208_01_BU.dop
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_05208_01/lowman_05208_01_BU_s1_2x8.llh
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_05208_01/lowman_05208_01_BU_s1_2x8.lkv

wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_05208_01/lowman_05208_01_BU.dop
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_05208_01/lowman_05208_01_BU_s2_2x8.llh
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_05208_01/lowman_05208_01_BU_s2_2x8.lkv

wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_05208_01/lowman_05208_01_BU.dop
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_05208_01/lowman_05208_01_BU_s3_2x8.llh
wget https://downloaduav2.jpl.nasa.gov/Release30/lowman_05208_01/lowman_05208_01_BU_s3_2x8.lkv

cd $CDIR