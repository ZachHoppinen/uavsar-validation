module load netcdf/gcc/64/4.6.2

export NCS_FP="/bsuhome/zacharykeskinen/scratch/data/uavsar/ncs"

echo "Number of UAVSAR Pairs:"
echo $(ls $NCS_FP | grep .sd.nc | wc -l)

echo "Number of 232 Pairs:"
echo $(ls $NCS_FP | grep .sd.nc | grep 232_ | wc -l)

echo "Number of 052 Pairs:"
echo $(ls $NCS_FP | grep .sd.nc | grep 052_ | wc -l)

echo "232 pairs:"
export UV232=$(ls $NCS_FP | grep .sd.nc | grep 232_ )

for FP in $UV232
do
    export FP=${FP%".sd.nc"}
    echo ${FP#"232_"}
done

echo "052 pairs:"
export UV052=$(ls $NCS_FP | grep .sd.nc | grep 052_ )

for FP in $UV052
do

    export S_FP=${FP%".sd.nc"}
    echo ${S_FP#"052_"}
done