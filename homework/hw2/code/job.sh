
# NOTICE: Please do not remove the '#' before 'PBS'

# Select which queue (debug, batch) to run on
#PBS -q batch 

# Name of your job
#PBS -N ZYH_JOB

# Declaring job as not re-runnable
#PBS -r n

# Resource allocation (how many nodes? how many processors per node?)
#PBS -l nodes=4:ppn=12

# Max execution time of your job (hh:mm:ss)
# Debug cluster max limit: 00:05:00 
# Batch cluster max limit: 00:30:00
# Your job may got killed if you exceed this limit
#PBS -l walltime=00:05:00
#PBS -e error
#PBS -o result

host=$(cat $PBS_NODEFILE | sort | uniq)
host=$(echo $host | sed 's/\ /,/g')
nodes=$(cat $PBS_NODEFILE | sort | uniq | wc -l)
height=1024
width=768
a=-2
b=0.5
c=-1
d=1

cd $PBS_O_WORKDIR
export NUM_MPI_PROCESS_PER_NODE=1 # edit this line to set number of MPI process you want to use per node
export OMP_NUM_THREADS=12	# set max number of threads OpenMP can use per MPI task
rm output
rm output_static
time mpirun -np $((nodes*NUM_MPI_PROCESS_PER_NODE)) -ppn $NUM_MPI_PROCESS_PER_NODE -host $host ./MS_hybrid $OMP_NUM_THREADS $a $b $c $d $height $width output # edit this line to fit your needs!
time mpirun -np $((nodes*NUM_MPI_PROCESS_PER_NODE)) -ppn $NUM_MPI_PROCESS_PER_NODE -host $host ./MS_hybrid_static $OMP_NUM_THREADS $a $b $c $d $height $width output_static # edit this line to fit your needs!
time mpirun -np $((nodes*NUM_MPI_PROCESS_PER_NODE)) -ppn $NUM_MPI_PROCESS_PER_NODE -host $host ./MS_seq $OMP_NUM_THREADS $a $b $c $d $height $width output_seq # edit this line to fit your needs!
diff output output_static
