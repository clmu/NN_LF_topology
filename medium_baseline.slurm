#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account="share-ie-iel"
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30000
#SBATCH --job-name="medium_baseline"
#SBATCH --output=medium_baseline.out
#SBATCH --mail-user="clemensm@stud.ntnu.no"
#SBATCH --mail-type=ALL

set -o errexit # exit on errors

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"



module purge
module load Anaconda3/2022.05

module list

conda init bash
source /cluster/home/clemensm/.bash_profile

conda activate clemens

python -c "import NN_function;
NN_function.NN_obj_based(dataset='slim', network_name='medium', remark='baseline', l_rate=1e-3, batch_size=20, epochs=50, loss_function_list=['SquaredLineFlowLoss'])"


uname -a


