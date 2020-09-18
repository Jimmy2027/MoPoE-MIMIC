

DEBUG=false
LOGDIR=""
METHOD="jsd"
LIKELIHOOD_M1="laplace"
LIKELIHOOD_M2="laplace"
LIKELIHOOD_M3="categorical"
DIR_DATA="/cluster/home/suttetho/projects/multimodality/data"
DIR_CLF="/cluster/home/$USER/projects/multimodality/trained_classifiers/MNISTSVHN"
DIR_EXPERIMENT_BASE="/cluster/work/vogtlab/Group/thomas_work/multimodality/experiments/joint_elbo"
DIR_EXPERIMENT="${DIR_EXPERIMENT_BASE}/${METHOD}/non_factorized/${LIKELIHOOD_M1}_${LIKELIHOOD_M2}_${LIKELIHOOD_M3}"
PATH_INC_V3="/cluster/home/suttetho/projects/multimodality/experiments/inception_v3/pt_inception-2015-12-05-6726825d.pth"
DIR_FID="$TMPDIR/MNIST_SVHN_text"

module load cuda/10.0.130
module load cudnn/7.5
module load openblas/0.2.19

source activate vae

cp -r "${DIR_DATA}/MNIST" "${TMPDIR}/"
cp -r "${DIR_DATA}/SVHN" "${TMPDIR}/"
cp -r "${DIR_DATA}/MNIST_SVHN" "${TMPDIR}/"

python main_svhnmnist.py --dir_data=$TMPDIR \
			--dir_clf=$DIR_CLF \
			--dir_experiment=$DIR_EXPERIMENT \
			--inception_state_dict=$PATH_INC_V3 \
			--dir_fid=$DIR_FID \
			--method=$METHOD \
			--style_m1_dim=0 \
			--style_m2_dim=0 \
			--style_m3_dim=0 \
			--class_dim=20 \
			--beta=5.0 \
			--likelihood_m1=$LIKELIHOOD_M1 \
			--likelihood_m2=$LIKELIHOOD_M2 \
			--likelihood_m3=$LIKELIHOOD_M3 \
			--batch_size=256 \
			--initial_learning_rate=0.001 \
			--eval_freq=25 \
			--eval_freq_fid=25 \
			--data_multiplications=20 \
			--num_hidden_layers=1 \
			--end_epoch=250 \

