
DEBUG=false
LOGDIR=""
METHOD="moe"
LIKELIHOOD_M1="laplace"
LIKELIHOOD_M2="categorical"
DIR_DATA="/cluster/work/vogtlab/Projects"
DIR_TEXT="/cluster/home/$USER/projects/multimodality/data/CelebA"
DIR_CLF="/cluster/home/$USER/projects/multimodality/trained_classifiers/CelebA"
DIR_EXPERIMENT_BASE="/cluster/work/vogtlab/Group/thomas_work/multimodality/experiments/joint_elbo"
DIR_EXPERIMENT="${DIR_EXPERIMENT_BASE}/CelebA/${METHOD}/factorized/${LIKELIHOOD_M1}_${LIKELIHOOD_M2}"
PATH_INC_V3="/cluster/home/suttetho/projects/multimodality/experiments/inception_v3/pt_inception-2015-12-05-6726825d.pth"
DIR_FID="$TMPDIR/CelebA"

module load cuda/10.0.130
module load cudnn/7.5
module load openblas/0.2.19

source activate vae

mkdir "${TMPDIR}/CelebA"
cp "${DIR_DATA}/CelebA/img_align_celeba.zip" "${TMPDIR}/CelebA/"
cp "${DIR_DATA}/CelebA/list_eval_partition.csv" "${TMPDIR}/CelebA/"
cp "${DIR_DATA}/CelebA/list_attr_celeba.csv" "${TMPDIR}/CelebA/"
unzip -o "${TMPDIR}/CelebA/img_align_celeba.zip" -d "${TMPDIR}/CelebA/"
ls "${TMPDIR}/CelebA/img_align_celeba/"

python main_celeba.py --dir_data=$TMPDIR \
	              --dir_text=$DIR_TEXT \
	              --dir_clf=$DIR_CLF \
	              --dir_experiment=$DIR_EXPERIMENT \
	              --inception_state_dict=$PATH_INC_V3 \
	              --dir_fid=$DIR_FID \
		      --method=$METHOD \
	              --beta=2.5 \
	              --beta_style=2.0 \
	              --beta_content=1.0 \
	              --beta_m1_style=1.0 \
	              --beta_m2_style=5.0 \
	              --div_weight_m1_content=0.35 \
	              --div_weight_m2_content=0.35 \
	              --div_weight_uniform_content=0.3 \
	              --likelihood_m1=$LIKELIHOOD_M1 \
	              --likelihood_m2=$LIKELIHOOD_M2 \
	              --batch_size=256 \
	              --initial_learning_rate=0.0005 \
	              --eval_freq=25 \
	              --eval_freq_fid=25 \
	              --end_epoch=250 \
	              --factorized_representation \
	              --calc_nll \
	              --eval_lr \
	              --calc_prd \
	              --use_clf
	


