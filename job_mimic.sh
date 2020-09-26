DEBUG=false
LOGDIR=""
DATASET="Mimic"
METHOD="moe"
DIR_DATA="/cluster/work/vogtlab/Projects/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0"
DIR_CLF="/cluster/home/$USER/projects/multimodality/trained_classifiers/${DATASET}128"
DIR_EXPERIMENT_BASE="/cluster/home/klugh/multimodality/experiments/joint_elbo"
DIR_EXPERIMENT="${DIR_EXPERIMENT_BASE}/${DATASET}/${METHOD}/non_factorized"
PATH_INC_V3="/cluster/home/suttetho/projects/multimodality/experiments/inception_v3/pt_inception-2015-12-05-6726825d.pth"
DIR_FID="$TEMPFOLDER/${DATASET}"
TEMPFOLDER="${HOME}/tmp"

module load cuda/10.0.130
module load cudnn/7.5
module load openblas/0.2.19

source activate mimic

if [ ! -d "${TEMPFOLDER}/${DATASET}" ]; then
  echo "creating dir ${TEMPFOLDER}/${DATASET}"
  mkdir "${TEMPFOLDER}/${DATASET}" || exit 1
  cp "${DIR_DATA}/files_small128_pt.zip" "${TEMPFOLDER}/${DATASET}/"
  unzip -o "${TEMPFOLDER}/${DATASET}/files_small128_pt.zip" -d "${TEMPFOLDER}/${DATASET}/"
fi

python main_mimic.py --dir_data=$TEMPFOLDER/${DATASET} \
               	     --dir_clf=$DIR_CLF \
               	     --dir_experiment=$DIR_EXPERIMENT \
               	     --inception_state_dict=$PATH_INC_V3 \
               	     --dir_fid=$DIR_FID \
               	     --DIM_img=64 \
               	     --initial_learning_rate=0.0005 \
               	     --style_pa_dim=0 \
               	     --style_lat_dim=0 \
               	     --style_text_dim=0 \
               	     --class_dim=64 \
               	     --method=$METHOD \
               	     --beta=1.0 \
               	     --beta_style=3.0 \
               	     --beta_content=1.0 \
               	     --beta_m1_style=1.0 \
               	     --beta_m2_style=1.0 \
               	     --beta_m3_style=1.0 \
               	     --div_weight_m1_content=0.25 \
               	     --div_weight_m2_content=0.25 \
               	     --div_weight_m3_content=0.25 \
               	     --div_weight_uniform_content=0.25 \
               	     --batch_size=256 \
               	     --eval_freq=20 \
               	     --eval_freq_fid=300 \
               	     --end_epoch=300 \
                     --calc_nll \
                     --eval_lr \
                     --calc_prd \
                     --use_clf
               	#--factorized_representation \
	


