
import argparse
from utils.BaseFlags import parser as parser


parser.add_argument('--dataset', type=str, default='Mimic', help="name of the dataset")

# DATA DEPENDENT
parser.add_argument('--len_sequence', type=int, default=1024, help="length of sequence")
parser.add_argument('--style_pa_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_lat_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_text_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--DIM_img', type=int, default=128, help="number of classes on which the data set trained")
parser.add_argument('--DIM_text', type=int, default=128, help="number of classes on which the data set trained")
parser.add_argument('--likelihood_m1', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m2', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m3', type=str, default='categorical', help="output distribution")

# paths to save models
parser.add_argument('--encoder_save_m1', type=str, default='encoderM1', help="model save for encoder")
parser.add_argument('--encoder_save_m2', type=str, default='encoderM2', help="model save for encoder")
parser.add_argument('--encoder_save_m3', type=str, default='encoderM3', help="model save for decoder")
parser.add_argument('--decoder_save_m1', type=str, default='decoderM1', help="model save for decoder")
parser.add_argument('--decoder_save_m2', type=str, default='decoderM2', help="model save for decoder")
parser.add_argument('--decoder_save_m3', type=str, default='decoderM3', help="model save for decoder")
parser.add_argument('--vae_trimodal_save', type=str, default='vae_trimodal', help="model save for vae_bimodal")
parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")

# SAVE and LOAD
parser.add_argument('--clf_save_m1', type=str, default='clf_m1', help="model save for clf")
parser.add_argument('--clf_save_m2', type=str, default='clf_m2', help="model save for clf")
parser.add_argument('--clf_save_m3', type=str, default='clf_m3', help="model save for clf")


# LOSS TERM WEIGHTS
parser.add_argument('--beta_m1_style', type=float, default=1.0, help="default weight divergence term style modality 1")
parser.add_argument('--beta_m2_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--beta_m3_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--div_weight_m1_content', type=float, default=0.25, help="default weight divergence term content modality 1")
parser.add_argument('--div_weight_m2_content', type=float, default=0.25, help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_m3_content', type=float, default=0.25, help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_uniform_content', type=float, default=0.25, help="default weight divergence term prior")


