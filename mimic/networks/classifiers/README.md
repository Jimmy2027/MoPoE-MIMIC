# Training classifiers on the MIMIC-database

Run the script `main_train_clf_mimic.py` to train classifiers for any modality in [PA, Lateral, text].

Different models as image classifiers can be chosen with the flag `--img_clf_type`. Chose between cheXnet and resnet.
 
The trained classifiers will be saved under `--clf_dir`/`--clf_save_m{1,2,3}}`.