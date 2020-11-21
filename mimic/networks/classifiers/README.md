# Training classifiers on the MIMIC-database

Run the script `main_train_clf_mimic.py` to train classifiers for any modality in [PA, Lateral, text].

Different models as image classifiers can be chosen with the flag `--img_clf_type`. Chose between densenet and resnet.
 
The trained classifiers will be saved under `--clf_dir`/`--clf_save_m{1,2,3}}`.

## Example run:
```
python networks/classifiers/main_train_clf.py --modality PA --img_clf_type resnet --config_path path_to_my_json_config
```