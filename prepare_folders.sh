mkdir input
mkdir input/folds
mkdir weights
mkdir notebooks
mkdir working
mkdir submissions

# Python files
touch working/config.py
touch working/dataset.py
touch working/models.py
touch working/predict.py
touch working/train.py
touch working/validation.py
touch working/prepare_data.py

opus_read -d JW300 -s fr -t fon -wm moses -w input/jw300_fon.fr input/jw300.fon
opus_read -d JW300 -s fr -t ee -wm moses -w input/jw300_ee.fr input/jw300.ee