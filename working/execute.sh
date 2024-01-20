opus_read -d JW300 -s fr -t fon -wm moses -w input/jw300_fon.fr input/jw300.fon
opus_read -d JW300 -s fr -t ee -wm moses -w input/jw300_ee.fr input/jw300.ee
pip install rouge_score

python3 -m working.prepare_data

python3 -m working.pretraining

python3 -m working.train --fold 0
python3 -m working.predict --beam 3 --pn beam3fold0 --fold 0

python3 -m working.train --fold 1
python3 -m working.predict --beam 3 --pn beam3fold1 --fold 1

python3 -m working.train --fold 2
python3 -m working.predict --beam 3 --pn beam3fold2 --fold 2

python3 -m working.train --fold 3
python3 -m working.predict --beam 3 --pn beam3fold3 --fold 3

python3 -m working.train --fold 4
python3 -m working.predict --beam 3 --pn beam3fold4 --fold 4

python3 -m working.ensemble
