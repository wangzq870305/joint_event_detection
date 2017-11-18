cd ../rouge-summary
perl prepare4rouge.pl
cp result.py OUTPUT/
cp ROUGE-1.5.5.pl OUTPUT/
cp -rf data OUTPUT/
cd OUTPUT
python result.py
