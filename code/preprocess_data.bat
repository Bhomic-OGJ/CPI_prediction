::@echo off
REM Set the dataset name
::set DATASET=human
REM set DATASET=celegans
set DATASET=b_cancer

REM Set the radius
REM set radius=0
set radius=1
REM set radius=2
REM set radius=3

REM Set the ngram
set ngram=2
REM set ngram=3

REM Run the Python script with arguments
python preprocess_data.py %DATASET% %radius% %ngram%
