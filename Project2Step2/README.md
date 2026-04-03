### Data
Put the data from `(https://drive.google.com/file/d/1FAJfMkx86tyJP9pJqOn3SofvYGksSeP3/view?usp=drive_link)` into the data/raw directory. The name should be `HIGGS.csv.gz` exactly so that the correct file is called.

### Activate  Environment
In your terminal run the following commands
```
python3 -m venv .venv
source .venv/bin/activate
```

After this press `Ctrl+Shift+P` and select the virtual environment as your python interpreter. 

Once you have selected your python interpreter you can run the cleaning script by typing this in your terminal.
```
python scripts/clean.py
```
