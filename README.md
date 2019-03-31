# CS3244 Project

## Developer Manual
### Setting up with virtualenv
```bash
git clone git@github.com:jlks96/CS3244-Project.git
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```

Download the image dataset from `https://www.kaggle.com/paultimothymooney/breast-histopathology-images` and extract.

### Setting up without virtualenv (Ensure that you are on python 3!)
```bash
git clone git@github.com:jlks96/CS3244-Project.git
pip install -r requirements.txt
```

### Running the networks

## With python (For training)
```bash
python feedforward.py
python convnet.py
```
## With jupyter (For visualization and analysis)
```bash
jupyter notebook 
```
Then run use the jupyter GUI with feedforward.ipnyb or convnet.ipynb