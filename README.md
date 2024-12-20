# FinalProject

Machine Learning Final Project

## Preparation

Download [googlenews-vectors-negative300.bin](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300) and move it to `./resources`.

## Command

```bash
conda create -n ml python=3.10 -y
conda activate ml
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py
```
## Docker
If docker has been installed, you can run `make run` to execute the program through the container.