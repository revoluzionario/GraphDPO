# GraphDPO

The codes and datasets for "Unlearning of Knowledge Graph Embedding via Preference Optimization". 

## Framework

![](README.assets/framework.jpg)



## Folder Structure

The structure of the folder is shown below:

```
 IncDE
 ├─checkpoint_pretrain
 ├─checkpoint_unlearning
 ├─data
 ├─logs
 ├─src
 ├─main.sh
 ├─ablation.sh
 ├─pretrain.py
 ├─unlearning.py
 ├─requirements.txt
 └README.md
```

Introduction to the structure of the folder:

- /checkpoint_pretrain: The generated pre-trained models are stored in this folder.
- /checkpoint_unlearning: The generated unlearning models are stored in this folder.
- /data: The datasets are stored in this folder.
- /logs: Logs for the training are stored in this folder.
- /src: Source codes are in this folder.
- /main.py: To run the IncDE.
- data_preprocess.py: To prepare the data processing.
- README.md: Instruct on how to realize IncDE.

## Folder Structure

All experiments are implemented on the NVIDIA RTX 3090Ti GPU with the PyTorch. The version of Python is 3.9.

Please run as follows to install all the dependencies:

```shell
pip3 install -r requirements.txt
```

## Usage

### Preparation

1. Unzip the dataset $fb15k-237-10.zip$ , $fb15k-237-20.zip$ ,$wn18rr-10.zip$ , and $wn18rr-20.zip$ in the folder of $data$.

### Main Results

2. Pretrain the KGE model:

```python
python pretrain.py
```

3. Run the unlearning script with this in the shell:

```shell
./main.sh
```

### Ablation Results

4. Run the ablation script with this in the shell:

```shell
./ablation.sh
```

