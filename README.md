# An empirical convolutional neural network approach for semantic relation classification

PyTorch implementation of Relcation Classification model described in our Neurocomputing paper [An empirical convolutional neural network approach for semantic relation classification(https://www.sciencedirect.com/science/article/pii/S0925231216000023) on the SemEval-2010 Task-8 dataset.

## Steps to run the experiments

### Requirements
* ``Python 2.7.12 ``
* ``PyTorch 0.4.1``
* ``panda 0.19.1``

### Datasets and word embeddings
* Dataset is already included in the directory ``./SemEval2010_task8_all_data``.
* Embedding file ``./data/word_vecs.pkl`` is generated from the released word embedding file ``GoogleNews-vectors-negative300.bin`` (http://code.google.com/p/word2vec/) by Mikolov.


### Training
* python train.py

### Output
* The proposed anwser is outputed to the directory ``./Answers``. 

### Test
* In the directory ``./SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2``
* perl semeval2010_task8_scorer-v1.2.pl ../../Answers/proposed_answer.txt TEST_KEY_19.TXT

### Reference
```
@article{Qin2016An,
  title={An empirical convolutional neural network approach for semantic relation classification},
  author={Qin, Pengda and Xu, Weiran and Guo, Jun},
  journal={Neurocomputing},
  volume={190},
  pages={1-9},
  year={2016},
}
```
