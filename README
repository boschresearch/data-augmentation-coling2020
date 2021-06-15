# An Analysis of Simple Data Augmentation for Named Entity Recognition

This repository has a pytorch implementation of data augmentation for NER, introduced in our COLING 2020 paper:

> Xiang Dai and Heike Adel. 2020. An Analysis of Simple Data Augmentation for Named Entity Recognition. In COLING, Online.

Please cite this paper if you use this code. The paper can be found at the [ACL Anthology](https://www.aclweb.org/anthology/2020.coling-main.343/) or at
[ArXiv](https://arxiv.org/abs/2010.11683).

## Purpose of this Software 
This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

## Prepare the i2b2-2010 dataset
Note that the given dataset in data/ contains only sample files, showing the needed format
~~~
cp /data/dai031/Experiments/2020-06-03-01/50/* data/
~~~

## Experiments
### No augmentation
~~~
python main.py --data_folder data --embedding_type bert --pretrained_dir /data/dai031/Corpora/SciBERT/scibert_scivocab_cased --result_filepath baseline.json
~~~
### Label-wise token replacement
~~~
python main.py --data_folder data --embedding_type bert --pretrained_dir /data/dai031/Corpora/SciBERT/scibert_scivocab_cased --augmentation LwTR --result_filepath lwtr.json
~~~
### Synonym replacement
~~~
python main.py --data_folder data --embedding_type bert --pretrained_dir /data/dai031/Corpora/SciBERT/scibert_scivocab_cased --augmentation SR --result_filepath sr.json
~~~
### Mention replacement
~~~
python main.py --data_folder data --embedding_type bert --pretrained_dir /data/dai031/Corpora/SciBERT/scibert_scivocab_cased --augmentation MR --result_filepath mr.json
~~~
### Shuffle within segments
~~~
python main.py --data_folder data --embedding_type bert --pretrained_dir /data/dai031/Corpora/SciBERT/scibert_scivocab_cased --augmentation SiS --result_filepath sis.json
~~~
### All
~~~
python main.py --data_folder data --embedding_type bert --pretrained_dir /data/dai031/Corpora/SciBERT/scibert_scivocab_cased --augmentation MR LwTR SiS SR --result_filepath all.json
~~~

## Results
| Method | F1 score |
| --- | --- |
| No augmentation | 37.9 |
| Label-wise token replacement | 40.8 |
| Synonym replacement | 40.8 |
| Mention replacement | 41.2 |
| Shuffle within segments | 38.1 |
| All | 42.5 |

## License
The code in this repository is open-sourced under the Apache 2.0 license. See the
[LICENSE](LICENSE) file for details.
For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).