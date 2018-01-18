# Unsupervised-Neural-Machine-Translation

## Platform

Our program is tested on the following machines:

```
Ubuntu 16.04
macOS 10.13.1
```



## Dependencies

Our program requires the following programs/packages to run successfully:

```
python 3.6
numpy v1.3
pytorch v0.2
```



## Instructions for Running the Program

1. Clone the repository

   **Notice:** Please downoad the master branch. Thanks.

   ​

2. Download the dataset

Download the dataset from the following link:

https://drive.google.com/open?id=11bDQXpQOiUw5hEwtbmr1IMxDHk-uMFjF

Then unzip the downloaded file and put all the files into the folder `UNMT/data`

The folder `data	` should contain the following files

```
cross_en
cross_fr
data_en.subword.clean
data_fr.subword.clean
emb_en.dms
emb_fr.dms
test_en_100.proc
test_fr_100.proc
vocab_en.pkl
vocab_fr.pkl
```

3. Download our trained model

Download our trained model from the following link:

https://drive.google.com/open?id=1KmcwT5QkEfSTxpdPXD4SNWp-Cj5FiIbk

Then unzip the file and put `Nonee_2000.pkl` into the folder `UNMT` (the same folder as train.py, main.py...)

4. Run the training program

To run the training program, type the following command

```
python3 main.py --train
```

Note that our program only works on python 3. Specifically, we run our training and testing program on python 3.6 (for more information of packages version, please checkout the depencies section). 

The training program will save the model file into the folder `UNMT` and the log file into `save/`

5. Run the testing program

To run the testing program type the following command:

```
python3 main.py --test --model_path Nonee_2000.pkl
```

Then the program will translate each English sentence  in `test_en_100.proc` into a French sentece and save the translation into the file `l1l2.txt` (in the folder `UNMT`). The ground-truth translation is in the file `test_fr_100.proc`, which will be used later to evaluate the results of translation

To evaluate the translation results, type the following command:

```
./mosesdecoder/scripts/generic/multi-bleu.perl data/test_fr_100.proc < l1l2.txt
```

Notice that the it is normal to see BLEU score around 0.00 to 0.01.