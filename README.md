# Pre-trained Model-based Actionable Warning Identification: A Feasibility Study

## Project Structure

Folder '1_validity' contains the code that explores the effectiveness of Pre-Trained model (PTM) in AWI and PTMs' performance under different data preprocessing ways (i.e., warning context extraction and abstraction). Folder '2_pretrain' and '3_finetune' explores PTMs' performance under different components in the model training. Folder '4_project' explores PTMs' performance in different model prediction scenations (i.e., within and cross AWI scenarios). Folder 'data' contains the necessary dataset required by the research questions. 

## How to run

```
cd the model and rq you want
python run.py
```

The file run.py contains the necessary commands to run the experiment. Here are the explanations of some important parameters.

```
--output_dir where the trained weights will be saved
--do_train set this if you want to train the model
--do_test set this if you want to run the inference process
--train_data_file where the train dataset is located
--epochs how many epochs do you want to train the model
--rq the prefix of the result file (to distinguish between different datasets/scenarioes)
--model_name the file name of the saved trained model weight name.
```

