# Pre-trained Model-based Actionable Warning Identification: A Feasibility Study

## Project Structure

Folder 'code' contains all related code related to the paper. Folder 'data' contains the warning dataset.

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

