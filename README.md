# fraud-detection

We are using the **Credit Card Fraud Detection Dataset 2023** dataset for this project.

# HOW TO RUN 

   1. Make sure to only run the files under the 'scripts' folder. 
   2. If you wish to train a neural network, run train.py. If you wish to test a neural network, run test.py
   3. When training or testing a neural network, make sure that the dataset is the one you want to use. 
         This can be changed in line 31 of train.py, and line 21 of test.py.
   4. When testing a neural network model, be sure to change the model path to the appropriate name in line 16 of test.py (models/(insert_model_name_here_no_brackets)).
         The list of model names (pre-existing and the ones you created) can be found inside the models folder
   5. For the rest of the models, training and testing is done by running rq2.py (Change the model in line 29).
   6. Feature importance is automatically calculated when you train and test every model except for SVM (Not applicable) and the Feedforward Neural Network (FFNN)
   7. To calculate feature importance for the FFNN, run nn_feature_importance.py with the appropriate dataset and model.
   8. There are two pre-existing FFNN models under models already. real was trained using the realistic dataset, noundersample was trained using the original dataset.


To contribute to this project, follow the steps below:

1. Download and Install Anaconda3 if you haven't already.

   > Find instructions here:
   > [Anaconda3 installation docs](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

   I recommend leaving all options default when clicking through the prompts.

2. Activate the _base_ conda environment in your terminal.

   For **Windows 10/11** users, run the following command in a **PowerShell** terminal:

   ```bash
   ~\Anaconda3\Scripts\activate.bat base
   ```

3. Clone this repository and cd into it.

   Open a terminal and change directory (`cd`) into some project directory. Use `git` to clone this
   repository by running the following commands:

   ```bash
   git clone https://github.com/HunterCoker/fraud-detection.git
   cd fraud-detection
   ```

4. Use the provided `environment.yml` file to create the conda environment for this project.

   Run the following command:

   ```bash
   conda env create --file=environment.yml
   ```

5. Open Visual Studio Code in the root of the repo as shown below:

   ```bash
   code .  # make sure working directory is project-dir/fraud-detection
   ```

6. Make sure Visual Studio Code is using your new environment.

   - Use the keyboard command `Ctrl+Shft+P` to open the **Command Palette**

   - Search for and select the `Python: Select Interpreter` option.

   - Search for and select the `fraud-detection` environment we just made.

Once you have completed all of the above steps, Visual Studio Code should be able to find all of the modules installed in the environment. You may have to restart Visual Studio Code to see these effects.
