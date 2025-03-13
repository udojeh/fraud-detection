# fraud-detection

We are using the **Credit Card Fraud Detection Dataset 2023** dataset for this project.

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
