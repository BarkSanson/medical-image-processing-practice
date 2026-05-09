# Co-registration and Segmentation of PET/MR Brain Scans
This project implements the code for the Medical Images Processing final practice. This practice has
three different objectives:
1. Loading a dynamic PET and MR and analyze their headers. The PET must be rearranged based on these headers
2. Co-register the average of all frames of the dyanmic PET onto de space of the MR. 
3. Segment the last frame of the PET using a semi-automatic tumor segmentation model
## Inputs
The code needs the MR and dynamic PET dicoms. You can place it in a `data/` directory in the root of the project. However,
if you have the files in another path, you should modify the `DATA_PATH`, `DYNAMIC_PET_FILE` and `MR_FILE` contstants
in `main.py` accordingly.

## Results
By default, all results (GIFs and images) will be stored in `results/` in the root of the project. If you want to change
this location, you should change the `RESULTS_PATH` constant in `main.py`.

## Executing the code
This project uses uv as the Python project manager. Using it is the recommended way of installing all the dependencies 
and  executing the code. You can follow the installation steps of uv using [this tutorial](https://docs.astral.sh/uv/getting-started/installation/).  However, a requirements.txt 
file is provided for convenience in case you don't want to install uv. 

### Using uv
Once you have uv installed, you can install all dependencies by simply using
```shell
uv sync
```
in the root of the project.

To execute the project, use 
```shell
uv run main.py
```

### Using pip
Before installing the dependencies, it is recommended to create a virtual environment. You can do so by executing
```shell
python -m venv .venv
```
Make sure you are using the version of Python present in the `.python-version` file to avoid any library
conflicts. You can then install all requirements with
```shell
source .venv/bin/activate

# If you are in a Windows environment, execute the following command instead of 
# the one above
# .venv/Scripts/Activate.ps1 # PowerShell is the recommended way of activating the environment

pip install -r requirements.txt
```
Finally, run the project with
```shell
python main.py
```
