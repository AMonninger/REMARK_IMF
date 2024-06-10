# HARK meets SSJ for Policy Maker
This REMARK contains notebooks to solve HANK models using HARK and SSJ.

## To run the code within the IMF

1. From Software  install:
- python 3.10.0 
- GitBash
- VS Code (If you want)

2. Clone GitHub folder
- open a command prompt or terminal in VS code.
- Navigate to a folder you like eg 
```
cd "OneDrive - International Monetary Fund (PRD)"
```
Create a folder GitHub
```
mkdir GitHub
cd GitHub
```
(You can navigate with the command "cd" to get one folder back use "cd ." and to see what folders are in a directory use "ls")

- Execute the following commands
```
git config --global http.sslbackend schannel
git clone https://github.com/AMonninger/REMARK_IMF
```

Navigate into REMARK_IMF folder:
```
cd REMARK_IMF
```

3. Install econ-ark
```
pip install --no-dependencies --default-timeout=500 econ-ark==0.13.0 --user 
```

4. Open notebook:
```
jupyter notebook
```

5. Run a file
- go to code/python
- Open 00_Journey_3_PolicyMakers.ipynb
- run the file (with the two arrows/ fast forward button)

6. To update the Repo
- If you want to make changes on a notebook, always create a copy
- To 'pull' the latest changes from the GitHub Repo: open a terminal
- navigate to the correct folder
```
git pull
```


## If you run the code from outside the IMF
Create a new environment by following the following steps

1. Open a new terminal
2. Navigate to this folder using `cd`
2. type
```
conda env create -f environment.yml
```
