Updated on 2023-08-23 by Duc Nguyen - Colby College '24

# AlphaFold2 on Dartmouth HPC cluster

### **Overview**
    
To install [AlphaFold2](https://github.com/deepmind/alphafold) on Dartmouth HPC, you needs a Singularity image of the program, all necessary databases, and the Python script that compiles the command line.

#### *Image file*
    
- Docker-based program requires `sudo` privilege to be executed, which is not available on HPC. Therefore, we need to build a Singularity image of AlphaFold from an existing Docker image. Use the command below:
    
    `singularity build <image_name>.sif docker://<account_name>/<program_name>:<version>`
        
    To get the latest release, set `<version>` to `latest`. For instance, 
    ```singularity build alphafold_232.sif docker://catgumag/alphafold:2.3.2```
    
- The `.sif` file our lab are using corresponds with **AlphaFold 2.3.2**. To view the full documentation for this version of AlphaFold, run 
    
    ```
    singularity exec /dartfs/rc/lab/H/HowellA/AlphaFold/alphafold_232.sif python /app/alphafold/run_alphafold.py --helpfull
    ```
    
- You can find and build other versions of AlphaFold available on this [DockerHub](https://hub.docker.com/r/catgumag/alphafold).
        
#### *Databases*
    
- The databases is available in `/dartfs/rc/nosnapshots/A/Appdata/AlphaFoldDB`

- AlphaFold requires:
    + *BFD*
    + *Mgnify*
    + *UniRef90*
    + *UniRef30*/*UniClust30*
    + *UniProt*: only for AlphaFold-Multimer
    + *PDB* (in mmCIF format)
    + *PDB seqres*: only for AlphaFold-Multimer
    + *PDB70*

    Model parameters were downloaded into `params` folder, which includes:
    + 5 models which were used during CASP14, and were extensively validated for structure prediction quality 
    + 5 pTM models, which were fine-tuned to produce pTM (predicted TM-score) and (PAE) predicted aligned error values alongside their structure predictions
    + 5 AlphaFold-Multimer models that produce pTM and PAE values alongside their structure predictions.

- In case you want to install more recent versions of these databases and parameters, contact Dartmouth ITC.
    
    #### **Hotfix for AlphaFold-Multimer parameters**
    
    The current parameters correspond with AlphaFold 2.2, <br> which is incompatible to AlphaFold 2.3. Therefore, we cannot run AlphaFold-Multimer using the existing shared resources. Right now we are asking for an update of these databases and parameters.
    
    If you want to run AlphaFold with the most recent parameters while waiting for the official update, follow the guide below.
    
    -  First, move to your folder of design and download the newest AlphaFold parameters.
        ```
        mkdir alphafold-params
        cd alphafold-params/
        mkdir params
        wget -P ./params https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
        tar --extract --verbose --file=./params/alphafold_params_2022-12-06.tar --directory=./params --preserve-permissions
        rm ./params/alphafold_params_2022-12-06.tar
        ```

        Your folder should contain *multimer_v3.npz* files as below:

        ```
        alphafold-params/params/
        |-- params_model_1_multimer_v3.npz
        |-- params_model_2_multimer_v3.npz
        |-- params_model_3_multimer_v3.npz
        |-- params_model_4_multimer_v3.npz
        |-- params_model_5_multimer_v3.npz
        |-- params_model_1.npz
        |-- params_model_1_ptm.npz
        |-- params_model_2.npz
        |-- params_model_2_ptm.npz
        |-- params_model_3.npz
        |-- params_model_3_ptm.npz
        |-- params_model_4.npz
        |-- params_model_4_ptm.npz
        |-- params_model_5.npz
        |-- params_model_5_ptm.npz
        `-- LICENSE
        ```

    - Convert the Singularity image file into a writable directory (called a sandbox)

        ```singularity build --sandbox alphafold_232/ alphafold_232.sif```

    - In `/alphafold_232/app/alphafold/run_alphafold.py` (around line 419), set the directory containing the downloaded parameters to the argument of `get_model_haiku_params()` function

        ![](https://hackmd.io/_uploads/Hk2HhizT3.png)

    - Rebuild another Singularity image file from the sandbox

        ```singularity build alphafold_232_multimer_v3.sif alphafold_232/```

    - Run AlphaFold-multimer with `--singularity_image_path` in `run_alphafold_singularity_gpu.py` pointing to this new `.sif` file
    
    #### **Use custom databases**
    
    In case you want to download more recent version of aforementioned databases, please use scripts available at `/script` in the official [AlphaFold Github](https://github.com/deepmind/alphafold).
    
    - Before running these scripts, you need to install [`aria2c`](https://github.com/aria2/aria2), an ultra fast tool for downloading large files, into your personal directory.
    
        ```
        cd $HOME
        wget https://github.com/aria2/aria2/releases/download/release-1.36.0/aria2-1.36.0.tar.gz
        tar xvzf aria2-1.36.0.tar.gz
        cd aria2-1.36.0
        module load gcc
        ./configure --prefix=$HOME/.local
        make
        make install
        export PATH="$HOME/.local/bin:$PATH"
        which aria2c
        ```
        
        In order to load `aria2c` every time you log in Discovery, write the command ```export PATH="$HOME/.local/bin:$PATH"``` into `~/bash_profile` file.
        
    - Check libraries built with `aria2c`. 
        
        `aria2c -v`
        
        In order to download AlphaFold databases in HTTPS, `aria2c` needs to be built with either `GnuTLS` or `OpenSSL` as below:
        
        ![](https://hackmd.io/_uploads/HyehGJEph.png) 
        
    - Follow the desription in [AlphaFold Github](https://github.com/deepmind/alphafold). The [BFD](https://bfd.mmseqs.com/) database only has one version, and it is available in `/AlphaFoldDB`, so you don't need to update it. Downloading all other databases consumes around 800 GB. Make sure you have enough storage to host it.
    
#### *Python scripts*
    
- You need two scripts: `run_alphafold_singularity_gpu.py` and `visualize_alphafold_results.py`. The former runs AlphaFold, the later visualizes results. Both of them are available [here](https://github.com/DartmouthAlphaFold/AlphaFold).

- Copy these files to your personal/lab folder and modify them as desired.

***

### **Set up**
    
#### *`conda` environment*
    
In fact, you can run AlphaFold2 without building a `conda` environment. However, the later visualization of AlphaFold requires `JAX` library, which is not available in the universal `python3` library on Discovery & Polaris. Hence, personal build of Python environment is recommended. Besides, you are able to maintain the same set of Python libraries for consistency when you run a program multiple times.
    
- Enable the `conda` command
    
    `source /optnfs/common/miniconda3/etc/profile.d/conda.sh`
    You should put this command in the `~/.bashrc` file as well, so `conda` is enabled automatically every time you login.

- Create folder in your personal directory to store `conda` environments (only need to run this once)
    
    ```
    cd ~
    mkdir -p .conda/pkgs/cache .conda/envs
    ```
    
- Create `conda` environment with customized library. Here we use Python 3.8, and the lastest versions of `numpy`, `pandas` and `matplotlib`. You can switch to more recent version of Python. We suggest Python version $\geq$ 3.8

    `conda create --name alphafold-python python=3.8 numpy pandas matplotlib`

    If a question pops up, type `y` to continue.

- Activate the environment

    `conda activate alphafold-python`

    You should see the name of the `conda` environment appears prior to your path

    ![](https://i.imgur.com/ZYgEzWZ.png)


- Install `JAX` library

    `conda install -c conda-forge jax`

    If the above command doesn't work, try other commands [here](https://anaconda.org/conda-forge/jax).

    After that, type `python3`, then `import jax` to confirm whether the `JAX` library is installed successfully. 

- Deactivate the environment

    `conda deactivate`

    Find out more about `conda` on Dartmouth HPC [here](https://services.dartmouth.edu/TDClient/1806/Portal/KB/ArticleDet?ID=72888).
    
#### *Input files*
    
- Should be `.fasta` format (`.fasta` or `.fa` or `.faa`)

- The amino acid sequences should contain all **capital** letters. Otherwise, the program will cause error when it tries reading the input. If you are not sure whether all letters are uppercase, run `tr a-z A-Z < raw_input_file > capitalized_input_file` to get a new input file in which all characters are uppercase.

- The amino acid sequences **should not contain any stop codon ($*$) sign**. Otherwise, it will cause error at HHsearch step.

#### *AlphaFold arguments*
    
- Common arguments are:

    + `--fasta_paths`: Paths to FASTA files, each containing a prediction target that will be folded one after another. If a FASTA file contains multiple sequences, then it will be folded as a multimer. Paths should be separated by commas. All FASTA paths must have a unique basename as the basename is used to name the output directories for each prediction.

    + `--output_dir`: Path to a directory that will store the results

    + `--max_template_date`: Maximum template release date on PDB to consider (format YYYY-MM-DD). If you refold a protein that is already available on PDB, set `max_template_date` to a time before the protein's release date.

    + `--use_gpu`: Enable NVIDIA runtime to run with GPUs. `True` by default. On Polaris, it must be set to `False`.

    + `--models_to_relax`: The models to run the final relaxation step on. If `all`, all models are relaxed, which may be time consuming. If `best`, only the most confident model is relaxed. If `none`, relaxation is not run. Turning off relaxation might result in predictions with distracting stereochemical violations but might help in case you are having issues with the relaxation stage. `best` by default.

    + `--use_gpu_relax`: Whether to perform AMBER relaxation with OpenMM energy minimization using GPU. Turn off this feature if relaxation on GPU doesn't work. `True` by default.

    + `--model_preset`: Choose preset model configuration - the monomer model (`monomer`), the monomer model with extra ensembling (`monomer_casp14`), monomer model with pTM head and PAE results (`monomer_ptm`), or multimer model (`multimer`). By default, the model is `monomer_ptm`.

    + `--singularity_image_path`: Path to the AlphaFold Singularity image.

    Arguments below don't need to be modified most of time.

    + `--db_preset`: Choose preset MSA database configuration - smaller genetic database (`reduced_dbs`) or full genetic database (`full_dbs`). `full_dbs` is set by default because we don't have the small BFD database on Dartmouth HPC to run `reduce_dbs`.

    + `--data_dir`: Path to directory with supporting data, including AlphaFold parameters and genetic and template databases. By default, it is `/dartfs/rc/nosnapshots/A/Appdata/AlphaFoldDB` 

    + `--num_multimer_predictions_per_model`: How many predictions (each with a different random seed) will be generated per model. For example, if this is 2 and there are 5 models then there will be 10 predictions per input. This flag ONLY applies if `--model_preset=multimer`. By default, it is 1.

    + `--benchmark`: Run multiple JAX model evaluations to obtain a timing that excludes the compilation time, which should be more indicative of the time required for inferencing many proteins. `False` by default.

    + `--use_precomputed_msas`: Whether to read MSAs that have been written to disk. This will NOT check if the sequence, database or configuration have changed. `False` by default.

    + `--gpu_devices`: Comma separated list of GPU identifiers. Not recommended to pass this variable manually, as GPUs on Discovery are allocated via Slurm.

    You can view the help menu by typing 
    ```python3 run_alphafold_singularity_gpu.py --help```
        
***

### **AlphaFold on Polaris**

Polaris does not have GPU, so AlphaFold will use only CPUs.

- Create a `screen` session to run AlphaFold in background. Give it a name for easier tracking. If you want multiple simultaneous runs, create multiple screens.

    `screen -S <screen_name>`

- Activate the Python environment via `conda`. Remember to enable `conda` command first if you have not done it, or did not put the `source` command in `.bashrc`.

    `conda activate alphafold-python`

- Move to the folder that contains the run scripts

    `cd /dartfs/rc/lab/H/HowellA`

- Run AlphaFold: 
`python3 run_alphafold_singularity_gpu.py <arguments>`

    For example, to predict the CasX2 protein, whose crystalized structure was released on [PDB](https://www.rcsb.org/structure/7WAY) in 2022-03-16, we run 

    ```
    python3 run_alphafold_singularity_gpu.py \
        --fasta_paths /dartfs/rc/lab/H/HowellA/CasX2.faa \
        --use_gpu False --use_gpu_relax False \
        --max_template_date 2022-03-01 \
        --output_dir /dartfs/rc/lab/H/HowellA/alphafold-output/2023-03-28_CasX2
    ```

Remember to set both `use_gpu` and `use_gpu_relax` to False, since Polaris does not have GPU.

- Detach from the `screen` using `Ctrl`+`A`, then press `D`. To go back to the `screen` where AlphaFold is running, type `screen -r <screen_id>/<screen_name>`. If you don't remember the screen ID, type `screen -ls` to list all running screens.

- Be patient. The runtime of AlphaFold on Polaris depends on length of the amino acid sequence. For short and template-abundant proteins like EGFP, it takes 3.5 hours. For proteins with ~900 residues like CasX2, it takes over 30 hours. For proteins with ~1200 residues like CasY15, it takes 57 hours!

- Visualize the results:
    `python3 visualize_alphafold_results.py <output_directory>`

- Deactivate the `conda` environment:

    `conda deactivate`

- Terminate the `screen` that you ran AlphaFold on unless you want to come back and read the log.

    + Re-attach to the screen with `screen -r <screen_id>`. Type `Ctrl`+`A`, then press `K`. If a question pops up, press `y` (Windows)

    or

    + Type `screen -X -S <screen_id> quit` (Mac)

***

### **AlphaFold on Discovery**

GPU-based relaxation is unstable. It looks like some part of Amber on GPU runs in the background and does not do anything if the path is not correctly defined. If the predicted structure has no clashes (aka distracting stereochemical violations), it will finish in seconds. However, if clashes exist, it will run forever.

Therefore, we suggest running relaxation with CPU only (`--use_gpu_relax=False`). The current default setting is to relax only the best model.

- Prepare the Python scripts `run_alphafold_singularity_gpu.py` and `visualize_alphafold_results.py` in your folder.

- Make sure that the `conda` environment is ready.

- Prepare the Slurm script below.

    ```
    #!/bin/bash

    #SBATCH --job-name={job_name}
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={cpu_count}
    #SBATCH --mem={allocated_memory}
    #SBATCH --time={time_limit}
    #SBATCH --partition={partition_name}
    #SBATCH --gres=gpu:{node_type}:{node_number}
    #SBATCH --mail-user={email}
    #SBATCH --mail-type={ALL|BEGIN|END|FAIL|REQUEUE|NONE}
    #SBATCH --output={output_logging_path}
    #SBATCH --error={error_logging_path} 

    # View hostname and state of assigned GPUs
    hostname
    nvidia-smi

    # Unload all modules
    module purge

    # Enable Multi-Process Service for GPU-based relaxation
    # nvidia-cuda-mps-control -d

    # Activate conda
    source /optnfs/common/miniconda3/etc/profile.d/conda.sh
    conda activate alphafold-python

    # Set environmental variables
    export SINGULARITYENV_TF_FORCE_UNIFIED_MEMORY=1
    export SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
    export SINGULARITYENV_LD_LIBRARY_PATH=/optnfs/el7/cuda/11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

    # Output directory
    OUTPUT_DIR={output_directory}

    # Run with GPU
    python3 {your_folder}/run_alphafold_singularity_gpu.py \
        --fasta_paths {input_fasta_files} \
        --output_dir $OUTPUT_DIR

    # Visualize result
    python3 {your_folder}/visualize_alphafold_results.py $OUTPUT_DIR

    # Clean up
    conda deactivate
    # echo quit | nvidia-cuda-mps-control
    ```

    For example, below is a prediction of CasY8 structure using 1 node from the `v100_12` partition with on 1 V100 GPU and 16 CPUs, 50GB per CPU. You will receive email notification on the submitted job whenever it begins, ends, or fails with an error.

    ```
    #!/bin/bash

    #SBATCH --job-name=alphafold_2023-07-16_CasY8
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=16
    #SBATCH --mem=64G
    #SBATCH --time=120:00:00
    #SBATCH --partition=v100_12
    #SBATCH --gres=gpu:v100:1
    #SBATCH --mail-user=duc.nguyen@dartmouth.edu
    #SBATCH --mail-type=ALL
    #SBATCH --output=/dartfs/rc/lab/H/HowellA/alphafold-log/%x.%j.out     
    #SBATCH --error=/dartfs/rc/lab/H/HowellA/alphafold-log/%x.%j.err 

    # View hostname and state of assigned GPUs
    hostname
    nvidia-smi

    # Unload all modules
    module purge

    # Enable Multi-Process Service for GPU-based relaxation
    # nvidia-cuda-mps-control -d

    # Activate conda
    source /optnfs/common/miniconda3/etc/profile.d/conda.sh
    conda activate alphafold-python

    # Set environmental variables
    export SINGULARITYENV_TF_FORCE_UNIFIED_MEMORY=1
    export SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
    export SINGULARITYENV_LD_LIBRARY_PATH=/optnfs/el7/cuda/11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

    # Output directory
    OUTPUT_DIR=/dartfs/rc/lab/H/HowellA/alphafold-output/alphafold_2023-07-16_CasY8

    # Run with GPU
    python3 /dartfs/rc/lab/H/HowellA/run_alphafold_singularity_gpu.py \
        --fasta_paths /dartfs/rc/lab/H/HowellA/faa_files/CasY8.fa\
        --output_dir $OUTPUT_DIR

    # Visualize result
    python3 /dartfs/rc/lab/H/HowellA/visualize_alphafold_results.py $OUTPUT_DIR

    # Clean up
    conda deactivate
    # echo quit | nvidia-cuda-mps-control
    ```

For *Howell lab's members*, all Python scripts and Slurm script are ready in the lab folder. Most of arguments are preset in the `run_alphafold_singularity_gpu.sh` file. 

Typically, you only need to modify `--job-name`, `--mail-user`, `OUTPUT_DIR`, and the input file(s) `--fasta_paths`; save the script; then run `sbatch run_alphafold_singularity_gpu.sh`.

- Resource estimate:

    + You should run AlphaFold on Discovery using the new V100 partitions. They are much faster than the old GPU nodes in `gpuq`. To view the list of available partitions, use `squeue` command.

    + Number of nodes and number of tasks must remain 1, since AlphaFold can only run on **one GPU**. Also, one GPU V100 32G is more than enough to fold any protein shorter than 2500 residues. 

    + Number of CPUs should be at least 8 in order to maximize the speed of MSA step, since Jackhmmer is hard-coded to use 8 CPUs. Longer protein requires more memory, thus more CPUs. However, using too many CPUs will not increase the overall speed and drain off resources for other GPU-based job. We recommend 8 to 16 CPUs each run.

    + Longer proteins require more memory. We recommend running a test job first, then use `seff` command to check how much memory was used, finally pad the resource requirements by 30-50%. A 1200-residue protein typically needs 50GB. Increasing allocated memory to 64GB speeds up HHblits.

    + Logging files are not required, but we highly recommend them for debugging.

- Environmental variables:
    
    + The first two env variables allow AlphaFold to utilize more memory for very long proteins. For the above example, it can use $32 \cdot 4 = \text{32 GB from V100 GPU + 96GB from host RAM} = 128 GB$

    + The last env variable allows the Singularity container to link to the CUDA library. The CUDA version should be compatible with the AlphaFold version. Here we use CUDA 11.2 installed on Discovery (You can use other CUDA version for a different AlphaFold version. Use `module display` command to check CUDA_ROOT of your desired CUDA module).

        It is unnecessary to adjust these variables in most cases.
    
    + NVIDIA Multi-Process Service (MPS) is necessary to run relaxation on GPU. If you want to run this step on CPU (take very long time, but more stable), or don't want relaxed model at all, you do not need to enable MPS.

- Save the above Slurm script, then submit it with `sbatch` command (it is recommended to do so within your lab folder) and wait. To check the current status of the submitted job, use `squeue` command.

An earlier non-docker non-singularity AlphaFold instance is installed by Dartmouth IT in `/optnfs/el7/alphafold/alphafold`. It has its own database in `/optnfs/el7/alphafold/AlphaFoldDB`. A `conda` evironment named `alphafold` is also created in `/optnfs/el7/alphafold/conda/envs/alphafold`.  However, we are not sure how to use since it hasn't been built into a module.

***

### **Output**

 - AlphaFold will automatically create subfolders named after each protein. If the `output_dir` is `/dartfs/rc/lab/H/HowellA/alphafold-output/2023-03-28_CasX2`, then we will have a folder inside this output directory named `CasX2`. It contains:

    + `.pdb`: protein database format. These files contain the actual structures, including the pLDDT (local confidence) in the b-factor column.

    + `.pkl`: pickle data format, used in visualization, but in a binary format and thus not readable from the command line

    + `timings.json`: information on how long different processes of AlphaFold took in seconds

    + `ranking_debug.json`: information on pLDDT of each model, and how they were ranked

    + `relax_metrics.json`: AMBER relaxation parameters applied on the protein. Only available if `--run_relax` is `True`.

    + `msa` folder: contain multiple sequence alignment files that have `.sto` or `.a3m` extensions, readable with `cat` and other commands

    For example, the standard output directory for CasX2 looks like below:

    ```
    └── CasX2
        ├── features.pkl
        ├── msas
        │   ├── bfd_uniref_hits.a3m
        │   ├── mgnify_hits.sto
        │   ├── pdb_hits.hhr
        │   └── uniref90_hits.sto
        ├── ranked_0.pdb
        ├── ranked_1.pdb
        ├── ranked_2.pdb
        ├── ranked_3.pdb
        ├── ranked_4.pdb
        ├── ranking_debug.json
        ├── relaxed_model_1_ptm_pred_0.pdb
        ├── relaxed_model_2_ptm_pred_0.pdb
        ├── relaxed_model_3_ptm_pred_0.pdb
        ├── relaxed_model_4_ptm_pred_0.pdb
        ├── relaxed_model_5_ptm_pred_0.pdb
        ├── relax_metrics.json
        ├── result_model_1_ptm_pred_0.pkl
        ├── result_model_2_ptm_pred_0.pkl
        ├── result_model_3_ptm_pred_0.pkl
        ├── result_model_4_ptm_pred_0.pkl
        ├── result_model_5_ptm_pred_0.pkl
        ├── timings.json
        ├── unrelaxed_model_1_ptm_pred_0.pdb
        ├── unrelaxed_model_2_ptm_pred_0.pdb
        ├── unrelaxed_model_3_ptm_pred_0.pdb
        ├── unrelaxed_model_4_ptm_pred_0.pdb
        └── unrelaxed_model_5_ptm_pred_0.pdb
    ```

- Customized figures and data files include:
    + `coverage.png`: coverage figure by multiple sequence alignment
    + `LDDT.png` and `plddts.csv`: plot and table of predicted LDDT per residue of all 5 models
    + `PAE.png`: Predicted Alignment Error plots

    If `model_preset` is `monomer`, only the MSA coverage and pLDDT plots will be created. If `model_preset` is either `monomer_ptm` or `multimer`, we should have additional PAE plots for each model.

    For example, we predicted CasX2 with `--model_preset=monomer_ptm`, so 7 plots were created (MSA coverage, pLDDTs, and 5 PAE plots).

    ![](https://i.imgur.com/ZfsY6Gf.png)

***

### **Interpretation of results**

This [website](https://elearning.bits.vib.be/courses/alphafold/) explains the pipeline and outputs of AlphaFold very well. 

Check out videos on [USCF ChimeraX](https://www.youtube.com/@ucsfchimerax8387) Youtube channel and guides on [AlphaFoldDB](https://alphafold.ebi.ac.uk/entry/Q8W3K0). The video [below](https://www.youtube.com/watch?v=oxblwn0_PMM) is especially helpful for interpretation of PAE plots.

    
#### Question

For any other question about this note, contact me at dnnguy23@colby.edu


