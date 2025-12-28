# N2ZND (Neural Network Zhu-Nakamura Dynamics)

N2ZND is a Zhu-Nakamura nonadiabatic molecular dynamics toolkit that couples
classical trajectories with electronic-state hopping logic. The codebase
provides reusable building blocks for integrators, thermostat control, and
interfaces to quantum chemistry backends or neural-network force fields.

## Repository layout

- `src/znmd/` - Python package that exposes the simulation drivers and helpers.
- `src/znmd/example_config.yaml` - Sample configuration you can copy to start a run.
- `tools/` - Ancillary scripts for dataset preparation, Wigner sampling, and MD post-processing.
- `example/1-scf/` - Reference OM2/MRCI (`OM2MRCI.inp`) and MRSF-TDDFT (`MRSF-TDDFT.inp`) inputs; `example/2-mlmd/` - Sample MLFF launch files (`coordinate.inp`, `momentum.inp`, `example_config.yaml`, `sub_job.slurm`).
- `environment.yml` - Conda specification (Python 3.9 + CUDA 11.3 + PyTorch stack) that installs the package in editable mode.
- `pyproject.toml` - Python packaging metadata for pip-based installs.

## Requirements

- Python **3.9+**.
- Conda (Miniconda/Anaconda) if you wish to recreate the GPU-ready environment.
- Optional NVIDIA GPU/driver combo compatible with CUDA 11.3 when running the default `environment.yml`.

## Installation

### Conda (recommended)

```bash
git clone https://github.com/JianzhengMa/n2znd.git
cd n2znd
conda env create -f environment.yml
conda activate n2znd
```

The `environment.yml` file finishes by running `pip install -e .`, so the
package is immediately available in editable mode. Update later with:

```bash
conda env update -f environment.yml --prune
```

If you only need CPU support (or are on native Windows), copy `environment.yml`,
remove CUDA/PyTorch GPU entries, and point `conda env create` to your edited file.

## Workflow (dataset -> MLFF -> dynamics)

### 1. Dataset preparation

This stage follows the Hierarchical NAMD-Driven Sampling strategy: low-cost OM2/MRCI trajectories explore reaction channels, manual screening removes unphysical paths, and only chemically relevant geometries are relabeled with high-level MRSF-TDDFT before training.

1. **Generate initial conditions.**
   - Run `tools/wigner_sampling/wigner.py` (Wigner sampling at 300 K) on a frequency file to obtain `initconds`.
   - Convert those samples into per-trajectory folders with the proper electronic-structure template by running, for example:

     ```bash
     python tools/initialization_run/set_ini_run.py -w initconds -s example/1-scf/OM2MRCI.inp -f mndo
     ```

     Use `-f gamess` or `-f mlmd` when preparing GAMESS/MRSF or MLMD inputs. This step produces `run0001/`, `run0002/`, ... each containing `coordinate.inp` and `momentum.inp`.

2. **Propagate inexpensive OM2/MRCI trajectories.**
   - Launch 600 fs OM2/MRCI ZNMD runs from each folder to explore reaction channels.
   - Inspect the last frame of each trajectory with `python tools/md_processing/traj_last.py -n NUM_RUN` and drop runs that show dissociation or broken chemistry.

3. **Subsample curated structures.**
   - Within the retained `run****` directories, down-sample configurations with:

     ```bash
     python tools/dataset_builder_gamess_mrsf/extract_many_run.py
     ```

     Adjust `START_RUN`, `END_RUN`, `NSTEP`, and `NSCF` inside the script to collect roughly 2,000 snapshots.

4. **Prepare single-point inputs for labeling.**
   - Point `tools/dataset_builder_gamess_mrsf/init_static.py` to the aggregated XYZ and to the MRSF template in `example/1-scf/`:

     ```bash
     python tools/dataset_builder_gamess_mrsf/init_static.py
     ```

     This splits the master XYZ into `./run/1`, `./run/2`, ... each containing an `input.inp` ready for high-level MRSF-TDDFT jobs.
   - Submit single-point calculations (energies + forces for S0/S1) for every generated folder.

5. **Assemble the labeled dataset.**
   - After all MRSF jobs finish, convert their outputs into an ASE `extxyz` dataset:

     ```bash
     python tools/dataset_builder_gamess_mrsf/data-gamess-mlff.py
     ```

     The script scans `run_S0/*/input.out` (adjust the directory pattern if needed) and writes `mlff_dat.extxyz`, which becomes the training set. Smaller benchmark subsets (250/500/1000 structures) can be created by random sampling from this file.

### 2. Allegro MLFF training

1. Edit a copy of `tools/allegro/base.yaml` and set `dataset_file_name` to the path of `mlff_dat.extxyz` (plus any hyperparameters you wish to change).
2. Train the network:

   ```bash
   nequip-train base.yaml --warn-unused | tee log000
   ```

   This creates a directory such as `outputs/baseline`.
3. Export the inference-ready checkpoint and rename it to the numbering scheme expected by the MD driver:

   ```bash
   nequip-deploy build --train-dir outputs/baseline --model deploy.pth
   mkdir -p nets/s0
   cp outputs/baseline/deploy.pth nets/s0/1.pth
   ```

   If you train multiple models (e.g., S0/S1 or an ensemble), place them in the same parent folder as `1.pth`, `2.pth`, ...; this is exactly what `net_num` counts.

### 3. Zhu-Nakamura dynamics

1. Copy `src/znmd/example_config.yaml` to a working file and set:
   - `program_name: MLFF`
   - `net_path` to the directory that contains numbered checkpoints (`1.pth`, `2.pth`, ... as produced above)
   - `net_num` equal to the number of `.pth` models present in `net_path`
   - Simulation details such as `init_input`, `init_momentum`, time step, thermostat, etc.
2. Run the MD driver:

   ```bash
   python -m znmd.md_main --config my_namd.yaml --set init_input=E.inp init_momentum=momentum.inp
   ```

   Use `--set KEY=VALUE` overrides for quick experiments. The driver performs Zhu-Nakamura surface hopping directly on the MLFF without explicit NAC calculations.

## Verifying the install

```bash
python -m znmd.md_main --help
```

You should see the CLI usage with options for `--config` and `--set`. Use
`pip show n2znd` or `conda list n2znd` if you need to confirm metadata.


## Development tips

- After pulling new commits, re-run `pip install -e .` (inside your environment)
  to pick up metadata or dependency changes.
- Use `python -m pytest` (or any project-specific test runner) once test suites
  are added.
- Keep environment names, package names, and documentation references aligned
  with the new project title "N2ZND".

With the environment active and configuration prepared, you are ready to explore
Neural Network Zhu-Nakamura nonadiabatic dynamics simulations. Happy hacking!

## Acknowledgments

- `tools/wigner_sampling/wigner.py` is an unmodified copy of the Wigner sampling utility shipped with **SHARC 3.0** (Mai, S.; Avagliano, D.; Heindl, M.; Marquetand, P.; Menger, M. F. S. J.; Oppel, M.; Plasser, F.; Polonius, S.; Ruckenbauer, M.; Yinan Shu; Truhlar, D. G.; Linyao Zhang; Zobel, P.; Gonzalez, L. *SHARC3.0: Surface Hopping Including Arbitrary Couplings - Program Package for Non-Adiabatic Dynamics.* 2023. https://zenodo.org/record/7828641). The script keeps the upstream GPLv3 licensing and copyright notices.
