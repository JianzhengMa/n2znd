from nequip.ase import NequIPCalculator
from ase.io import read

filename = "energy.dat"
net_path = "/public/home/majianzheng/LDMRM_ML/1-Templete_Mole_ML-2/5-ML_namd/net/1.pth"
atoms_all = read('coordinate.xyz', index=":")

with open(filename, 'w') as f:
    for index, atoms in enumerate(atoms_all):
        species_name = {s: s for s in atoms.get_chemical_symbols()}
        
        calc = NequIPCalculator.from_deployed_model(
            model_path=net_path,
            set_global_options=True,
            species_to_type_name=species_name
        )
        
        atoms.set_calculator(calc=calc)
        energy = atoms.get_potential_energy() / 27.2113956555
        
        f.write(f"{index}\t{energy:.8f}\n")
