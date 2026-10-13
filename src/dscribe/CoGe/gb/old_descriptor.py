import os
import sys
import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# INPUT: Your raw structures (Read Only)
ROOT_FOLDERS = [
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_100_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_110_GBs/Interface",
    "/mnt/c/Users/calvi/OneDrive/Documents/USF/Research_Group/ML_Interface_Project/data/raw/CoGeGrainBoundaries/CoGe_111_GBs/Interface"
]

# OUTPUT: Where the descriptors will be saved (Write Only)
# Change this to where you want the "clean" data to live
OUTPUT_BASE = "/home/calvi/Research_Group/ML_Interface_Project/descriptors/descriptor_output"

def collect_jobs():
    jobs = []
    print("Scanning directories for structures...")
    
    for root_folder in ROOT_FOLDERS:
        if not os.path.exists(root_folder):
            print(f"Warning: Directory not found: {root_folder}")
            continue
            
        # Determine the GB Type (e.g., CoGe_100_GBs)
        # Assuming structure: .../CoGe_100_GBs/Interface/...
        # We want the output to be .../descriptor_output/CoGe_100_GBs/...
        # So we go up two levels from 'Interface' to get the GB Type folder name
        parent_dir = os.path.dirname(root_folder) 
        gb_type_name = os.path.basename(parent_dir)

        for dirpath, dirnames, filenames in os.walk(root_folder):
            dirnames[:] = [d for d in dirnames if 'relax' not in d.lower()]
            
            target_files = [f for f in filenames if "POSCAR" in f or f.endswith(".vasp")]
            
            for filename in target_files:
                poscar_path = os.path.join(dirpath, filename)
                
                # Construct relative path from the root folder (e.g. sym00001/4x)
                rel_path = os.path.relpath(dirpath, root_folder)
                
                # Construct clean output path
                # Output: OUTPUT_BASE / GB_Type / sym00001 / 4x
                output_dir = os.path.join(OUTPUT_BASE, gb_type_name, rel_path)
                
                jobs.append((poscar_path, output_dir))
                
    return jobs

def process_structure(job):
    poscar_path, output_dir = job
    
    try:
        # Create destination folder
        os.makedirs(output_dir, exist_ok=True)

        if os.path.getsize(poscar_path) == 0:
            return f"Skipped (Empty File): {poscar_path}"

        try:
            structure = read(poscar_path)
        except Exception as e:
            return f"Read Error: {poscar_path} -> {e}"

        # Setup SOAP
        species = ["Co", "Ge"]
        soap = SOAP(
            species=species,
            r_cut=6.0, 
            n_max=12, 
            l_max=12, 
            sigma=0.3,
            weighting={"function": "pow", "r0": 5.0, "m": 6, "c": 1, "d": 1},
            periodic=True,
            sparse=False
        )
        
        # Generate raw 2D array
        raw_descriptor = soap.create(structure, n_jobs=1)
        
        # THE FIX: Compress the array using Max-Pooling BEFORE saving
        pooled_descriptor = np.max(raw_descriptor, axis=0)
        
        # Save to the SEPARATE location
        output_file = os.path.join(output_dir, "initial_soap_descriptor_run.npy")
        np.save(output_file, pooled_descriptor)
        
        return None 

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    jobs = collect_jobs()
    total_jobs = len(jobs)
    
    if total_jobs == 0:
        print("No structures found! Check your paths.")
        sys.exit()
        
    print(f"Found {total_jobs} structures. Saving to: {OUTPUT_BASE}")

    n_cores = max(1, cpu_count() - 2)
    print(f"Starting processing on {n_cores} cores...")
    
    with Pool(n_cores) as pool:
        results = list(tqdm(pool.imap(process_structure, jobs), total=total_jobs))

    errors = [r for r in results if r is not None]
    
    print("\n" + "="*40)
    print("PROCESSING COMPLETE")
    print("="*40)
    print(f"Successfully processed: {total_jobs - len(errors)} / {total_jobs}")
    
    if errors:
        print(f"Errors encountered: {len(errors)}")
        for err in errors[:5]:
            print(err)
            
    print("="*40)
