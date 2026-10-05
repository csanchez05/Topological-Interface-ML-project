# src/vasp_inputs

- `scaffold_interface_tree.py`: builds `calculations/CoGe_interface/<plane>/symNNNNN/<k>x/` (POSCAR copied
  from the GB catalogue, plus empty `convergence/` and `non-soc|soc/{scf,dos,bandstructure}/` with POTCAR
  symlinks, inside every thickness folder). Folder scaffold only; writes no INCAR/KPOINTS.
  Usage: `python3 -B src/vasp_inputs/scaffold_interface_tree.py [--planes 100 ...] [--dry-run]`

The verified VASP input generator (CLAUDE.md section 7) will also live here; it does not exist yet.
