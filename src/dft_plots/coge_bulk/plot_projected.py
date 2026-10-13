if __package__:
    from .paths import configure_plot
else:
    from paths import configure_plot

paths = configure_plot(orientation="100", structure="sym00000",
                       category="non-soc", required=('bandstructure', 'reference'))

import pyvista.core.utilities as _pv_util
if not hasattr(_pv_util, "NORMALS"):
    from pyvista.core.utilities.helpers import _NORMALS as _pv_normals
    _pv_util.NORMALS = _pv_normals

import pyprocar
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun

BS = str(paths.stage("bandstructure"))
output_dir = paths.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# efermi lives inside the <dos> block of vasprun.xml, so parse_dos must be True
# or pymatgen returns efermi=None and the bands are not shifted to E_F.
efermi = Vasprun(str(paths.stage("reference") / "vasprun.xml"),
                 parse_dos=True, parse_eigen=False,
                 parse_potcar_file=False).efermi

structure = Structure.from_file(paths.stage("bandstructure") / "POSCAR")
projections = [
    (f"{element}_{orbital}",
     [i for i, site in enumerate(structure) if site.specie.symbol == element],
     indices)
    for element in ("Co", "Ge")
    for orbital, indices in (("s", [0]), ("p", [1, 2, 3]), ("d", [4, 5, 6, 7, 8]))
]

for label, atoms, orbitals in projections:
    pyprocar.bandsplot(
        code="vasp", dirname=BS, mode="parametric",
        fermi=efermi, atoms=atoms, orbitals=orbitals,
        cmap="jet", elimit=[-3, 3],
        title=label.replace("_", " "),
        savefig=str(output_dir / f"proj_{label}.png"), show=False,
    )
    print("wrote", output_dir / f"proj_{label}.png")
