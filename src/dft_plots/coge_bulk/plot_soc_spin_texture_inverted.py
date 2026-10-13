if __package__:
    from .paths import configure_plot
else:
    from paths import configure_plot

paths = configure_plot(orientation="100", structure="sym00000",
                       category="soc", required=('soc_bandstructure', 'soc_reference'))


# Compatibility shim: pyvista >= 0.45 renamed the public NORMALS constant to the
# private _NORMALS, which pyprocar 6.x still imports at package init.
import pyvista.core.utilities as _pv_util
if not hasattr(_pv_util, "NORMALS"):
    from pyvista.core.utilities.helpers import _NORMALS as _pv_normals
    _pv_util.NORMALS = _pv_normals

import pyprocar
from pymatgen.io.vasp.outputs import Vasprun

SOC_BS = str(paths.stage("soc_bandstructure"))
output_dir = paths.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Use the matching SOC campaign DOS reference (SCF for direct-stage data).
efermi = Vasprun(str(paths.stage("soc_reference") / "vasprun.xml"),
                 parse_dos=True, parse_eigen=False,
                 parse_potcar_file=False).efermi
print("SOC band-reference E-fermi =", efermi)

# For a non-collinear (LSORBIT=.TRUE.) PROCAR, "spin" is not a discrete
# up/down channel -- there IS no such split once SOC mixes spin and orbital
# character into two-component spinors. What PyProcar reads instead is the
# expectation value of each Pauli spin operator per band/k-point:
#   spins=[0]        -> spin density magnitude
#   spins=[1,2,3]     -> Sx, Sy, Sz components respectively
# Color scale runs negative-to-positive; red/blue = spin pointing in
# opposite directions along that axis, near-white = unpolarized at that
# band/k-point. This requires LSORBIT=.TRUE. + LORBIT=11 in the INCAR that
# produced this PROCAR -- both are already set in your SOC bandstructure run.
spin_components = [
    (1, "Sx"),
    (2, "Sy"),
    (3, "Sz"),
]

for spin_index, label in spin_components:
    pyprocar.bandsplot(
        code="vasp",
        dirname=SOC_BS,
        mode="parametric",
        fermi=efermi,
        spins=[spin_index],
        cmap="coolwarm",       # diverging colormap: spin-up vs spin-down direction
        clim=[-1, 1],          # fix scale to match the non-inverted structure's plots
        elimit=[-2, 2],
        title=f"CoGe {paths.orientation} {paths.structure} SOC bands - spin {label}",
        savefig=str(output_dir / f"soc_spin_{label}.png"),
        show=False,
    )
    print("wrote", output_dir / f"soc_spin_{label}.png")
