if __package__:
    from .paths import configure_plot
else:
    from paths import configure_plot

paths = configure_plot(orientation="100", structure="sym00000",
                       category="non-soc", required=('reference', 'bandstructure'))


# Compatibility shim: pyvista >= 0.45 renamed the public NORMALS constant to the
# private _NORMALS, which pyprocar 6.x still imports at package init (for its 3D
# Fermi plotting, unused here). Restore the alias before importing pyprocar.
import pyvista.core.utilities as _pv_util
if not hasattr(_pv_util, "NORMALS"):
    from pyvista.core.utilities.helpers import _NORMALS as _pv_normals
    _pv_util.NORMALS = _pv_normals

import pyprocar
from pymatgen.io.vasp.outputs import Vasprun

output_dir = paths.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Use the completed campaign DOS reference (SCF for direct-stage data).
# efermi lives inside the <dos> block, so parse_dos must be True.
efermi = Vasprun(str(paths.stage("reference") / "vasprun.xml"),
                 parse_dos=True, parse_eigen=False,
                 parse_potcar_file=False).efermi
print("band-reference E-fermi =", efermi)

pyprocar.bandsplot(
    code="vasp",
    dirname=str(paths.stage("bandstructure")),
    mode="plain",
    fermi=efermi,
    elimit=[-3, 3],
    colorbar_title_size=16,
    colorbar_tick_labelsize=12,
    savefig=str(output_dir / "bands_nosoc.png"),
)
print("wrote", output_dir / "bands_nosoc.png")
