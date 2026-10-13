if __package__:
    from .paths import configure_plot
else:
    from paths import configure_plot

paths = configure_plot(orientation="110", structure="sym00000",
                       category="soc", required=('soc_reference', 'soc_bandstructure', 'soc_dos'))

import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun
from pymatgen.electronic_structure.plotter import BSDOSPlotter

output_dir = paths.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Use the matching SOC campaign DOS reference (SCF for direct-stage data).
def reference_efermi(scf_dir):
    return Vasprun(str(scf_dir / "vasprun.xml"), parse_dos=True,
                   parse_eigen=False, parse_potcar_file=False).efermi

ef = reference_efermi(paths.stage("soc_reference"))
print("SOC bands/DOS E-fermi (from matching reference run) =", ef)

bs = BSVasprun(str(paths.stage("soc_bandstructure") / "vasprun.xml"),
               parse_potcar_file=False).get_band_structure(
    kpoints_filename=str(paths.stage("soc_bandstructure") / "KPOINTS"),
    line_mode=True, efermi=ef)

dosv = Vasprun(str(paths.stage("soc_dos") / "vasprun.xml"),
               parse_dos=True, parse_eigen=False, parse_potcar_file=False)
cdos = dosv.complete_dos
print("SOC DOS E-fermi (own run)   =", dosv.efermi)

plotter = BSDOSPlotter(
    bs_projection=None,
    dos_projection="elements",
    vb_energy_range=3,
    cb_energy_range=3,
    egrid_interval=1,
)
plotter.get_plot(bs, dos=cdos)

plt.savefig(output_dir / "bands_dos_combined.png", dpi=800, bbox_inches="tight")
print("wrote", output_dir / "bands_dos_combined.png")
