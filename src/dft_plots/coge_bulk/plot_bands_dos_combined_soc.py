if __package__:
    from .paths import configure_plot
else:
    from paths import configure_plot

paths = configure_plot(orientation="100", structure="sym00001",
                       category="non-soc", required=('reference', 'bandstructure', 'dos'))

import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun
from pymatgen.electronic_structure.plotter import BSDOSPlotter

output_dir = paths.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Bands use the campaign's matching DOS E_F; direct-stage data use SCF E_F.
# DOS uses its own run's E_F. Never use a line-mode Fermi estimate.
# ---------------------------------------------------------------------------
def reference_efermi(scf_dir):
    return Vasprun(str(scf_dir / "vasprun.xml"), parse_dos=True,
                   parse_eigen=False, parse_potcar_file=False).efermi

ef_bands = reference_efermi(paths.stage("reference"))
print("bands E-fermi (from matching reference run) =", ef_bands)

bs = BSVasprun(str(paths.stage("bandstructure") / "vasprun.xml"),
               parse_potcar_file=False).get_band_structure(
    kpoints_filename=str(paths.stage("bandstructure") / "KPOINTS"),
    line_mode=True, efermi=ef_bands)

dosv = Vasprun(str(paths.stage("dos") / "vasprun.xml"),
               parse_dos=True, parse_eigen=False, parse_potcar_file=False)
cdos = dosv.complete_dos
print("DOS E-fermi (own run)   =", dosv.efermi)

# ---------------------------------------------------------------------------
# Combined plot: bands (left) + DOS (right), sharing the energy axis.
# bs_projection=None  -> plain bands, no orbital coloring needed here.
# dos_projection='elements' -> Co/Ge breakdown in the DOS panel (uses the
#   same complete_dos you already plotted in plot_dos.py, no extra cost).
#   Set to None instead if you just want a single black total-DOS curve.
# vb_energy_range / cb_energy_range: window below/above E_F, in eV.
#   Tightened to 3 eV each since CoGe is a metal -- no "valence/conduction"
#   split in the traditional sense, just a symmetric window around E_F.
# ---------------------------------------------------------------------------
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
