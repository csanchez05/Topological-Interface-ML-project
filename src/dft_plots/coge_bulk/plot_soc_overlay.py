if __package__:
    from .paths import configure_plot
else:
    from paths import configure_plot

paths = configure_plot(orientation="100", structure="sym00001",
                       category="comparison", required=('reference', 'soc_reference', 'bandstructure', 'soc_bandstructure'))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun
from pymatgen.electronic_structure.plotter import BSPlotter

output_dir = paths.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

def reference_efermi(scf_dir):
    # efermi lives inside the <dos> block, so parse_dos must be True or
    # pymatgen returns None.
    return Vasprun(str(scf_dir / "vasprun.xml"), parse_dos=True,
                   parse_eigen=False, parse_potcar_file=False).efermi

ef_nosoc = reference_efermi(paths.stage("reference"))
ef_soc   = reference_efermi(paths.stage("soc_reference"))
print("E-fermi  non-SOC =", ef_nosoc, " SOC =", ef_soc)

bs_nosoc = BSVasprun(str(paths.stage("bandstructure") / "vasprun.xml"),
                     parse_potcar_file=False).get_band_structure(
    kpoints_filename=str(paths.stage("bandstructure") / "KPOINTS"),
    line_mode=True, efermi=ef_nosoc)

bs_soc = BSVasprun(str(paths.stage("soc_bandstructure") / "vasprun.xml"),
                   parse_potcar_file=False).get_band_structure(
    kpoints_filename=str(paths.stage("soc_bandstructure") / "KPOINTS"),
    line_mode=True, efermi=ef_soc)

plotter = BSPlotter(bs_nosoc)   # non-SOC first  -> color C0
plotter.add_bs(bs_soc)          # SOC overlaid   -> color C1
ax = plotter.get_plot(ylim=(-2, 2))
# BSPlotter hardcodes fontsize=30 for the axis labels, so override on the ax.
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.tick_params(labelsize=12)

# BSPlotter labels each line by its internal band/spin index ("Band 0 up",
# "Band 0 down", "Band 1 up"), which says nothing about SOC. Replace the legend
# with one entry per calculation. BSPlotter colors by the order band structures
# were added (0 -> non-SOC, 1 -> SOC) using matplotlib's default color cycle.
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ax.legend(handles=[
    Line2D([], [], color=colors[0], lw=1.25, label="non-SOC"),
    Line2D([], [], color=colors[1], lw=1.25, label="SOC"),
], fontsize=16, loc="upper right")

plt.savefig(output_dir / "bands_soc_vs_nosoc.png", dpi=800)
print("wrote", output_dir / "bands_soc_vs_nosoc.png")
