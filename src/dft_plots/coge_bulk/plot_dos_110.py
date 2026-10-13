if __package__:
    from .paths import configure_plot
else:
    from paths import configure_plot

paths = configure_plot(orientation="110", structure="sym00000",
                       category="non-soc", required=('dos',))

import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import OrbitalType

output_dir = paths.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# parse_dos=True is required -- efermi and the projected DOS both live in the
# <dos> block. LORBIT=11 in the DOS run's INCAR is what makes element/orbital
# projections available at all.
vr = Vasprun(str(paths.stage("dos") / "vasprun.xml"),
             parse_dos=True, parse_eigen=False, parse_potcar_file=False)

cdos = vr.complete_dos
efermi = vr.efermi
print("DOS run E-fermi =", efermi)

energies = cdos.energies - efermi   # shift so E_F = 0 on every plot

# CompleteDos.densities sums spin channels automatically if summed manually;
# get_densities() handles ISPIN=1 or ISPIN=2 (spin-up + spin-down) the same way.
def total(dos_obj):
    return dos_obj.get_densities()

# ---------------------------------------------------------------------------
# Panel 1: total DOS + per-element DOS (Co vs Ge)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(energies, total(cdos), color="black", lw=1.5, label="Total")

element_dos = cdos.get_element_dos()   # {Element("Co"): Dos, Element("Ge"): Dos}
for el, dos_obj in element_dos.items():
    ax.plot(energies, total(dos_obj), lw=1.5, label=str(el))

ax.axvline(0, color="gray", ls="--", lw=1)
ax.set_xlim(-8, 4)
ax.set_xlabel("E − E$_F$ (eV)")
ax.set_ylabel("DOS (states/eV/cell)")
ax.set_title(f"CoGe {paths.orientation} {paths.structure} — total and element-projected DOS")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "dos_total_elements.png", dpi=300)
plt.close(fig)
print("wrote", output_dir / "dos_total_elements.png")

# ---------------------------------------------------------------------------
# Panel 3: total DOS only
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(energies, total(cdos), color="black", lw=1.5)

ax.axvline(0, color="gray", ls="--", lw=1)
ax.set_xlim(-8, 4)
ax.set_xlabel("E − E$_F$ (eV)")
ax.set_ylabel("DOS (states/eV/cell)")
ax.set_title(f"CoGe {paths.orientation} {paths.structure} — total DOS")
fig.tight_layout()
fig.savefig(output_dir / "dos_total_only.png", dpi=300)
plt.close(fig)
print("wrote", output_dir / "dos_total_only.png")

# ---------------------------------------------------------------------------
# Panel 2: orbital-projected DOS, Co and Ge on separate subplots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

for ax, el_symbol in zip(axes, ["Co", "Ge"]):
    element = [e for e in element_dos if str(e) == el_symbol][0]
    spd = cdos.get_element_spd_dos(element)   # {OrbitalType.s: Dos, .p: Dos, .d: Dos}

    for orb_type in (OrbitalType.s, OrbitalType.p, OrbitalType.d):
        if orb_type in spd:
            ax.plot(energies, total(spd[orb_type]), lw=1.5, label=f"{el_symbol}-{orb_type.name}")

    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_ylabel("DOS (states/eV/cell)")
    ax.legend()
    ax.set_title(f"{el_symbol} orbital-projected DOS")

axes[-1].set_xlabel("E − E$_F$ (eV)")
axes[-1].set_xlim(-8, 4)
fig.tight_layout()
fig.savefig(output_dir / "dos_orbitals.png", dpi=300)
plt.close(fig)
print("wrote", output_dir / "dos_orbitals.png")

# ---------------------------------------------------------------------------
# N(E_F) -- the number you quote as proof of metallicity
# ---------------------------------------------------------------------------
import numpy as np
idx = np.argmin(np.abs(energies))
n_ef = total(cdos)[idx]
print(f"N(E_F) ~ {n_ef:.3f} states/eV/cell")
