# Script to simulate the advancement of ions advancing through acceleration gaps.
# The user can choose to load a flat-top field or a real field that was simulated
# in Warp and then exported to a numpy type file. There are a host of settings
# that can be modified early in the script. These variables are usually prefixed
# with 'dsgn_' to represent the design values used to construct the simulation
# lattice. Particle diagnostics are avaialble by setting the z-locations.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy.constants as SC
import scipy.integrate as integrate
import scipy.optimize as opt
import os
import pdb
import time

import warp as wp

mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.minor.top"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.major.right"] = True
mpl.rcParams["ytick.minor.right"] = True
mpl.rcParams["figure.max_open_warning"] = 60

# different particle masses in eV
# amu in eV
amu_to_eV = wp.amu * pow(SC.c, 2) / wp.echarge
kV = 1000
keV = 1000
MHz = 1e6
cm = 1e-2
mm = 1e-3
um = 1e-6
us = 1e-6
ns = 1e-9  # nanoseconds
mA = 1e-3
uA = 1e-6
twopi = 2 * np.pi

# ------------------------------------------------------------------------------
#     Functions
# This section creates necessary functions for the script.
# ------------------------------------------------------------------------------


def getindex(mesh, value, spacing):
    """Find index in mesh for or mesh-value closest to specified value

    Function finds index corresponding closest to 'value' in 'mesh'. The spacing
    parameter should be enough for the range [value-spacing, value+spacing] to
    encompass nearby mesh-entries .

    Parameters
    ----------
    mesh : ndarray
        1D array that will be used to find entry closest to value
    value : float
        This is the number that is searched for in mesh.

    Returns
    -------
    index : int
        Index for the mesh-value closest to the desired value.
    """

    # Check if value is already in mesh
    if value in mesh:
        index = np.where(mesh == value)[0][0]

    else:
        index = np.argmin(abs(mesh - value))

    return index


def beta(E, mass, q=1, nonrel=True):
    """Velocity of a particle with energy E."""
    if nonrel:
        sign = np.sign(E)
        beta = np.sqrt(2 * abs(E) / mass)
        beta *= sign
    else:
        gamma = (E + mass) / mass
        beta = np.sqrt(1 - 1 / gamma / gamma)

    return beta


def rf_volt(t, freq=13.6 * MHz):
    return np.cos(2 * np.pi * freq * t)


def calc_dipole_deflection(voltage, energy, length=50 * mm, g=11 * mm, drift=185 * mm):
    """Calculate an ion's deflection from a dipole field"""

    coeff = voltage * length / g / energy
    deflection = coeff * (length / 2 + drift)

    return deflection


def calc_Hamiltonian(W_s, phi_s, W, phi, f, V, m, g=2 * mm, T=1, ret_coeffs=False):
    """Calculate the Hamiltonian.

    The non-linear Hamiltonian is dependent on the energy E, RF-frequency f,
    gap voltage V, gap width g, ion mass m and transit time factor T.

    Parameters
    ----------
    W_s : float
        Synchronous kinetic energy
    phi_s : float
        Synchronous phase
    W : float or array
        Kinetic energy for particles.
    phi : float or array
        phase of particles
    f : float
        Frequency of RF gaps.
    V : float
        Voltage applied on RF gaps.
    g : float
        Width of acceleration gap.
    m : float
        Ion mass.
    T : float
        Transit time factor.
    ret_coeffs : bool
        If True, the coeffcients A and B will be returned.

    Returns
    -------
    H : float or array
        Hamiltonian values.

    """

    bs = beta(W_s, mass=m)
    hrf = SC.c / f
    A = twopi / hrf / pow(bs, 3)
    B = V * T / g / m

    term1 = 0.5 * A * pow((W - W_s) / m, 2)
    term2 = B * (np.sin(phi) - phi * np.cos(phi_s))
    H = term1 + term2

    if ret_coeffs:
        return H, (A, B)

    else:
        return H


def calc_root_H(phi, phi_s=-np.pi / 2):
    """"Function for finding the 0-root for the Hamiltonian.

    The non-linear Hamiltonian has roots at phi=phi_s and phi_2.

    """
    term1 = np.sin(phi) + np.sin(phi_s)
    term2 = -np.cos(phi_s) * (phi - phi_s)

    return term1 + term2


def calc_emittance(x, y):
    """Calculate the RMS emittance for x,y"""
    term1 = np.mean(pow(x, 2)) * np.mean(pow(y, 2))
    term2 = pow(np.mean(x * y), 2)
    emit = np.sqrt(term1 - term2)

    return emit


def uniform_elliptical_load(Np, xmax, ymax, xc=0, yc=0, seed=42):
    """Create Np particles within ellipse with axes xmax and ymax


    Parameters:
    -----------
    Np : Int
        Number of sample points

    xmax : float
        Half-width of x-axis

    ymax : float
        Half-width of y-axis

    xc: float
        Center of ellipse in x

    yc: float
        Center of ellipse in y

    """

    keep_x = np.zeros(Np)
    keep_y = np.zeros(Np)
    kept = 0
    np.random.seed(seed)
    while kept < Np:
        x = np.random.uniform(xc - xmax, xc + xmax)
        y = np.random.uniform(yc - ymax, yc + ymax)
        coord = np.sqrt(pow((x - xc) / xmax, 2) + pow((y - yc) / ymax, 2))
        if coord <= 1:
            keep_x[kept] = x
            keep_y[kept] = y
            kept += 1

    return (keep_x, keep_y)


def uniform_box_load(Np, xlims, ylims, xc=0, yc=0, seed=42):
    """Create Np particles within ellipse with axes xmax and ymax


    Parameters:
    -----------
    Np : Int
        Number of sample points

    xlims : tuple
        Tuple containing (xmin, xmax)

    ylims : tuple
        Tuple containing (ymin, ymax)

    xc: float
        Center of box in x

    yc: float
        Center of box in y

    """
    xmin, xmax = xlims
    ymin, ymax = ylims

    x_coords = np.zeros(Np)
    y_coords = np.zeros(Np)
    np.random.seed(seed)
    for i in range(Np):
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        x_coords[i] = x
        y_coords[i] = y

    return (x_coords, y_coords)


# ------------------------------------------------------------------------------
#     Simulation Parameters
# This section is dedicated to naming and initializing design parameters that are
# to be used in the script. These names are maintained throughout the script and
# thus, if varied here are varied everywhere. In addition, some useful values
# such as the average DC electric field and initial RF wavelength are computed.
# ------------------------------------------------------------------------------
st = time.time()
# Designate particle type
ion = wp.Species(type=wp.Argon, charge_state=+1)
mass = ion.mass * pow(SC.c, 2) / wp.echarge

# Simulation Parameters for design particle
dsgn_phase = -np.pi / 2
dsgn_initE = 7 * kV
Np = int(1e5)

# Simulation parameters for gaps and geometries
Ng = 2
gap_width = 2.0 * mm
dsgn_gap_volt = 5.0 * kV
real_gap_volt = dsgn_gap_volt
dsgn_freq = 13.6 * MHz
real_freq = dsgn_freq
E_DC = real_gap_volt / gap_width
h_rf = SC.c / dsgn_freq
T_rf = 1.0 / dsgn_freq

# Energy analyzer parameters
Fcup_dist = 10 * mm
dist_to_dipole = 25.0 * mm * 0
dipole_length = 50.0 * mm
dipole_gap_width = 11.0 * mm
dist_to_slit = 185.0 * mm
slit_width = 1.0 * mm
slit_center = 37 * mm

# Compute useful values
E_DC = real_gap_volt / gap_width
dsgn_omega = twopi * dsgn_freq

# Fractions to mask particles in order to create a bucket for analysis.
alpha_E = 0.00
alpha_t = 0.15 * T_rf

# ------------------------------------------------------------------------------
#     Logical Flags
# Flags for different routines in the script.
# ------------------------------------------------------------------------------
l_use_flattop_field = False
plots_filename = "all-diagnostics.pdf"

# ------------------------------------------------------------------------------
#     Gap Centers
# Here, the gaps are initialized using the design values listed. The first gap
# is started fully negative to ensure the most compact structure. The design
# particle is initialized at z=0 t=0 so that it arrives at the first gap at the
# synchronous phase while the field is rising.
# ------------------------------------------------------------------------------
# Calculate additional gap centers if applicable. Here the design values should
# be used. The first gap is always placed such that the design particle will
# arrive at the desired phase when starting at z=0 with energy W.
phi_s = np.ones(Ng) * dsgn_phase * 0
phi_s[1:] = np.linspace(-np.pi / 3, -0.0, Ng - 1) * 0

gap_dist = np.zeros(Ng)
E_s = dsgn_initE
for i in range(Ng):
    this_beta = beta(E_s, mass)
    this_cent = this_beta * SC.c / 2 / dsgn_freq
    cent_offset = (phi_s[i] - phi_s[i - 1]) * this_beta * SC.c / dsgn_freq / twopi
    if i < 1:
        gap_dist[i] = (phi_s[i] + np.pi) * this_beta * SC.c / twopi / dsgn_freq
    else:
        gap_dist[i] = this_cent + cent_offset

    dsgn_Egain = dsgn_gap_volt * np.cos(phi_s[i])
    E_s += dsgn_Egain

gap_centers = np.array(gap_dist).cumsum()
# ------------------------------------------------------------------------------
#    Mesh setup
# Here the mesh is created with some mesh design values.
# ------------------------------------------------------------------------------
# Specify a mesh resolution
mesh_res = 100 * um
zmin = 0
zmax = gap_centers[-1] + gap_width / 2 + Fcup_dist
Nz = int((zmax - zmin) / mesh_res)
z = np.linspace(zmin, zmax, Nz)
dz = z[1] - z[0]
Ez0 = np.zeros(shape=z.shape)

# ------------------------------------------------------------------------------
#    Field Load
# The locations of the gaps are found in the z-mesh and using the gap thickness
# the flat-top field is loaded onto th emesh. The first gap  is maximally negative
# and the following gaps are 180º out of phase from the previous.
# The fields can either be loaded using a flat top or the real field which must
# extracted from the script 'fit_function_to_gap_field.py' This script reads in
# a field array that is generated in the "gap_field_function" script.
# TODO:
#    - Add mesh refinement.
# ------------------------------------------------------------------------------
# Instantiate the flat-top field values in the gap regions.
if l_use_flattop_field:
    for i, cent in enumerate(gap_centers):
        if i % 2 == 0:
            field_loc = np.where(
                (z >= cent - gap_width / 2) & (z <= cent + gap_width / 2)
            )
            Ez0[field_loc] = -real_gap_volt / gap_width
        else:
            field_loc = np.where(
                (z >= cent - gap_width / 2) & (z <= cent + gap_width / 2)
            )
            Ez0[field_loc] = real_gap_volt / gap_width

# ------------------------------------------------------------------------------
#    Particle Histories and Diagnostics
# The particle arrays are created here for the particle advancement. The design
# particle is the only particle with the history saved. The rest of the particle
# arrays are created and updated at each step.
# Diagnostics are placed at each gap, three diagnostics for gap: the entrance,
# gap center, and exit are all locations where the particle energy and time is
# recorded. Lastly, there are separate diagnostics placed at the midpoint between
# successive gaps where the Hamiltonian is computed.
# TODO:
#    - Add option to select particles to track and record history.
# ------------------------------------------------------------------------------
# Create design particle arrays. The design particle is always to begin at z=zmin,
# t=0 and given the design initial energy. The design particle is assume to be
# at z=0 at t=0 and is then moved backward in time to zmin
dsgn_pos = np.zeros(Nz)
dsgn_E = np.zeros(Nz)
dsgn_time = np.zeros(Nz)

dsgn_pos[0] = z.min()
dsgn_E[0] = dsgn_initE
dsgn_time[0] = z.min() / beta(dsgn_E[0], mass) / SC.c

# Create particle arrays to store histories
parts_pos = np.zeros(Np)
parts_pos[:] = z.min()

parts_E = np.zeros(Np)
parts_E[:] = dsgn_initE

parts_time = np.zeros(Np)
# Initialize particles be distributed around the synchronous particle's phase
phi_dev_plus = np.pi - dsgn_phase
phi_dev_minus = abs(-np.pi - dsgn_phase)
E_dev = 0.0 * keV
init_phase = np.linspace(dsgn_phase - phi_dev_minus, dsgn_phase + phi_dev_plus, Np)
init_time = np.linspace(-T_rf / 2, T_rf / 2, Np)

parts_pos[:] = z.min()
parts_time[:] = init_time

# Create diagnostic locations.
zdiagnostics = [z.min()]
for loc in gap_centers:
    zdiagnostics.append(loc)
zdiagnostics.append(z.max())
zdiagnostics = np.array(zdiagnostics)

idiagnostic = np.zeros(len(zdiagnostics), dtype=int)
for i, zloc in enumerate(zdiagnostics):
    ind = np.argmin(abs(z - zloc))
    idiagnostic[i] = ind

# Initialize diagnostic arrays. Find locations on mesh and use indexes to
# data from histories.
E_sdiagnostic = np.zeros(len(zdiagnostics))
t_sdiagnostic = np.zeros(len(zdiagnostics))
Ediagnostic = np.zeros(shape=(Np, len(zdiagnostics)))
tdiagnostic = np.zeros(shape=(Np, len(zdiagnostics)))
transmission_diagnostic = np.ones(len(zdiagnostics)) * Np

# Append initial diagnostics
E_sdiagnostic[0] = dsgn_E[0]
t_sdiagnostic[0] = dsgn_time[0]
Ediagnostic[:, 0] = parts_E
tdiagnostic[:, 0] = parts_time
# ------------------------------------------------------------------------------
#    System Outputs
# Print some of the system parameters being used.
# ------------------------------------------------------------------------------
print("")
print("#----- Injected Beam")
print(f"{'Ion:':<30} {ion.type.name}")
print(f"{'Number of Particles:':<30} {Np:.0e}")
print(f"{'Injection Energy:':<30} {dsgn_E[0]/keV:.2f} [keV]")
print(f"{'Initial Energy Spread:':<30} {E_dev/keV:.3f} [keV]")
print(f"{'Injected Average Current Iavg:':<30} {Iavg/mA:.4e} [mA]")
print(f"{'Predicted Final Design Energy:':<30} {E_s/keV:.2f} [keV]")

print("#----- Acceleration Lattice")
print(f"{'Number of Gaps':<30} {int(Ng)}")
print(f"{'Fcup Distance (from final plate):':<30} {Fcup_dist/mm:.2f} [mm]")
print(f"{'Gap Centers:':<30} {np.array2string(gap_centers/cm, precision=4)} [cm]")
print(
    f"{'Gap Distances:':<30} {np.array2string(np.diff(gap_centers/cm), precision=4)} [cm]"
)
print(f"{'System Length:':<30} {z.max()/cm:.3f} [cm]")
print(f"{'Gap Voltage:':<30} {dsgn_gap_volt/kV:.2f} [kV]")
print(f"{'Gap Width:':<30} {gap_width/mm:.2f} [mm]")
print(f"{'RF Frequency:':<30} {dsgn_freq/MHz:.2f} [MHz]")
print(f"{'RF Wavelength:':<30} {SC.c/dsgn_freq:.2f} [m]")
print(f"{'Sync Phi:':<30} {np.array2string(phi_s*180/np.pi,precision=3)} [deg]")

print("#----- Numerical Parameters")
print(f"{'Number of grid points:':<30} {len(z)}")
print(f"{'Grid spacing:':<30} {dz:>.4e} [m]")
print(f"{'Grid spacing in gap:':<30} {z_iso[1] - z_iso[0]:.4e} [m]")
print(
    f"{'Number of gridp points in gap:':<30} {int(np.floor(gap_width/(z_iso[1]-z_iso[0])))}"
)

# ------------------------------------------------------------------------------
#    Particle Advancement
# Particles are initialized to times corresponding to z=0. The advancement is
# then done evaluated the energy gain using the field value and mesh spacing dz.
# TODO:
#    Calculate Hamiltonian and tag particles in bucket.
# ------------------------------------------------------------------------------
# Main loop to advance particles. Real parameter settings should be used here.
idiagn_count = 1
for i in range(1, len(z)):

    # Do design particle
    this_dz = z[i] - z[i - 1]
    this_vs = beta(dsgn_E[i - 1], mass) * SC.c
    this_dt = this_dz / this_vs
    dsgn_time[i] = dsgn_time[i - 1] + this_dt

    Egain = Ez0[i - 1] * rf_volt(dsgn_time[i], freq=real_freq) * this_dz

    dsgn_E[i] = dsgn_E[i - 1] + Egain
    dsgn_pos[i] = dsgn_pos[i - 1] + this_dz
    dsgn_time[i] = dsgn_time[i - 1] + this_dt

    # Do other particles
    mask = parts_E > 0
    this_v = beta(parts_E[mask], mass) * SC.c
    this_dt = this_dz / this_v
    parts_time[mask] += this_dt

    Egain = Ez0[i - 1] * rf_volt(parts_time[mask], freq=real_freq) * this_dz
    parts_E[mask] += Egain
    parts_pos[mask] += this_dz

    # Check diagnostic point
    if i == idiagnostic[idiagn_count]:
        E_sdiagnostic[idiagn_count] = dsgn_E[i]
        t_sdiagnostic[idiagn_count] = dsgn_time[i]
        Ediagnostic[:, idiagn_count] = parts_E[:]
        tdiagnostic[:, idiagn_count] = parts_time[:]
        transmission_diagnostic[idiagn_count] = np.sum(mask)

        idiagn_count += 1


# Convert nan values to 0
final_E = np.nan_to_num(parts_E[:])
final_t = np.nan_to_num(parts_time[:])

Ediagnostic = np.nan_to_num(Ediagnostic)
tdiagnostic = np.nan_to_num(tdiagnostic)

phase_sdiagnostic = twopi * dsgn_freq * t_sdiagnostic
phase_diagnostic = twopi * dsgn_freq * tdiagnostic
et = time.time()
print(f"End time: {et-st:.4f}")
