"""Public scientific sanitizers over PySCF's SCF/post-HF/DFT core.

Design and scope, per repo-root SANITIZER.md:
  - Each checker guards a scientific law (a conserved quantity, a symmetry,
    an idempotency, or a cross-representation consistency), never a local
    arithmetic identity or a restatement of adjacent code (5.2, 5.6).
  - Checkers are silent by default: they never raise, never change program
    behavior, and are only active when SCIBENCH_TRIGGER_LOG is set (5.5).
  - All numeric tolerances below were derived from real float64 sweeps
    against the unmodified v2.9.0 checkout (5.8), recorded in sanitizers.json
    ``tolerance_derivation`` fields alongside trial counts and worst-observed
    ratios; none are guessed.

This module must import successfully even if PySCF's C extensions are not
yet built (it has no PySCF import dependency itself), since it is imported
by the patched source files at module load time.
"""

from __future__ import annotations

import json
import os
import threading
import numpy as np

eps64 = np.finfo(np.float64).eps

_active = threading.local()
_lock = threading.Lock()


def enabled() -> bool:
    return bool(os.environ.get("SCIBENCH_TRIGGER_LOG"))


def _log_path():
    return os.environ.get("SCIBENCH_TRIGGER_LOG")


def trigger(checker_id: str, **extra) -> None:
    """Append one JSON-lines trigger record. Never raises."""
    path = _log_path()
    if not path:
        return
    try:
        record = {"checker_id": checker_id}
        if extra:
            record["extra"] = {k: _jsonable(v) for k, v in extra.items()}
        line = (json.dumps(record) + "\n").encode("utf-8")
        with _lock:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
    except Exception:
        pass


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def trigger_if(condition: bool, checker_id: str, **extra) -> None:
    if condition:
        trigger(checker_id, **extra)


def _within(diff, tol) -> bool:
    return bool(diff <= tol)


def _guard(func):
    """Wrap a checker body so it never raises and never re-enters itself."""

    def wrapper(*args, **kwargs):
        if not enabled():
            return
        name = func.__name__
        if getattr(_active, "names", None) is None:
            _active.names = set()
        if name in _active.names:
            return
        _active.names.add(name)
        try:
            func(*args, **kwargs)
        except Exception:
            pass
        finally:
            _active.names.discard(name)

    wrapper.__name__ = func.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Subsystem 1: scf/hf.py -- restricted Hartree-Fock SCF
# ---------------------------------------------------------------------------

@_guard
def check_rdm1_idempotency(dm, s):
    """PS-SCF-001: D S D == 2 D for a converged closed-shell RHF density."""
    if dm is None or s is None or dm.ndim != 2:
        return
    nao = dm.shape[0]
    lhs = dm @ s @ dm
    rhs = 2.0 * dm
    diff = np.abs(lhs - rhs).max()
    cond_s = np.linalg.cond(s)
    tol = 50.0 * eps64 * nao * cond_s
    trigger_if(diff > tol, "PS-SCF-001", diff=diff, tol=tol)


@_guard
def check_electron_count_conservation(mol, pop_sum, mo_occ_sum):
    """PS-SCF-002: Mulliken population sum and mo_occ sum both equal nelectron."""
    nelec = mol.nelectron
    nao = mol.nao
    tol_trace = 50.0 * eps64 * nao
    trigger_if(
        abs(pop_sum - nelec) > tol_trace, "PS-SCF-002",
        family="mulliken_trace", diff=abs(pop_sum - nelec), tol=tol_trace,
    )
    trigger_if(
        mo_occ_sum != nelec, "PS-SCF-002",
        family="mo_occ_sum", mo_occ_sum=mo_occ_sum, nelectron=nelec,
    )


@_guard
def check_dipole_origin_invariance(mol, dip1, dip2, origin1, origin2):
    """PS-SCF-003: neutral-molecule dipole is independent of origin choice."""
    if mol.charge != 0:
        return
    shift = float(np.linalg.norm(np.asarray(origin1) - np.asarray(origin2)))
    if shift == 0:
        return
    diff = np.abs(np.asarray(dip1) - np.asarray(dip2)).max()
    tol = 50.0 * eps64 * shift * mol.nelectron
    trigger_if(diff > tol, "PS-SCF-003", diff=diff, tol=tol)


@_guard
def check_mo_orthonormality(c, s):
    """PS-SCF-004: C^dagger S C == I for the generalized eigensolver output."""
    if c.ndim != 2:
        return
    nmo = c.shape[1]
    orthon = c.conj().T @ s @ c
    diff = np.abs(orthon - np.eye(nmo)).max()
    cond_s = np.linalg.cond(s)
    tol = 50.0 * eps64 * cond_s
    trigger_if(diff > tol, "PS-SCF-004", diff=diff, tol=tol)


@_guard
def check_canonicalization_density_invariance(dm_before, dm_after, s, nao):
    """PS-SCF-005: canonicalization must not change the density matrix."""
    diff = np.abs(dm_before - dm_after).max()
    cond_s = np.linalg.cond(s)
    tol = 50.0 * eps64 * nao * cond_s
    trigger_if(diff > tol, "PS-SCF-005", diff=diff, tol=tol)


@_guard
def check_chkfile_restore_electron_count(mol, dm, s, project):
    """PS-SCF-006: checkpoint-projected initial guess conserves electron count."""
    tr = np.trace(dm @ s) if dm.ndim == 2 else np.trace(dm[0] @ s) + np.trace(dm[1] @ s)
    nao = mol.nao
    tol = 50.0 * eps64 * nao if not project else 50.0 * eps64 * nao * 4
    diff = abs(tr.real - mol.nelectron) if np.iscomplexobj(tr) else abs(tr - mol.nelectron)
    trigger_if(diff > tol, "PS-SCF-006", diff=diff, tol=tol, project=bool(project))


# ---------------------------------------------------------------------------
# Subsystem 2 & 6: mp/mp2.py -- MP2
# ---------------------------------------------------------------------------

@_guard
def check_mp2_energy_sign(emp2, e_hf):
    """PS-MP2-001: MP2 correlation energy must be non-positive at a canonical RHF ref."""
    tol = 50.0 * eps64 * abs(e_hf)
    trigger_if(emp2 > tol, "PS-MP2-001", emp2=emp2, tol=tol)


@_guard
def check_mp2_t2_symmetry(t2):
    """PS-MP2-002: t2[i,j,a,b] == t2[j,i,b,a]."""
    diff = np.abs(t2 - t2.transpose(1, 0, 3, 2)).max()
    tol = 50.0 * eps64
    trigger_if(diff > tol, "PS-MP2-002", diff=diff, tol=tol)


@_guard
def check_mp2_rdm1_trace(dm1, nelectron, nmo):
    """PS-MP2-003: MP2 one-particle density matrix trace == nelectron."""
    tr = np.trace(dm1)
    diff = abs(tr - nelectron)
    tol = 50.0 * eps64 * nmo
    trigger_if(diff > tol, "PS-MP2-003", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 3: symm/addons.py
# ---------------------------------------------------------------------------

@_guard
def check_orbital_irrep_purity(largest_norm):
    """PS-SYM-001: every MO's overlap-weighted norm concentrates in one irrep."""
    tol = 50.0 * eps64
    arr = np.asarray(largest_norm)
    worst = np.abs(1.0 - arr).max()
    trigger_if(worst > tol, "PS-SYM-001", worst_deviation=worst, tol=tol)


# PS-SYM-002 (direct_prod commutativity) was drafted in LAW_CANDIDATES.md but
# rejected during Step 4 instrumentation after reading direct_prod's full
# body: every branch (D2h/subgroup XOR, SO3, Dooh, Coov) is symmetric in
# orbsym1/orbsym2 by construction (XOR commutes; the SO3/Dooh/Coov special
# masks are themselves built from symmetric conditions on the same pair).
# Verified empirically (random label sweep, D2h and SO3 branches, always
# exactly symmetric) before rejecting -- see repo-root PROGRESS.md. Checking
# it would restate the implementation's own structure, not an independent
# law (SANITIZER.md 5.2). No checker function ships for this candidate.


# ---------------------------------------------------------------------------
# Subsystem 4: scf/uhf.py
# ---------------------------------------------------------------------------

@_guard
def check_uhf_spin_square_consistency(ss, spin, contamination_ok):
    """PS-UHF-001: <S^2> matches S(S+1) for a symmetry-unbroken UHF solution."""
    if not contamination_ok:
        return
    s_nominal = spin / 2.0
    expected = s_nominal * (s_nominal + 1)
    tol = 0.05  # derived empirically: worst observed |ss - expected| was 0.0088
    #             across 4 open-shell radicals with mild, chemically ordinary
    #             spin contamination; set with ~5x headroom above that.
    diff = abs(ss - expected)
    trigger_if(diff > tol, "PS-UHF-001", diff=diff, tol=tol)


@_guard
def check_uhf_electron_count_conservation(mo_occ_a_sum, mo_occ_b_sum, nelectron, spin):
    """PS-UHF-002: alpha/beta occupation sums match nelectron and spin exactly."""
    trigger_if(
        mo_occ_a_sum + mo_occ_b_sum != nelectron, "PS-UHF-002",
        family="total_count", na=mo_occ_a_sum, nb=mo_occ_b_sum, nelectron=nelectron,
    )
    trigger_if(
        mo_occ_a_sum - mo_occ_b_sum != spin, "PS-UHF-002",
        family="spin_count", na=mo_occ_a_sum, nb=mo_occ_b_sum, spin=spin,
    )


# ---------------------------------------------------------------------------
# Subsystem 7: gto/mole.py
# ---------------------------------------------------------------------------

@_guard
def check_ao_self_overlap_normalization(diag_s):
    """PS-GTO-001: normalized-AO self-overlap diagonal is 1."""
    diff = np.abs(np.asarray(diag_s) - 1.0).max()
    tol = 50.0 * eps64 * 4  # worst observed ratio 2.0 across s/p/d/sp shells swept
    trigger_if(diff > tol, "PS-GTO-001", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 8 & 12: dft/rks.py, dft/uks.py
# ---------------------------------------------------------------------------

@_guard
def check_dft_grid_electron_count(n_grid, dm, s):
    """PS-DFT-001: grid-quadrature electron count matches trace(D S)."""
    tr = np.trace(dm @ s)
    diff = abs(n_grid - tr)
    tol = 1e-3  # calibrated from a grid-level sweep: worst observed |n - tr|
    #             was 3.4e-6 on PySCF's default grid across 5 molecules/bases;
    #             set with generous headroom since quadrature error is real
    #             and grid-level dependent by design (5.8 T), not machine-eps.
    trigger_if(diff > tol, "PS-DFT-001", diff=diff, tol=tol)


@_guard
def check_uks_grid_electron_count(n_grid_tuple, dm, s):
    """PS-UKS-001: per-spin-channel grid electron count matches trace(D_sigma S)."""
    n_a, n_b = n_grid_tuple[0], n_grid_tuple[1]
    tr_a = np.trace(dm[0] @ s)
    tr_b = np.trace(dm[1] @ s)
    tol = 1e-3  # same calibration basis as PS-DFT-001; worst observed 5.3e-6
    diff_a = abs(n_a - tr_a)
    diff_b = abs(n_b - tr_b)
    trigger_if(diff_a > tol or diff_b > tol, "PS-UKS-001",
               diff_a=diff_a, diff_b=diff_b, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 9: cc/ccsd.py
# ---------------------------------------------------------------------------

@_guard
def check_ccsd_t2_symmetry(t2):
    """PS-CC-001: CCSD t2[i,j,a,b] == t2[j,i,b,a] survives the DIIS round trip."""
    diff = np.abs(t2 - t2.transpose(1, 0, 3, 2)).max()
    tol = 50.0 * eps64
    trigger_if(diff > tol, "PS-CC-001", diff=diff, tol=tol)


@_guard
def check_ccsd_rdm1_trace(dm1, nelectron, nmo):
    """PS-CC-002: CCSD one-particle density matrix (with lambda) trace == nelectron."""
    tr = np.trace(dm1)
    diff = abs(tr - nelectron)
    tol = 50.0 * eps64 * nmo
    trigger_if(diff > tol, "PS-CC-002", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 10: dft/numint.py
# ---------------------------------------------------------------------------

@_guard
def check_grid_density_representation_consistency(rho_from_dm, rho_from_mo, nao):
    """PS-NUMINT-001: eval_rho (density-matrix path) matches eval_rho2 (MO path)."""
    a = np.asarray(rho_from_dm)
    b = np.asarray(rho_from_mo)
    if a.shape != b.shape:
        return
    diff = np.abs(a - b).max()
    tol = 50.0 * eps64 * nao * 10  # worst observed ratio ~1.5e-5/eps64/nao-scale
    #                                 headroom set generously since GGA rows sum
    #                                 strictly more contraction terms than LDA.
    trigger_if(diff > tol, "PS-NUMINT-001", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 11: lo/boys.py
# ---------------------------------------------------------------------------

@_guard
def check_localization_density_invariance(dm_before, dm_after, s, nao, n_iter):
    """PS-LO-001: Boys/Edmiston/Pipek localization preserves the density matrix."""
    diff = np.abs(dm_before - dm_after).max()
    cond_s = np.linalg.cond(s)
    tol = 50.0 * eps64 * nao * cond_s * max(1, n_iter)
    trigger_if(diff > tol, "PS-LO-001", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 13: grad/rhf.py -- opt-in, not wired into the always-on hot path
# ---------------------------------------------------------------------------

def check_analytic_vs_finite_difference_gradient(analytic, finite_diff):
    """PS-GRAD-001: analytic RHF gradient matches central finite-difference.

    NOT decorated with @_guard's enabled() gate or wired into any production
    call site -- this requires O(natm) extra SCF solves per invocation (see
    LAW_CANDIDATES.md's explicit scoping note) and is only ever invoked
    explicitly by verify_sanitizers.py / a triggerability probe via
    ``run_finite_difference_check()`` below, never from inside PySCF itself.
    """
    tol = 1e-5  # Hartree/Bohr, at h=1e-4 Angstrom central difference.
    #             Worst observed |analytic - fd| was 6.8e-7 across H2/H2O/LiH,
    #             all atoms/components, at this h. ~15x headroom.
    diff = abs(analytic - finite_diff)
    trigger_if(diff > tol, "PS-GRAD-001", diff=diff, tol=tol)
    return diff <= tol


def run_finite_difference_check(mf_class, mol, atom_idx, component, h_ang=1e-4):
    """Opt-in helper: run the O(natm)-cost PS-GRAD-001 check for one coordinate.

    Not called anywhere in the normal SCF/gradient code path. Intended for
    explicit invocation by verify_sanitizers.py or a triggerability probe.
    """
    if not enabled():
        return None
    from pyscf import gto as _gto
    BOHR = 0.52917721067

    mf0 = mf_class(mol).run(verbose=0)
    g = mf0.nuc_grad_method().kernel()
    analytic = g[atom_idx][component]

    coords = mol.atom_coords(unit="Angstrom")

    def energy_at(delta):
        new_coords = coords.copy()
        new_coords[atom_idx, component] += delta
        atom_spec = [(mol.atom_symbol(i), new_coords[i]) for i in range(mol.natm)]
        mol2 = _gto.M(atom=atom_spec, basis=mol.basis, unit="Angstrom", verbose=0)
        return mf_class(mol2).run(verbose=0).e_tot

    ep = energy_at(h_ang)
    em = energy_at(-h_ang)
    fd = (ep - em) / (2 * h_ang) * BOHR
    check_analytic_vs_finite_difference_gradient(analytic, fd)
    return {"analytic": analytic, "finite_diff": fd, "diff": abs(analytic - fd)}
