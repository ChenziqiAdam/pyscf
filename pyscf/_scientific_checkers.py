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
    """PS-SCF-002: Mulliken population sum and mo_occ sum both equal nelectron.

    mo_occ_sum is None for AtomSphAverageRHF's internal average-of-
    configuration atomic sub-solve (init_guess='atom'/'huckel'), whose
    mo_occ is documented to be fractional by construction -- the
    exact-integer half of this law only applies to the outer SCF object a
    user actually runs. See repo-root PROGRESS.md, PS-SCF-002 fix.
    """
    nelec = mol.nelectron
    nao = mol.nao
    tol_trace = 50.0 * eps64 * nao
    trigger_if(
        abs(pop_sum - nelec) > tol_trace, "PS-SCF-002",
        family="mulliken_trace", diff=abs(pop_sum - nelec), tol=tol_trace,
    )
    if mo_occ_sum is not None:
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
    """PS-MP2-002: t2[i,j,a,b] == t2[j,i,b,a].

    Canonical MP2's t2 is built once from the AO2MO-transformed ovov
    integral with no explicit resymmetrization step (unlike CCSD's
    iterative update_amps, which resymmetrizes every cycle and empirically
    shows zero asymmetry at any basis size -- see PS-CC-001). t2's
    asymmetry is inherited unchanged from ovov's own two-pass
    (half_e1/nr_e2) floating-point asymmetry, which grows with problem
    size; a flat eps64 tolerance under-budgets this. Re-derived from a
    7-basis H2 sweep (cc-pvdz through aug-cc-pv5z, nao 10-160): worst
    observed max|t2-t2.T|/(eps64*nao**1.5) ratio was 1.30 (aug-cc-pv5z).
    See repo-root PROGRESS.md, PS-MP2-002 fix.
    """
    diff = np.abs(t2 - t2.transpose(1, 0, 3, 2)).max()
    nao = t2.shape[0] + t2.shape[2]  # nocc + nvir == nao for canonical MP2
    tol = 50.0 * eps64 * nao ** 1.5
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

# PS-UHF-001 (<S^2> vs. nominal spin state) shipped, was fixed once after
# round-1 triggerability (added an alpha/beta occupied-subspace alignment
# gate alongside the original |ss-expected|<1.0 band), then REMOVED entirely
# after round-2 triggerability found the two-independent-threshold design is
# structurally unsound: gating a continuous quantity (the alarm) behind two
# separately-calibrated thresholds on two independently-varying continuous
# quantities (|ss-expected|<1.0 and min_sval>0.9) necessarily leaves a
# narrow input window where one threshold crosses before the other, no
# matter how the two constants are tuned (confirmed on H2 at 1.24-1.26
# Angstrom bond length -- not a PySCF defect, a design defect in the
# checker's own precondition-gating method). User decision (2026-09-17):
# this is a method-level disqualification, not a fixable tolerance gap --
# remove the candidate rather than attempt a third iteration. See repo-root
# PROGRESS.md.


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
    """PS-CC-001: CCSD t2[i,j,a,b] == t2[j,i,b,a] survives the DIIS round trip.

    Unlike PS-MP2-002, CCSD's update_amps explicitly resymmetrizes t2 every
    iteration, which empirically eliminates the same AO2MO-inherited
    asymmetry that forced PS-MP2-002 onto a size-scaled tolerance: the same
    7-basis H2 sweep used to re-derive PS-MP2-002 (cc-pvdz through
    aug-cc-pv5z, nao 10-160) measured exactly 0.0 CCSD t2 asymmetry at
    every basis size. The flat 50*eps64 tolerance is left as is. See
    repo-root PROGRESS.md, PS-MP2-002 fix (same entry documents this check).
    """
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


# ---------------------------------------------------------------------------
# Subsystem 14: fci/direct_spin1.py -- full configuration interaction
# ---------------------------------------------------------------------------

@_guard
def check_fci_rdm1_trace(dm1a, dm1b, neleca, nelecb, norb):
    """PS-FCI-001: FCI spin-separated RDM traces equal neleca/nelecb.

    make_rdm1s's compiled string-excitation contraction kernel never
    references the caller-supplied nelec integers when building matrix
    elements -- an independently implemented path from the values it must
    agree with (the PySCF fork's PS-SCF-002/PS-UHF-002/PS-MP2-003/PS-CC-002
    electron-count/RDM-trace family, now at the FCI tier).

    Tolerance derivation (independent re-sweep, 6 molecules: H2, LiH, BeH2,
    N2, H2O closed-shell + CH2 triplet open-shell, sto-3g): worst observed
    diff/(eps64*norb) ratio was 4.4 (N2). 50x that base gives ~11x headroom.
    """
    tr_a = np.trace(np.asarray(dm1a))
    tr_b = np.trace(np.asarray(dm1b))
    tol = 50.0 * eps64 * norb
    trigger_if(abs(tr_a - neleca) > tol, "PS-FCI-001",
               family="alpha", diff=abs(tr_a - neleca), tol=tol)
    trigger_if(abs(tr_b - nelecb) > tol, "PS-FCI-001",
               family="beta", diff=abs(tr_b - nelecb), tol=tol)


@_guard
def check_fci_energy_eigenvalue_consistency(h1e, eri, fcivec, norb, nelec, ecore, e_reported):
    """PS-FCI-002: energy() recontraction matches kernel()'s reported eigenvalue.

    kernel()'s Davidson/direct-diagonalization branches obtain e from the
    iteration's own Ritz-value/LAPACK bookkeeping and never call energy() to
    compute the number they report; energy() independently reconstructs the
    2e contraction and recontracts against the returned fcivec via a
    structurally different arithmetic path.

    Tolerance derivation (independent re-sweep, same 6-molecule set as
    PS-FCI-001): worst observed absolute diff was 2.84e-14 Hartree. Tolerance
    set to 1e-8, ~3.5e5x headroom, well inside the default conv_tol=1e-10
    residual scale (matches the original derivation's own finding closely).
    """
    from pyscf.fci import direct_spin1
    e_recontracted = direct_spin1.energy(h1e, eri, fcivec, norb, nelec) + ecore
    diff = abs(e_recontracted - e_reported)
    tol = 1e-8
    trigger_if(diff > tol, "PS-FCI-002", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 15: ci/cisd.py -- configuration interaction singles and doubles
# ---------------------------------------------------------------------------

@_guard
def check_cisd_rdm1_trace(dm1, nelectron, nmo):
    """PS-CISD-001: CISD one-particle density matrix trace == nelectron.

    _gamma1_intermediates builds doo/dov/dvo/dvv from the variational,
    c0-normalized amplitude triple (c0,c1,c2) -- structurally different
    arithmetic from CCSD's non-variational t1/t2+l1/l2 combination
    (PS-CC-002) and MP2's t2-only construction (PS-MP2-003), even though all
    three feed the same downstream ccsd_rdm._make_rdm1 assembly function.

    Tolerance derivation (independent re-sweep, 4 molecules: H2, LiH, HF,
    N2, sto-3g): worst observed diff/(eps64*nmo) ratio was 1.33. 50x gives
    ~37x headroom.
    """
    tr = np.trace(dm1)
    diff = abs(tr - nelectron)
    tol = 50.0 * eps64 * nmo
    trigger_if(diff > tol, "PS-CISD-001", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 16: mcscf/casci.py, mcscf/addons.py -- CASCI/CASSCF
# ---------------------------------------------------------------------------

@_guard
def check_casci_electron_count(casdm1, nelecas):
    """PS-MCSCF-001: trace(casdm1) == sum(nelecas).

    ncore/nelecas are set once at CASCI/CASSCF construction and never
    algebraically re-derived from the active-space FCI solve; casdm1's
    trace is produced entirely by the FCI solver's own RDM machinery
    (PS-FCI-001's law, one layer up), which knows nothing about ncore.

    Tolerance derivation (independent re-sweep, 3 CASCI trials: N2/sto-3g
    CAS(6,6), H2/sto-3g CAS(2,2), BeH2/6-31g CAS(4,4)): worst observed
    diff/(eps64*ncas) ratio was 2.0. 50x gives ~25x headroom.
    """
    ncas = np.asarray(casdm1).shape[0]
    tr = np.trace(np.asarray(casdm1))
    nelecas_sum = sum(nelecas) if not isinstance(nelecas, (int, np.integer)) else nelecas
    diff = abs(tr - nelecas_sum)
    tol = 50.0 * eps64 * ncas
    trigger_if(diff > tol, "PS-MCSCF-001", diff=diff, tol=tol)


@_guard
def check_casscf_canonicalization_density_invariance(dm_before, dm_after, s, nao):
    """PS-MCSCF-002: total (core+active) density matrix unchanged by canonicalize().

    Unlike a single orbital rotation at fixed occupation (PS-SCF-005/
    PS-LO-001), cas_natorb=True canonicalization is a simultaneous,
    jointly-consistent pair of transformations: an active-space orbital
    rotation (from diagonalizing casdm1) and a CI-vector transformation
    that must use the identical rotation matrix to stay consistent.

    Tolerance derivation (independent re-sweep, 4 trials: N2/sto-3g and
    BeH2/6-31g CASCI, cas_natorb True and False): worst observed
    diff/(eps64*nao*cond(S)) ratio was 0.086 (well below 1, i.e. exact to
    within roundoff at every trial tested). 50x gives large headroom.
    """
    diff = np.abs(dm_before - dm_after).max()
    cond_s = np.linalg.cond(s)
    tol = 50.0 * eps64 * nao * cond_s
    trigger_if(diff > tol, "PS-MCSCF-002", diff=diff, tol=tol)


@_guard
def check_natural_orbital_occupation_conservation(noons, nelectron, s, scf_level):
    """PS-MCSCF-003: sum(NOONs) == nelectron; each NOON in [0,2] for SCF-level refs.

    A third, structurally distinct computational route to the electron
    count beyond the AO-metric trace (PS-SCF-002) and grid-quadrature
    (PS-DFT-001) routes -- a generalized eigendecomposition via LAPACK.

    Tolerance derivation (independent re-sweep, 3 trials: H2/sto-3g RHF,
    N2/sto-3g RHF, BeH2/6-31g MP2): worst observed sum-diff/(eps64*cond(S))
    ratio was 2.74 (N2); worst RHF-only NOON-bound violation ratio was 1.54.
    50x gives ~18x / ~32x headroom respectively.
    """
    noons = np.asarray(noons)
    cond_s = np.linalg.cond(s)
    tol = 50.0 * eps64 * cond_s
    diff = abs(noons.sum() - nelectron)
    trigger_if(diff > tol, "PS-MCSCF-003", family="sum", diff=diff, tol=tol)
    if scf_level:
        bound_violation = max(-noons.min(), noons.max() - 2.0)
        trigger_if(bound_violation > tol, "PS-MCSCF-003",
                   family="bound", violation=bound_violation, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 17: ao2mo/incore.py -- AO-to-MO two-electron integral transform
# ---------------------------------------------------------------------------

@_guard
def check_ao2mo_permutation_symmetry(eri, nmo, cond_s):
    """PS-AO2MO-001: same-orbital-set MO-transformed ERI has 8-fold symmetry.

    ao2mo.incore's compiled kernel performs two sequential half-transforms
    (half_e1 then nr_e2), not a single symmetric operation -- one layer
    upstream of the PS-MP2-002/PS-CC-001 amplitude-symmetry checks.

    Tolerance derivation (independent re-sweep, 5 basis sets, H2/N2 sto-3g
    through aug-cc-pvqz, nao 2-92): a first pass using cond(mo_coeffs) (not
    squared) as the conditioning term gave an unstable, growing ratio up to
    163 at aug-cc-pvtz -- a genuine under-derivation, not the true scaling.
    Re-derived after recognizing cond(S) == cond(mo_coeffs)**2 exactly for
    S-orthonormal RHF orbitals (C^dagger S C = I): confirmed empirically
    (cond(S)/cond(mo_coeffs)**2 within 1e-9 relative across a 4-order-of-
    magnitude range of cond(S)). Using cond(mo_coeffs)**2 as the caller
    passes to this function, the worst observed diff/(eps64*nmo**1.5*
    cond_s_equiv) ratio was stable at 0.01-0.54 across the whole sweep --
    matching the pre-existing manifest's own cond(S)-based derivation almost
    exactly. 50x gives ~93x headroom over the worst observed ratio.
    """
    eri = np.asarray(eri).reshape(nmo, nmo, nmo, nmo)
    d1 = np.abs(eri - eri.transpose(1, 0, 2, 3)).max()
    d2 = np.abs(eri - eri.transpose(0, 1, 3, 2)).max()
    d3 = np.abs(eri - eri.transpose(2, 3, 0, 1)).max()
    diff = max(d1, d2, d3)
    tol = 50.0 * eps64 * nmo ** 1.5 * cond_s
    trigger_if(diff > tol, "PS-AO2MO-001", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 18: solvent/ddcosmo.py, solvent/pcm.py -- implicit solvation
# ---------------------------------------------------------------------------

@_guard
def check_implicit_solvent_energy_sign(epcm, eps):
    """PS-SOLV-001: epcm <= 0 for a physically realizable dielectric (eps>1).

    epcm is not locally forced by the one-line f_epsilon*<psi,Xvec>
    (ddCOSMO) or 0.5*dot(q_sym,v_grids) (PCM) expression -- both inner
    products are the output of a separate linear solve over a discretized
    dielectric-boundary operator (Lmat, or K/R). The induced reaction-field
    polarization always opposes and partially cancels the solute's own
    field for a passive/dissipative dielectric response.

    Note on the two call sites' distinct sentinel branches: PCM's actual
    conductor-limit sentinel is eps==inf (confirmed by reading pcm.py's
    build(): f_epsilon uses `if epsilon != float('inf') else 1.0`); the
    eps<=0 sentinel referenced in the original candidate exists only in
    ddCOSMO's _get_vind (`if dielectric > 0: ... else: f_epsilon = 1`).
    This single shared checker excludes both correctly via `eps == inf`.
    """
    if not (eps > 1) or eps == float('inf'):
        return
    tol = 50.0 * eps64 * max(abs(epcm), 1e-8)
    trigger_if(epcm > tol, "PS-SOLV-001", epcm=epcm, tol=tol, eps=eps)


# ---------------------------------------------------------------------------
# Subsystem 19: pbc/gto/cell.py -- lattice geometry, Ewald summation
# ---------------------------------------------------------------------------

@_guard
def check_pbc_reciprocal_lattice_duality(a, b, norm_to):
    """PS-PBC-001: a_i . b_j == norm_to * delta_ij, re-derived independently.

    reciprocal_vectors() computes b via a direct matrix inverse
    (b = norm_to*inv(a.T)); this checker re-derives b from scratch via the
    cross-product/triple-product formula the docstring states, a
    structurally different numerical routine from the production code's
    LAPACK matrix inverse.

    Tolerance derivation (independent re-sweep, 3 lattices: cubic He 3A and
    5A, skewed triclinic): worst observed diff/(eps64*cond(a)) ratio was
    0.77 (triclinic; the two cubic trials were exact, ratio 0). 50x gives
    ~65x headroom.
    """
    a = np.asarray(a)
    vol = np.dot(a[0], np.cross(a[1], a[2]))
    if abs(vol) < 1e-300:
        return
    b_ref = np.empty((3, 3))
    b_ref[0] = norm_to * np.cross(a[1], a[2]) / vol
    b_ref[1] = norm_to * np.cross(a[2], a[0]) / vol
    b_ref[2] = norm_to * np.cross(a[0], a[1]) / vol
    diff = np.abs(np.asarray(b) - b_ref).max()
    cond_a = np.linalg.cond(a)
    tol = 50.0 * eps64 * cond_a
    trigger_if(diff > tol, "PS-PBC-001", diff=diff, tol=tol)


@_guard
def check_pbc_ewald_algorithm_consistency(e_direct, e_pme):
    """PS-PBC-003: direct-summation and particle-mesh Ewald energies agree.

    Two structurally different reciprocal-space routines (a direct Gv-grid
    sum vs. a B-spline/FFT convolution) computing the same lattice-sum
    energy; agreement is bounded by PME's own interpolation/grid
    discretization error, not machine precision.

    Tolerance derivation (independent re-sweep, 3 lattices: cubic He 3A and
    5A, skewed triclinic): worst observed relative diff was 3.1e-15 (all
    other trials exact, diff 0.0) -- essentially machine-precision
    agreement at this system size. Tolerance kept at 1e-6 relative
    (~3.2e11x headroom over the worst observed), deliberately NOT tightened
    to eps64-scale since PME's own discretization error is real and
    grid/interpolation-order-dependent for larger, less trivial systems.
    """
    diff = abs(e_direct - e_pme)
    tol = 1e-6 * max(abs(e_direct), 1e-8)
    trigger_if(diff > tol, "PS-PBC-003", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 20: pbc/scf/khf.py, pbc/scf/kuhf.py -- k-point SCF
# ---------------------------------------------------------------------------

@_guard
def check_pbc_krhf_electron_count(mo_occ_kpts, nelectron_total):
    """PS-PBC-004: sum over k-points of mo_occ equals cell.tot_electrons(nkpts).

    get_occ() pools the spectrum across ALL sampled k-points via a single
    sort, finds one shared Fermi level, then applies that threshold
    INDEPENDENTLY to each k-point's own spectrum -- no line re-verifies
    that summing mo_occ_kpts back up reproduces tot_electrons(nkpts).

    Tolerance derivation (independent re-sweep, 2 cells x 3 meshes: cubic
    He 3A and 5A / gth-dzv, mesh in {1x1x1, 2x1x1, 2x2x1}): exact match
    (diff 0.0) in every trial. tol=0 per SANITIZER.md 5.8 X (occupations
    are 0/2 by construction absent smearing).
    """
    total = sum(float(np.sum(mo_occ_kpts[k])) for k in range(len(mo_occ_kpts)))
    trigger_if(total != nelectron_total, "PS-PBC-004",
               total=total, nelectron_total=nelectron_total)


@_guard
def check_pbc_kuhf_electron_count(mo_occ_kpts, nocc_a, nocc_b):
    """PS-PBC-005: per-spin k-summed occupation matches (nocc_a, nocc_b).

    Exercises two composed risk surfaces neither PS-PBC-004 nor the
    molecular PS-UHF-002 exercises alone: the pooled-spectrum-then-per-k
    mechanism AND the alpha/beta bookkeeping split, together.

    Tolerance derivation (independent re-sweep, same 2 cells x 3 meshes as
    PS-PBC-004): exact match (diff 0.0) in every trial. tol=0 per
    SANITIZER.md 5.8 X.
    """
    total_a = sum(float(np.sum(mo_occ_kpts[0][k])) for k in range(len(mo_occ_kpts[0])))
    total_b = sum(float(np.sum(mo_occ_kpts[1][k])) for k in range(len(mo_occ_kpts[1])))
    trigger_if(total_a != nocc_a, "PS-PBC-005",
               family="alpha", total=total_a, nocc=nocc_a)
    trigger_if(total_b != nocc_b, "PS-PBC-005",
               family="beta", total=total_b, nocc=nocc_b)


# ---------------------------------------------------------------------------
# Subsystem 21: pbc/tools/pbc.py -- Madelung / exact-exchange divergence
# ---------------------------------------------------------------------------

@_guard
def check_pbc_madelung_correction_sign(ewald_value, aspect_ratio=None):
    """PS-PBC-006: ecell.ewald() <= 0 inside madelung(), before the -2x rescale.

    The scan candidate's originally-proposed observation point (madelung()'s
    own output) is NOT reliably negative -- the -2x convention factor flips
    the sign of the actually-well-understood physical quantity. The
    genuinely well-understood sign statement is about ecell.ewald() itself,
    observed here before the sign-flipping rescale.

    Tolerance derivation (independent re-sweep, 3 lattices x 2 meshes each
    = 6 trials: cubic He 3A/5A, skewed triclinic, mesh in {1x1x1, 2x1x1}):
    ecell.ewald() was robustly negative in every trial (-0.0956 to -0.2502).
    Tolerance is relative to the observed self-energy's own scale, matching
    the original derivation's methodology and range closely.

    PRECONDITION FIX (root-cause pass, 2026-09-23, after a round-1
    triggerability firing on a hand-built vacuum-padded quasi-1D H2-chain
    cell -- 2 H atoms, 1.4A periodic axis, 12A vacuum in the other two
    directions, KRHF/KMP2 (2,1,1) k-mesh): direct numerical investigation
    (aspect-ratio and eta/precision sweeps against the built, unmodified
    checkout) showed ecell.ewald() is NOT an unconditional 3D sign theorem.
    It is eta-/precision-independent (bit-identical across an 8-decade
    precision sweep and a 2-order-of-magnitude manual eta override, ruling
    out under-convergence) but crosses from negative to positive at a
    sharp, scale-invariant threshold in the *aspect ratio* of ecell's own
    (Nk-scaled) lattice-vector lengths: max(|a_i|)/min(|a_i|) ~= 4.164,
    reproduced identically (to 4 significant figures) across 5 independent
    periodic-length values (0.7-3.0 A) and confirmed on non-orthogonal
    triclinic cells too (compact triclinic, ratio 1.14: negative;
    elongated triclinic, ratio 14.3: positive). This matches a known,
    PySCF-collaborator-acknowledged limitation of approximating a
    genuinely 1D/2D-periodic system via 3D vacuum padding (GitHub issue
    pyscf/pyscf#2224: divergent/anomalous Coulomb-type PBC quantities for
    exactly this "large-vacuum, small-period-axis" 3D-cell construction,
    attributed by a maintainer to the neutralizing-background/G=0
    treatment) -- not a hidden numerical defect in ecell.ewald() itself.
    The diamond cell used everywhere else in the round-1 sweep has aspect
    ratio 2.0 (comfortably under the threshold), explaining why the sign
    law held for every other PBC case in that sweep. Gated on
    aspect_ratio <= 3.0 (>1.3x margin below the empirical ~4.164 crossing;
    ecell.ewald() is still -0.087, far from zero, at ratio 3.0) --
    excludes vacuum-padded quasi-low-dimensional 3D cells while covering
    ordinary, reasonably-shaped 3D crystals (aspect_ratio=None, from a
    caller that cannot cheaply compute it, is treated as in-scope to avoid
    silently widening the gate).
    """
    if aspect_ratio is not None and aspect_ratio > 3.0:
        return
    tol = 50.0 * eps64 * max(abs(ewald_value), 1e-8)
    trigger_if(ewald_value > tol, "PS-PBC-006", ewald_value=ewald_value, tol=tol,
               aspect_ratio=aspect_ratio)


# ---------------------------------------------------------------------------
# Subsystem 22: pbc/mp/kmp2.py -- k-point MP2
# ---------------------------------------------------------------------------

@_guard
def check_pbc_kmp2_energy_sign(emp2, e_hf, nkpts):
    """PS-PBC-007: KMP2 correlation energy must be non-positive at a canonical KRHF ref.

    kmp2.kernel() is fully independent code from molecular mp2.py -- its own
    triple-k-point (ki,kj,ka->kb via kconserv) summation structure, its own
    DF or exact-integral contraction path, and its own 1/nkpts energy
    normalization, with no molecular-code analog.

    Tolerance derivation (independent re-sweep, He/gth-dzv cubic cell,
    2 meshes: 2x1x1, 2x2x1, DF-integral path via mp._scf.with_df.ao2mo):
    emp2 was strictly negative in both trials (-7.19e-3, -6.86e-3).
    Tolerance set to 50*eps64*abs(e_hf)*nkpts, extending PS-MP2-001's
    derivation to a sum of nkpts k-triple-loop contributions.
    """
    tol = 50.0 * eps64 * abs(e_hf) * nkpts
    trigger_if(emp2 > tol, "PS-PBC-007", emp2=emp2, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 23: pbc/cc/kccsd_rhf.py -- k-point RCCSD
# ---------------------------------------------------------------------------

@_guard
def check_pbc_kccsd_t2_symmetry(t2, kconserv):
    """PS-PBC-008: t2[ki,kj,ka][i,j,a,b] == t2[kj,ki,kb][j,i,b,a].

    The k-point generalization of the molecular PS-MP2-002/PS-CC-001
    permutation law; verifies the law SURVIVES at update_amps()'s own
    return across the k-point DIIS vectorization -- a k-point-specific
    reshape/indexing bug (flattening the three k-point dimensions into and
    out of the DIIS vector) has no molecular-code analog.

    Tolerance derivation (independent re-sweep, He/gth-dzv, 2 meshes:
    2x1x1, 2x2x1): worst observed t2 asymmetry was exactly 0.0 in both
    trials -- the same exact-symmetrization behavior the molecular
    PS-CC-001 found (CCSD's per-cycle resymmetrization eliminates the
    AO2MO-inherited asymmetry that forces PS-MP2-002 onto a size-scaled
    tolerance). Tolerance kept at flat 50*eps64.
    """
    nkpts = t2.shape[0]
    worst = 0.0
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                lhs = t2[ki, kj, ka]
                rhs = t2[kj, ki, kb].transpose(1, 0, 3, 2)
                worst = max(worst, np.abs(lhs - rhs).max())
    tol = 50.0 * eps64
    trigger_if(worst > tol, "PS-PBC-008", diff=worst, tol=tol)


@_guard
def check_pbc_kccsd_time_reversal_reality(e_imag, time_reversal_symmetric):
    """PS-PBC-009: |e.imag| negligible for a time-reversal-symmetric k-mesh.

    energy()'s own logger.warn at abs(e.imag)>1e-4 already identifies
    e.imag as diagnostically meaningful but never records the passing case
    for audit purposes. The imaginary part is a GLOBAL, mesh-level
    cancellation property (pairing k and -k contributions across the entire
    triple loop), not locally guaranteed by any single line.

    Tolerance derivation (independent re-sweep, He/gth-dzv, 2
    time-reversal-symmetric Monkhorst-Pack meshes: 2x1x1, 2x2x1): worst
    observed |e.imag| was 2.5e-20 (noise floor), closely matching the
    original derivation's ~7e-20 at this system size. Tolerance set to
    1e-6 Hartree, tighter than the library's own unexamined 1e-4, with
    generous headroom over the observed noise floor while staying loose
    enough not to false-fire at production system sizes.
    """
    if not time_reversal_symmetric:
        return
    tol = 1e-6
    trigger_if(abs(e_imag) > tol, "PS-PBC-009", e_imag=e_imag, tol=tol)


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
