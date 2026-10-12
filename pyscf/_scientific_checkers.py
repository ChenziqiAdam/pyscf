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

import contextlib
import json
import os
import threading
import warnings
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


def _eps_of(*arrays):
    """Unit roundoff of the coarsest floating-point array the caller supplied.

    A result computed from reduced-precision input (float32 orbitals, amplitudes
    or CI vectors) is only defined to that input's roundoff, however the library
    stores it afterwards. Integers and non-array arguments are ignored.
    """
    eps = eps64
    stack = list(arrays)
    while stack:
        a = stack.pop()
        if a is None:
            continue
        if isinstance(a, (list, tuple)):
            stack.extend(a)
            continue
        dt = getattr(a, "dtype", None)
        try:
            if dt is not None and np.issubdtype(dt, np.inexact):
                eps = max(eps, float(np.finfo(dt).eps))
        except Exception:
            pass
    return eps


def _within(diff, tol) -> bool:
    return bool(diff <= tol)


def _is_floating(*arrays) -> bool:
    """True when every array supplied is floating-point or complex.

    The laws below are statements about real-valued orbitals and densities;
    a coefficient array of integer (or bool) dtype is silently truncated by
    the library's in-place rotations and is not a scientific input.
    ``None`` entries are ignored.
    """
    for a in arrays:
        if a is None:
            continue
        dt = getattr(a, "dtype", None)
        if dt is None:
            dt = np.asarray(a).dtype
        if not np.issubdtype(dt, np.inexact):
            return False
    return True


@contextlib.contextmanager
def _quiet():
    """Checker re-calls must not leak warnings or floating-point errors to user code.

    The ignore filter is inserted into ``warnings.filters`` directly instead of
    using ``catch_warnings``: the latter invalidates every module's
    once-per-location warning registry, which would make production warnings
    repeat after each checker call.
    """
    flt = ("ignore", None, Warning, None, 0)
    warnings.filters.insert(0, flt)
    try:
        try:
            import numpy as _np
        except Exception:
            yield
        else:
            with _np.errstate(all="ignore"):
                yield
    finally:
        try:
            warnings.filters.remove(flt)
        except ValueError:
            pass


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
            with _quiet():
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

def _blocks_to_ao(symm_orb, blocks, s):
    """Map irrep blocks B_ir = O_ir^T S O_ir back to the AO metric through the
    dual basis O^{-1}; symm_orb is complete but not necessarily Euclidean-
    orthonormal (cart=True with a rotated frame)."""
    o_all = np.hstack(symm_orb)
    if o_all.shape[0] != o_all.shape[1]:
        raise ValueError("incomplete symm_orb")
    oinv = np.linalg.inv(o_all)
    bd = np.zeros((o_all.shape[1],) * 2, dtype=np.result_type(s, o_all))
    p0 = 0
    for o, b in zip(symm_orb, blocks):
        n = o.shape[1]
        bd[p0:p0 + n, p0:p0 + n] = b
        p0 += n
    return oinv.conj().T @ bd @ oinv


def symm_block_overlap(mol, s):
    """Block-symmetrised overlap of a symmetry-enabled mol; None if not applicable."""
    try:
        if not mol.symmetry or getattr(mol, "symm_orb", None) is None:
            return None
        blocks = [o.conj().T @ s @ o for o in mol.symm_orb]
        if getattr(mol, "groupname", None) in ("Dooh", "Coov"):
            for ir, iid in enumerate(mol.irrep_id):
                if iid not in (0, 1, 4, 5) and iid % 2 == 1:
                    blocks[ir] = blocks[ir - 1]
        return _blocks_to_ao(mol.symm_orb, blocks, s)
    except Exception:
        return None


def solver_overlap(mf, s):
    """Overlap matrix the SCF eigensolver diagonalised against.

    A symmetry-adapted SCF (hf_symm.eig) solves each irrep block against the
    block-symmetrised overlap, so a geometry accepted as symmetric within
    ``symm_geom_tol`` but not symmetrised has orbitals orthonormal in that
    symmetrised metric rather than in the exact overlap. Any other SCF
    returns ``s`` unchanged.
    """
    try:
        from pyscf.scf import hf_symm, uhf_symm
        mol = mf.mol
        if (isinstance(mf, (hf_symm.SymAdaptedRHF, hf_symm.SymAdaptedROHF, uhf_symm.SymAdaptedUHF))
                and mol.symmetry and getattr(mol, "symm_orb", None) is not None):
            blocks = [o.conj().T @ s @ o for o in mol.symm_orb]
            if getattr(mol, "groupname", None) in ("Dooh", "Coov"):
                # the second component of a 2D irrep reuses the first one's coefficients
                for ir, iid in enumerate(mol.irrep_id):
                    if iid not in (0, 1, 4, 5) and iid % 2 == 1:
                        blocks[ir] = blocks[ir - 1]
            return _blocks_to_ao(mol.symm_orb, blocks, s)
    except Exception:
        pass
    return s


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
    # rounding of D S D is relative to the density magnitude (|D| ~ sqrt(cond(S))
    # for nearly redundant bases), which the nao*cond(S) budget does not follow
    tol = 50.0 * eps64 * max(nao * cond_s, float(np.abs(dm).max()) ** 2)
    trigger_if(diff > tol, "PS-SCF-001", diff=diff, tol=tol)


@_guard
def check_electron_count_conservation(mol, pop_sum, mo_occ_sum, s=None, n_target=None, pop_sum_solver=None):
    """PS-SCF-002: Mulliken population sum and mo_occ sum both equal nelectron.

    mo_occ_sum is None for AtomSphAverageRHF's internal average-of-
    configuration atomic sub-solve (init_guess='atom'/'huckel'), whose
    mo_occ is documented to be fractional by construction -- the
    exact-integer half of this law only applies to the outer SCF object a
    user actually runs. See repo-root PROGRESS.md, PS-SCF-002 fix.

    ``n_target``: particle number the density is defined to carry when it is
    not mol.nelectron (smearing at a fixed chemical potential mu0 is grand
    canonical; the target is then sum(mo_occ)).
    """
    nelec = mol.nelectron if n_target is None else n_target
    nao = max(mol.nao, 0 if s is None else s.shape[0])  # model Hamiltonians carry no AO basis on mol
    # trace(D S) accumulates rounding in proportion to the conditioning of S
    cond_s = 1.0 if s is None else max(1.0, float(np.linalg.cond(s)))
    tol_trace = 50.0 * eps64 * nao * cond_s
    # symmetry-adapted SCF on a geometry accepted within symm_geom_tol: orbitals are
    # orthonormal in the block-symmetrised overlap, whose trace is the exact count
    _d = abs(pop_sum - nelec)
    if pop_sum_solver is not None:
        _d = min(_d, abs(pop_sum_solver - nelec))
    trigger_if(
        _d > tol_trace, "PS-SCF-002",
        family="mulliken_trace", diff=_d, tol=tol_trace,
    )
    if mo_occ_sum is not None:
        trigger_if(
            mo_occ_sum != nelec, "PS-SCF-002",
            family="mo_occ_sum", mo_occ_sum=mo_occ_sum, nelectron=nelec,
        )


@_guard
def check_dipole_origin_invariance(mol, dip1, dip2, origin1, origin2, dm_trace=None, rmax=None,
                                    el_scale=None):
    """PS-SCF-003: neutral-molecule dipole is independent of origin choice.

    The electronic term shifts with tr(D S), not with the nuclear charge, so
    the invariance holds for an N-electron density (tr(D S) == nelectron).
    A density that is not normalised to nelectron (e.g. the MINAO/VSAP
    initial guess, tr(D S) = 1.975 for H2) has the exact origin dependence
    (tr(D S) - nelectron) * (origin2 - origin1); that known shift is removed
    before comparing, so only a deviation beyond it is an alarm. The nuclear
    and electronic terms each carry magnitude ~ nelectron * rmax (rmax: largest
    coordinate relative to the origins) and cancel only to that scale.
    """
    if mol.charge != 0:
        return
    delta = np.asarray(origin2, dtype=float) - np.asarray(origin1, dtype=float)
    shift = float(np.linalg.norm(delta))
    if shift == 0:
        return
    change = np.asarray(dip2) - np.asarray(dip1)
    n_ref = float(mol.nelectron)
    if dm_trace is not None:
        change = change - (float(dm_trace) - n_ref) * delta
        n_ref = max(n_ref, abs(float(dm_trace)))
    diff = np.abs(change).max()
    tol = 50.0 * eps64 * n_ref * (shift + (0.0 if rmax is None else float(rmax)))
    if el_scale is not None:
        # rounding of the two electronic contractions sum_ij |r_ij||D_ij| (one per origin)
        tol = max(tol, 50.0 * eps64 * 2.0 * float(el_scale))
        # the integral kernel accumulates nprim^2 primitive pairs per shell
        # pair; its rounding grows ~eps*nprim (measured <= 4*eps*nprim*el_scale)
        try:
            nprim = max([int(mol.bas_nprim(ib)) for ib in range(mol.nbas)] or [1])
        except Exception:
            nprim = 1
        tol = max(tol, 50.0 * eps64 * nprim * float(el_scale))
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
def check_canonicalization_density_invariance(dm_before, dm_after, s, nao, mo_occ=None, src=()):
    """PS-SCF-005: canonicalization must not change the density matrix.

    hf.canonicalize diagonalises the Fock matrix inside the doubly occupied,
    the open (all occupations other than 0 and 2) and the virtual block. A
    rotation inside a block leaves D unchanged only if every orbital of the
    block has the same occupation, so a block of unequal fractional
    occupations (a thermal/ensemble state) is outside the law.
    Precondition: floating-point orbitals.
    """
    if not _is_floating(*src):
        return
    if mo_occ is not None:
        occ = np.asarray(mo_occ, dtype=float)
        open_occ = occ[(occ != 2) & (occ != 0)]
        if open_occ.size > 1 and np.ptp(open_occ) > 1e-12:
            return
    diff = np.abs(dm_before - dm_after).max()
    cond_s = np.linalg.cond(s)
    # roundoff of the supplied orbitals (src), not of the float64 density rebuilt from them
    tol = 50.0 * _eps_of(dm_before, dm_after, *src) * nao * cond_s
    trigger_if(diff > tol, "PS-SCF-005", diff=diff, tol=tol)


@_guard
def check_chkfile_restore_electron_count(mol, dm, s, project, src=(), s_solver=None):
    """PS-SCF-006: checkpoint-projected initial guess conserves electron count.

    ``s`` is the overlap in which the restored orbitals are normalised: the
    target mol's for project=True (orbitals are renormalised there), the
    checkpoint molecule's for project=False (orbitals are reused as stored,
    so a displaced geometry legitimately changes tr(D S_target)).
    """
    tr = np.trace(dm @ s) if dm.ndim == 2 else np.trace(dm[0] @ s) + np.trace(dm[1] @ s)
    nao = mol.nao
    # D = C n C^T is stored with rounding eps*max|D|, which tr(D S) inherits
    dmax = max(1.0, float(np.abs(dm).max()))
    eps = _eps_of(dm, *src)
    tol = 50.0 * eps * nao * dmax if not project else 50.0 * eps * nao * 4 * dmax
    diff = abs(tr.real - mol.nelectron) if np.iscomplexobj(tr) else abs(tr - mol.nelectron)
    if s_solver is not None:
        _t = np.trace(dm @ s_solver) if dm.ndim == 2 else np.trace(dm[0] @ s_solver) + np.trace(dm[1] @ s_solver)
        diff = min(diff, abs(float(np.real(_t)) - mol.nelectron))
    trigger_if(diff > tol, "PS-SCF-006", diff=diff, tol=tol, project=bool(project))


# ---------------------------------------------------------------------------
# Subsystem 2 & 6: mp/mp2.py -- MP2
# ---------------------------------------------------------------------------

@_guard
def check_mp2_energy_sign(emp2, e_hf, homo=None, lumo=None):
    """PS-MP2-001: MP2 correlation energy must be non-positive at a canonical RHF ref.

    Precondition: Aufbau reference, i.e. every occupied orbital energy lies
    below every virtual one (homo < lumo). The sign follows from the
    denominators eps_i + eps_j - eps_a - eps_b < 0; a forced excited or
    non-Aufbau occupation has positive denominators and a positive MP2 energy.
    """
    if homo is not None and lumo is not None and not homo < lumo:
        return
    tol = 50.0 * eps64 * abs(e_hf)
    trigger_if(emp2 > tol, "PS-MP2-001", emp2=emp2, tol=tol)


@_guard
def check_mp2_t2_symmetry(t2, cmax=None):
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
    # Rounding is relative to the amplitude magnitude (t2 reaches 1e3-1e4 for
    # near-degenerate orbitals); the 50*eps*nao**1.5 base was derived at |t2|<~1.
    tol = 50.0 * eps64 * nao ** 1.5 * max(1.0, float(np.abs(t2).max()))
    # the ovov integrals are transformed with C in each of four indices, so their
    # rounding grows ~ eps * max|C|^4 (~ eps * cond(S)^2 for a redundant basis)
    if cmax is not None:
        tol *= max(1.0, float(cmax)) ** 4
    trigger_if(diff > tol, "PS-MP2-002", diff=diff, tol=tol)


@_guard
def check_mp2_rdm1_trace(dm1, nelectron, nmo, mo_occ=None, src=()):
    """PS-MP2-003: MP2 one-particle density matrix trace == nelectron.

    Precondition: the reference carries the molecule's occupations (a mocked
    reference whose mo_occ is absent or does not sum to nelectron does not
    describe an N-electron state).
    """
    if mo_occ is None or abs(float(np.sum(mo_occ)) - nelectron) > 1e-8:
        return
    tr = np.trace(dm1)
    diff = abs(tr - nelectron)
    # rounding of the trace is relative to the density magnitude
    tol = 50.0 * _eps_of(dm1, *src) * nmo * max(1.0, float(np.abs(dm1).max()))
    trigger_if(diff > tol, "PS-MP2-003", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 3: symm/addons.py
# ---------------------------------------------------------------------------

@_guard
def check_orbital_irrep_purity(proj_total, mo_norm, cond_s=None, src=(), s=None, symm_orb=None):
    """PS-SYM-001: the irrep projections of every MO add up to its overlap norm.

    label_orb_symm projects each orbital on every irrep of mol.symm_orb with
    the irrep-block overlap; because mol.symm_orb is a complete
    symmetry-adapted basis of the AO space (the irrep blocks are mapped back to
    the AO metric through its dual basis), sum_irrep <mo|P_irrep|mo>_S must
    equal <mo|mo>_S for ANY orbital, symmetry-pure or not. The purity of the
    caller-supplied orbitals (largest single-irrep share) is an input
    property that the function itself checks with its own tolerance, so it is
    not tested here (a 1e-6 symmetry-breaking rotation of valid orbitals
    would otherwise alarm, with deviation == angle**2, inside the band the
    library accepts). ``proj_total`` is the unnormalised sum over irreps and
    ``mo_norm`` the S-norm of each orbital.
    """
    proj_total = np.asarray(proj_total, dtype=float)
    mo_norm = np.asarray(mo_norm, dtype=float)
    scale = np.maximum(np.abs(mo_norm), 1e-300)
    worst = float(np.max(np.abs(proj_total - mo_norm) / scale))
    tol = 1e-12  # rounding of the Cholesky-solved projections (swept worst 3.1e-15)
    if cond_s is not None:
        # the irrep-block solves lose ~eps*cond of the block overlap (<= cond(S))
        tol = max(tol, 5.0 * eps64 * float(cond_s))
    if s is not None and symm_orb is not None:
        # The irrep projections use the block-symmetrised overlap while the norm uses
        # the exact one; for a geometry accepted as symmetric within symm_geom_tol the
        # two differ by d = ||S - S_blk||, shifting a normalised orbital's share by
        # d/lambda_min(S) (1 + d/lambda_min(S)) (first order vanishes for a pure irrep).
        s64 = np.asarray(s, dtype=float)
        try:
            s_blk = _blocks_to_ao(symm_orb, [o.conj().T @ s64 @ o for o in symm_orb], s64)
            d = float(np.linalg.norm(s64 - s_blk, 2))
        except (ValueError, np.linalg.LinAlgError):
            d = float("inf")  # not a complete basis: no near-symmetry allowance
        lam = float(np.linalg.eigvalsh((s64 + s64.T) / 2).min())
        # allowance only in the symmetry-tolerance regime (a basis that is not a symmetry
        # decomposition of the AO space shifts the shares by O(1) and must still alarm)
        if 0 < d <= 1e-4 * max(1.0, float(np.linalg.norm(s64, 2))) and lam > 0:
            tol += d / lam * (1.0 + d / lam)
    eps = _eps_of(*src)
    if eps > eps64:
        # reduced-precision orbitals/overlap: projections are defined to their roundoff
        tol = max(tol, 50.0 * eps * max(1.0, float(cond_s or 1.0)))
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
def check_uhf_electron_count_conservation(mo_occ_a_sum, mo_occ_b_sum, nelectron, spin, nmo=None):
    """PS-UHF-002: alpha/beta occupation sums match nelectron and spin exactly.

    Precondition: each spin channel fits in the orbital space (n_alpha, n_beta
    <= nmo); a request the basis cannot hold (e.g. triplet He in a one-function
    basis) is an impossible input, not a count-conservation question.
    """
    if nmo is not None:
        n_a = (nelectron + spin) // 2
        n_b = nelectron - n_a
        if n_a > nmo or n_b > nmo:
            return
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

def _contraction_cancellation(mol):
    """Per-AO cancellation factor kappa = sum|w_i w_j s_ij| / |sum w_i w_j s_ij|
    of the contracted radial function (w: coefficients of the normalised
    primitives, s_ij: their overlap). Normalising a contraction divides by
    its norm, so a rounding error eps in the self-overlap is amplified by
    kappa; kappa == 1 for an uncontracted or same-sign contraction.
    """
    from scipy.special import gammaln
    ao_loc = mol.ao_loc_nr()
    kap = np.ones(ao_loc[-1])
    for ib in range(mol.nbas):
        exps = np.asarray(mol.bas_exp(ib), dtype=float)
        if exps.size < 2:
            continue
        coefs = np.asarray(mol.bas_ctr_coeff(ib), dtype=float)
        l = mol.bas_angular(ib)
        p = l + 1.5
        # radial overlap of unnormalised r^l exp(-a r^2): Gamma(p)/(2 (ai+aj)^p)
        rii = 0.5 * np.exp(gammaln(p)) / (2 * exps) ** p
        sij = (2.0 * np.sqrt(exps[:, None] * exps[None, :]) / (exps[:, None] + exps[None, :])) ** p
        k = 1.0
        for c in coefs.T:
            w = c * np.sqrt(rii)
            ww = w[:, None] * w[None, :]
            den = abs((ww * sij).sum())
            if den > 0:
                k = max(k, float((np.abs(ww) * sij).sum() / den))
        kap[ao_loc[ib]:ao_loc[ib + 1]] = k
    return kap


def _screened_integral_mask(mol, lmax=6, exp_min=1e-4):
    """True for AOs outside the regime where the integral kernel's small-prefactor
    screening is valid: shells with l > lmax (beyond every standard basis, which
    stops at l = 6) or a primitive exponent below exp_min (radial extent of
    thousands of Bohr; the normalised coefficient underflows the screening)."""
    ao_loc = mol.ao_loc_nr()
    bad = np.zeros(ao_loc[-1], dtype=bool)
    for ib in range(mol.nbas):
        if mol.bas_angular(ib) > lmax or np.min(mol.bas_exp(ib)) < exp_min:
            bad[ao_loc[ib]:ao_loc[ib + 1]] = True
    return bad


def _unrepresentable_norm_mask(mol):
    """True for AOs whose raw contraction cannot be normalised in float64.

    Normalising a contraction inverts the quadratic form
    q = max|c|^2 * integral(r^(2l+2) exp(-2 a_min r^2)) of its raw
    coefficients; q must be a normal double with eps headroom
    (sqrt(tiny/eps) < q^(1/2)-scale, no overflow). Coefficients beyond about
    1e+-146..154 overflow or go subnormal in that product, far outside the
    O(1e-6..1e3) range of any basis set.
    """
    from pyscf import gto as _gto
    from pyscf.gto.mole import gaussian_int
    ao_loc = mol.ao_loc_nr()
    bad = np.zeros(ao_loc[-1], dtype=bool)
    fi = np.finfo(float)
    q_lo = fi.tiny / fi.eps
    q_hi = fi.max
    try:
        for ib in range(mol.nbas):
            symb = mol.atom_symbol(mol.bas_atom(ib))
            l = mol.bas_angular(ib)
            es = np.asarray(mol.bas_exp(ib), dtype=float)
            shells = mol._basis.get(symb, mol._basis.get(_gto.mole._std_symbol(symb), []))
            for shell in shells:
                if shell[0] != l:
                    continue
                rows = ([r for r in shell[1:] if not isinstance(r, (int, float))]
                        if isinstance(shell[1], (int, float)) else shell[1:])
                arr = np.array(rows, dtype=float)
                if arr.ndim != 2 or arr.shape[0] != len(es) or not np.allclose(arr[:, 0], es):
                    continue
                cmax = np.abs(arr[:, 1:]).max()
                with np.errstate(all="ignore"):
                    q = cmax ** 2 * gaussian_int(2 * l + 2, 2 * es.min())
                if not np.isfinite(q) or not (q_lo < q < q_hi):
                    bad[ao_loc[ib]:ao_loc[ib + 1]] = True
    except Exception:
        pass
    return bad


def _ao_nprim(mol):
    ao_loc = mol.ao_loc_nr()
    npr = np.ones(ao_loc[-1])
    for ib in range(mol.nbas):
        npr[ao_loc[ib]:ao_loc[ib + 1]] = mol.bas_nprim(ib)
    return npr


@_guard
def check_ao_self_overlap_normalization(diag_s, mol=None):
    """PS-GTO-001: normalized-AO self-overlap diagonal is 1.

    Precondition: shells with l <= 6 and exponents >= 1e-4 (the range of every
    standard basis set; outside it the integral kernel drops the
    self-overlap of the tiny normalised prefactor), and contraction
    coefficients whose normalisation integral is a normal float64 (no
    overflow or underflow).
    """
    d = np.abs(np.asarray(diag_s) - 1.0)
    if mol is not None:
        bad = _screened_integral_mask(mol) | _unrepresentable_norm_mask(mol)
        if bad.shape == d.shape:
            d = np.where(bad, 0.0, d)
    base = 2000.0 * eps64  # ECP/contracted-shell bases reach ~8e-14; 5x headroom
    tol = base
    if mol is not None:
        kap = _contraction_cancellation(mol)
        if kap.shape == d.shape:
            # cancellation inside a contraction amplifies roundoff by kappa
            tol = np.maximum(base, 50.0 * eps64 * kap)  # swept worst diff/(eps*kappa) = 3.6
        # the kernel sums nprim^2 primitive pairs; rounding grows ~eps*nprim
        # (measured |S_ii-1| <= 3.7*eps*nprim for same-sign contractions up to 1536)
        npr = _ao_nprim(mol)
        if npr.shape == d.shape:
            tol = np.maximum(tol, 50.0 * eps64 * npr * (kap if kap.shape == d.shape else 1.0))
    diff = d.max()
    trigger_if(bool(np.any(d > tol)), "PS-GTO-001", diff=diff, tol=float(np.max(tol)))


# ---------------------------------------------------------------------------
# Subsystem 8 & 12: dft/rks.py, dft/uks.py
# ---------------------------------------------------------------------------

def grid_count_error(mol, coords, weights, dms):
    """|tr(D (S_grid - S))| for each density matrix in ``dms``.

    S_grid = sum_g w_g ao_g ao_g^T is the overlap the integration grid
    reproduces; the quantity is exactly the electron-count error the grid
    quadrature alone introduces for density D (a diffuse or tight shell the
    grid cannot resolve, with N electrons in it, scales the error by N).
    Returns None when it cannot be evaluated.
    """
    if mol is None or coords is None or weights is None:
        return None
    coords = np.asarray(coords)
    weights = np.asarray(weights)
    if coords.shape[0] != weights.shape[0]:
        return None
    s = mol.intor_symmetric('int1e_ovlp')
    sg = np.zeros_like(s)
    step = 4000
    for p0 in range(0, coords.shape[0], step):
        ao = mol.eval_gto('GTOval', coords[p0:p0 + step])
        sg += (ao * weights[p0:p0 + step, None]).T @ ao
    return [float(abs(np.sum(np.asarray(d) * (sg - s)))) for d in dms]


def _grid_resolves_density(mol, coords, weights, dms, tol):
    """True unless the grid's own count error for the density exceeds tol/10."""
    err = grid_count_error(mol, coords, weights, dms)
    if err is None:
        return True
    return bool(max(err) <= 0.1 * tol)


def grid_not_coarser_than_default(grids, mol):
    """True when the integration grid is at least as fine as PySCF's default.

    The quadrature error is a property of the grid, so the electron-count law
    (tolerance calibrated on the default grid) only applies when the grid in
    use is no coarser: grids.level not below the default level, every
    explicitly requested atom_grid (n_rad, n_ang) not below the default for
    that element, and the default (or no) pruning.
    """
    try:
        from pyscf import gto as _gto
        from pyscf.dft import gen_grid as gg
        if grids.level < gg.Grids.level:
            return False
        # AO-value screening threshold: a coarser (larger) cutoff zeroes AO
        # values the quadrature needs; count error ~0.2*cutoff (H2/6-31G sweep)
        if float(getattr(grids, "cutoff", gg.Grids.cutoff)) > float(gg.Grids.cutoff):
            return False
        prune = getattr(grids, "prune", None)
        if prune is not None and prune is not gg.nwchem_prune:
            return False
        ag = getattr(grids, "atom_grid", None)
        if ag:
            if isinstance(ag, (list, tuple)):
                ag = {mol.atom_symbol(ia): ag for ia in range(mol.natm)}
            for ia in range(mol.natm):
                symb = mol.atom_symbol(ia)
                if symb in ag:
                    z = _gto.charge(symb)
                    n_rad, n_ang = ag[symb]
                    n_ang = gg.LEBEDEV_ORDER.get(n_ang, n_ang) if n_ang not in gg.LEBEDEV_NGRID else n_ang
                    if (n_rad < gg._default_rad(z, gg.Grids.level)
                            or n_ang < gg._default_ang(z, gg.Grids.level)):
                        return False
        return True
    except Exception:
        return False


@_guard
def check_dft_grid_electron_count(n_grid, dm, s, mol=None, coords=None, weights=None):
    """PS-DFT-001: grid-quadrature electron count matches trace(D S).

    Precondition: the grid resolves the density (the count error
    |tr(D (S_grid - S))| the quadrature itself causes is below a tenth of
    the tolerance).
    """
    tol = 1e-3
    if not _grid_resolves_density(mol, coords, weights, [dm], tol):
        return
    tr = np.trace(dm @ s)
    diff = abs(n_grid - tr)  # calibrated from a grid-level sweep: worst observed |n - tr|
    #             was 3.4e-6 on PySCF's default grid across 5 molecules/bases;
    #             set with generous headroom since quadrature error is real
    #             and grid-level dependent by design (5.8 T), not machine-eps.
    trigger_if(diff > tol, "PS-DFT-001", diff=diff, tol=tol)


@_guard
def check_uks_grid_electron_count(n_grid_tuple, dm, s, mol=None, coords=None, weights=None):
    """PS-UKS-001: per-spin-channel grid electron count matches trace(D_sigma S).

    Precondition: as PS-DFT-001, for each spin density.
    """
    if not _grid_resolves_density(mol, coords, weights, [dm[0], dm[1]], 1e-3):
        return
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
    Observed only after at least one update_amps step (the hook skips max_cycle == 0,
    which returns the unmodified MP2 initial guess).
    """
    diff = np.abs(t2 - t2.transpose(1, 0, 3, 2)).max()
    tol = 50.0 * eps64
    trigger_if(diff > tol, "PS-CC-001", diff=diff, tol=tol)


@_guard
def check_ccsd_rdm1_trace(dm1, nelectron, nmo, src=()):
    """PS-CC-002: CCSD one-particle density matrix (with lambda) trace == nelectron.

    Precondition: the full density (with_mf=True); the correlation-only part has
    trace zero by construction.
    """
    tr = np.trace(dm1)
    diff = abs(tr - nelectron)
    tol = 50.0 * _eps_of(dm1, *src) * nmo
    trigger_if(diff > tol, "PS-CC-002", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 10: dft/numint.py
# ---------------------------------------------------------------------------

@_guard
def check_grid_density_representation_consistency(rho_from_dm, rho_from_mo, nao, cmax=None, ao_max=None, src=()):
    """PS-NUMINT-001: eval_rho (density-matrix path) matches eval_rho2 (MO path)."""
    a = np.asarray(rho_from_dm)
    b = np.asarray(rho_from_mo)
    if a.shape != b.shape:
        return
    diff = np.abs(a - b).max()
    # rounding is relative to the density magnitude (rho reaches 1e4-1e5 for
    # tight exponents); the base was derived at rho = O(1)
    scale = max(1.0, float(np.abs(a).max()), float(np.abs(b).max()))
    # D = C n C^T is rounded at eps*max|C|^2, weighted by the AO values squared
    if cmax is not None and ao_max is not None:
        scale = max(scale, (float(cmax) * float(ao_max)) ** 2)
    tol = 50.0 * _eps_of(a, b, *src) * nao * 10 * scale  # worst observed ratio ~1.5e-5/eps64/nao-scale
    #                                 headroom set generously since GGA rows sum
    #                                 strictly more contraction terms than LDA.
    trigger_if(diff > tol, "PS-NUMINT-001", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 11: lo/boys.py
# ---------------------------------------------------------------------------

@_guard
def check_localization_density_invariance(dm_before, dm_after, s, nao, n_iter, src=()):
    """PS-LO-001: Boys/Edmiston/Pipek localization preserves the density matrix."""
    diff = np.abs(dm_before - dm_after).max()
    cond_s = np.linalg.cond(s)
    tol = 50.0 * _eps_of(dm_before, dm_after, *src) * nao * cond_s * max(1, n_iter)
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
def check_fci_rdm1_trace(dm1a, dm1b, neleca, nelecb, norb, norm2=1.0, ndet=None, civec_dtype=None):
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
    # The unnormalized-vector RDM carries the factor <c|c>; the electron-count
    # law is stated for the normalized state.
    if not np.isfinite(norm2) or norm2 <= 0:
        return
    # Precondition: float64 CI vector, the only dtype the compiled kernel (and
    # direct_spin1.energy) accepts; other dtypes are read as raw float64 memory.
    if civec_dtype is not None and np.dtype(civec_dtype) != np.float64:
        return
    tr_a = np.trace(np.asarray(dm1a)) / norm2
    tr_b = np.trace(np.asarray(dm1b)) / norm2
    # The compiled kernel accumulates the trace over all ndet determinants;
    # the rounding grows ~0.3*eps*ndet for a coherent (uniform) vector.
    tol = 50.0 * eps64 * norb + 4.0 * eps64 * float(ndet or 0)
    trigger_if(abs(tr_a - neleca) > tol, "PS-FCI-001",
               family="alpha", diff=abs(tr_a - neleca), tol=tol)
    trigger_if(abs(tr_b - nelecb) > tol, "PS-FCI-001",
               family="beta", diff=abs(tr_b - nelecb), tol=tol)


@_guard
def check_fci_energy_eigenvalue_consistency(h1e, eri, fcivec, norb, nelec, ecore, e_reported, solver=None):
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
    # Instance-level replacement of the contraction/Hamiltonian (e.g.
    # fix_spin_'s S^2 penalty) makes the reported eigenvalue belong to a
    # different operator than energy() recontracts.
    if solver is not None and (
            any(k in vars(solver) for k in ("contract_2e", "absorb_h1e", "make_hdiag", "eig"))
            or any(c.__name__ == "SpinPenaltyFCISolver" for c in type(solver).__mro__)):
        return
    e_recontracted = direct_spin1.energy(h1e, eri, fcivec, norb, nelec) + ecore
    diff = abs(e_recontracted - e_reported)
    # 1e-8 Ha absolute (Davidson residual scale), widened to the rounding of
    # the energy sum when the Hamiltonian terms are huge (tight exponents).
    ne = sum(nelec) if isinstance(nelec, (tuple, list)) else nelec
    scale = (abs(e_reported) + abs(e_recontracted) + abs(ecore)
             + ne * float(np.abs(h1e).max()) + ne ** 2 * float(np.abs(eri).max()))
    tol = max(1e-8, 50.0 * eps64 * scale)
    # The solver assumes the (pq|rs) = (rs|pq) = (qp|rs) symmetry; integrals
    # that violate it (rounding of an ill-conditioned AO->MO transform) enter
    # the energy contraction at O(N^2 * asymmetry).
    e = np.asarray(eri)
    if e.size == norb ** 4:
        e4 = e.reshape(norb, norb, norb, norb)
        asym = max(np.abs(e4 - e4.transpose(1, 0, 2, 3)).max(),
                   np.abs(e4 - e4.transpose(2, 3, 0, 1)).max())
    elif e.ndim == 2 and e.shape[0] == e.shape[1]:
        asym = np.abs(e - e.T).max()
    else:
        asym = 0.0
    tol += ne ** 2 * float(asym)
    trigger_if(diff > tol, "PS-FCI-002", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 15: ci/cisd.py -- configuration interaction singles and doubles
# ---------------------------------------------------------------------------

@_guard
def check_cisd_rdm1_trace(dm1, nelectron, nmo, mo_occ=None, src=()):
    """PS-CISD-001: CISD one-particle density matrix trace == nelectron.

    Precondition: the reference carries the molecule's occupations (a mocked
    reference whose mo_occ is absent or does not sum to nelectron does not
    describe an N-electron state).

    _gamma1_intermediates builds doo/dov/dvo/dvv from the variational,
    c0-normalized amplitude triple (c0,c1,c2) -- structurally different
    arithmetic from CCSD's non-variational t1/t2+l1/l2 combination
    (PS-CC-002) and MP2's t2-only construction (PS-MP2-003), even though all
    three feed the same downstream ccsd_rdm._make_rdm1 assembly function.

    Tolerance derivation (independent re-sweep, 4 molecules: H2, LiH, HF,
    N2, sto-3g): worst observed diff/(eps64*nmo) ratio was 1.33. 50x gives
    ~37x headroom.
    """
    if mo_occ is None or abs(float(np.sum(mo_occ)) - nelectron) > 1e-8:
        return
    tr = np.trace(dm1)
    diff = abs(tr - nelectron)
    tol = 50.0 * _eps_of(dm1, *src) * nmo
    trigger_if(diff > tol, "PS-CISD-001", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 16: mcscf/casci.py, mcscf/addons.py -- CASCI/CASSCF
# ---------------------------------------------------------------------------

@_guard
def check_casci_electron_count(casdm1, nelecas, ndet=None):
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
    # The Davidson eigenvector is normalized only to its convergence level, so
    # the RDM trace carries that residual (~1e-12 at default conv_tol).
    # plus the ndet-linear accumulation rounding of the FCI RDM kernel (PS-FCI-001)
    tol = max(50.0 * eps64 * ncas, 1e-9, 8.0 * eps64 * float(ndet or 0))
    trigger_if(diff > tol, "PS-MCSCF-001", diff=diff, tol=tol)


@_guard
def check_casscf_canonicalization_density_invariance(dm_before, dm_after, s, nao,
                                                     casdm1=None, cas_natorb=False, mo_coeff=None, src=()):
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
    if not _is_floating(mo_coeff):
        return
    diff = np.abs(dm_before - dm_after).max()
    cond_s = np.linalg.cond(s)
    tol = 50.0 * _eps_of(dm_before, dm_after, mo_coeff, *src) * nao * cond_s
    if cas_natorb and casdm1 is not None:
        # cas_natorb deliberately folds active natural orbitals whose occupation
        # is within FRAC_OCC_THRESHOLD of 0 or 2 into the core/virtual space
        # (pyscf issue 1041), which moves the density by O(occupation deviation).
        from pyscf.mcscf import casci
        occ = np.linalg.eigvalsh((casdm1 + casdm1.conj().T) / 2)
        dev = np.minimum(np.abs(occ), np.abs(2.0 - occ))
        folded = dev[dev < casci.FRAC_OCC_THRESHOLD]
        if folded.size:
            cmax2 = 1.0 if mo_coeff is None else max(1.0, float(np.abs(mo_coeff).max()) ** 2)
            tol += 2.0 * float(folded.sum()) * cmax2
    trigger_if(diff > tol, "PS-MCSCF-002", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 17: ao2mo/incore.py -- AO-to-MO two-electron integral transform
# ---------------------------------------------------------------------------

@_guard
def check_ao2mo_permutation_symmetry(eri, nmo, cond_s, cmax=None, eri_ao_max=None):
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
    # rounding is relative to the integral magnitude (eri reaches 1e4+ for tight exponents)
    # amplification of AO-integral rounding: eps*max|C|^4, which exceeds the
    # cond(C)^2 budget for orbitals built from differences of near-identical
    # (nearly redundant) functions
    amp = cond_s if cmax is None else max(cond_s, float(cmax) ** 4)
    esc = max(1.0, float(np.abs(eri).max()), 0.0 if eri_ao_max is None else float(eri_ao_max))
    tol = 50.0 * eps64 * nmo ** 1.5 * amp * esc
    trigger_if(diff > tol, "PS-AO2MO-001", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 18: solvent/ddcosmo.py, solvent/pcm.py -- implicit solvation
# ---------------------------------------------------------------------------

@_guard
def check_implicit_solvent_energy_sign(epcm, eps, lmax=None, nleb=None, grid_count_err=None, nelec=None):
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
    # Precondition: a resolved ddCOSMO expansion. The truncated operator (small
    # lmax) is non-symmetric and its truncation error can flip the sign.
    if lmax is not None and lmax < 4:
        return
    # The angular quadrature must resolve the (lmax+1)^2 harmonic basis,
    # otherwise the discretized operator is not negative-definite.
    if nleb is not None and lmax is not None and nleb < (lmax + 1) ** 2:
        return
    # The solute density on the volume grid (psi) and the analytic potential
    # (phi) come from the same charge only when the grid resolves the
    # density; otherwise psi^T L^-1 phi is not a definite quadratic form.
    if grid_count_err is not None and grid_count_err > 1e-3 * max(1.0, float(nelec or 0.0)):
        return
    # an energy that is zero by symmetry/neutrality is only defined to roundoff of the
    # underlying sums (~1e-14 Ha, up to ~1e-10 Ha for compact systems), so keep an
    # absolute floor of 1e-9 Ha
    tol = max(50.0 * eps64 * abs(epcm), 1e-9)
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
    # b = norm_to * a^-1 scales as 1/|a|: use the magnitude of b itself
    tol = 50.0 * eps64 * cond_a * np.abs(b_ref).max()
    trigger_if(diff > tol, "PS-PBC-001", diff=diff, tol=tol)


@_guard
def check_pbc_ewald_algorithm_consistency(e_direct, e_pme, ew_eta=None, ew_cut=None, precision=None,
                                           charge_sq=None):
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
    # Precondition: the cell precision is at least as tight as the 1e-6 claim;
    # the PME mesh is sized from cell.precision, so a looser user setting
    # legitimately allows a PME deviation of that size.
    if precision is not None and precision > 1e-6:
        return
    if ew_eta is not None and ew_cut is not None:
        # The two paths truncate the real-space sum differently (lattice shell vs
        # sphere r<cut); they agree to ~1.5*erfc(eta*cut) relative, so the 1e-6
        # law presumes a converged real-space sum.
        from scipy.special import erfc
        if erfc(float(ew_eta) * float(ew_cut)) > 1e-7:
            return
    diff = abs(e_direct - e_pme)
    # The terms of the lattice sum are of order eta/sqrt(pi) * sum(Z^2) (the self
    # term); PME's discretisation error is relative to that scale, not to a net
    # energy that cancels towards zero.
    scale = abs(e_direct)
    if ew_eta is not None and charge_sq is not None:
        scale = max(scale, float(ew_eta) / np.sqrt(np.pi) * float(charge_sq))
    tol = 1e-6 * max(scale, 1e-8)
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

def reduced_aspect_ratio(a):
    """max/min length of a Minkowski-style reduced basis of the lattice ``a``
    (rows), a basis-invariant shape measure (the supplied vectors may be an
    arbitrary skew choice of the same lattice)."""
    v = np.array(a, dtype=float)
    for _ in range(200):
        changed = False
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                m = np.round(v[i] @ v[j] / (v[j] @ v[j]))
                if m != 0 and np.linalg.norm(v[i] - m * v[j]) < np.linalg.norm(v[i]) - 1e-12:
                    v[i] = v[i] - m * v[j]
                    changed = True
            others = [k for k in range(3) if k != i]
            for sj in (-1, 1):
                for sk in (-1, 1):
                    t = v[i] + sj * v[others[0]] + sk * v[others[1]]
                    if np.linalg.norm(t) < np.linalg.norm(v[i]) - 1e-12:
                        v[i] = t
                        changed = True
        if not changed:
            break
    lens = np.linalg.norm(v, axis=1)
    return float(lens.max() / lens.min())


@_guard
def check_pbc_madelung_correction_sign(ewald_value, aspect_ratio=None, dimension=3):
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
    if dimension != 3:
        return  # the sign law is a 3D-periodic statement (0D/1D/2D cells use truncated Coulomb)
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
