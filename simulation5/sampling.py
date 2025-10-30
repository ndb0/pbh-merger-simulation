from __future__ import division
import numpy as np
import emcee

G_N = 4.302e-3 #(pc/solar mass) (km/s)^2

# Coalescence time - MODIFIED FOR M1, M2
def t_coal(a, e, M1, M2):
    """Calculates coalescence time for a binary with masses M1, M2."""
    if a <= 0 or e < 0 or e >= 1 or M1 <= 0 or M2 <= 0:
        return 0.0 # Invalid input

    M = M1 + M2  # Total mass
    # Check if M is zero or negative before division
    if M <= 0: return 0.0
    mu = (M1 * M2) / M # Reduced mass
    eta = mu / M # Symmetric mass ratio
    if eta <= 0: return 0.0 # Should not happen if M1, M2 > 0

    # Generalized Q factor's mass-dependent part (using Eq. 2.12 from Raidal et al. 2019)
    # Q_factor = 3 / (85 * eta * M**3 * G_N**3) # Direct implementation of Eq. 2.12 constant
    # To maintain consistency with original constant factor structure:
    Q_mass_term = G_N**3 * M1 * M2 * (M1 + M2) # M_PBH^3 equivalent term
    if Q_mass_term == 0: return 0.0
    Q_const_factor = (3.0/170.0) # Original code used 170, likely absorbs G_N etc. Check paper constants if needed.
    Q = Q_const_factor / Q_mass_term

    # Use j^7 = (1-e^2)^(7/2) approximation from Eq. 2.12
    j_squared = 1.0 - e**2
    if j_squared <= 0: return 0.0 # Avoid issues with power
    j_pow_7 = j_squared**(3.5)

    tc_geom = Q * a**4 * j_pow_7 # Units: pc^4 / (pc^3/s^6 * (km/s)^-6) -> pc * s^6 * (km/s)^6 ?? Check units carefully. Original code comments suggest s^6 pc km^-6

    # Conversion factors from original code
    tc_seconds = tc_geom * 3.086e13 * (3e5)**5 # Convert pc*(km/s)^6 term? Check conversion derivation.

    # Avoid negative or zero time
    if tc_seconds <= 0: return 0.0

    return tc_seconds / (60*60*24*365) # in years

# Coalescence j (for mergers at time t) - MODIFIED FOR M1, M2
def j_coal(a, t_years, M1, M2):
    """Calculates dimensionless angular momentum j = sqrt(1-e^2) for merger at t_years."""
    if a <= 0 or t_years <= 0 or M1 <= 0 or M2 <= 0:
        return 0.0 # Invalid input

    M = M1 + M2
    if M <= 0: return 0.0
    mu = (M1 * M2) / M
    eta = mu / M
    if eta <= 0: return 0.0

    # Use the same generalized Q as in t_coal
    Q_mass_term = G_N**3 * M1 * M2 * (M1 + M2)
    if Q_mass_term == 0: return 0.0
    Q_const_factor = (3.0/170.0)
    Q = Q_const_factor / Q_mass_term

    # Convert time back to geometric units used in formula
    tc_seconds = t_years * (60*60*24*365)
    # Be careful with units here, ensure consistency with t_coal conversion
    tc_geom = tc_seconds / ((3e5)**5 * 3.086e13)

    # Invert the approximate formula (Eq. 2.12) for j^7 = (1-e^2)^(7/2)
    if Q == 0 or a == 0: return 0.0
    j_pow_7 = tc_geom / (Q * a**4)

    # Check for physically possible values (tc must be positive -> j_pow_7 positive)
    if j_pow_7 <= 0:
        return 0.0 # Merger time requires j <= 0, unphysical for bound orbit

    # Check if result requires j > 1 (e^2 < 0)
    if j_pow_7 >= 1.0:
         # Merger time is too short for these parameters / requires e<0 (unphysical)
         # Return a value indicating it merges very fast, or is invalid.
         # Returning j=1 (circular, fastest merge for given a) might be an option,
         # or 0 if sampling should discard these. Check desired behavior.
         return 1.0 # Cap at circular orbit for fastest possible merge

    j = (j_pow_7)**(1.0/7.0)

    # Ensure j is not > 1 due to approximations/numerical issues
    return min(j, 1.0)


# Log-prior - MODIFIED FOR M1, M2
def lnprior(theta, M1, M2, a1, a2, tmin_sampling=1.e8, tmax_sampling=1.e11):
    """Log prior probability for parameters theta=(log10(a), log10(j))."""
    la, lj = theta
    # Check if parameters result in finite physical values
    if not (np.isfinite(la) and np.isfinite(lj)):
         return -np.inf

    a = 10**la
    j = 10**lj

    if not (j > 0 and j <= 1.0): # Ensure j is strictly positive and <= 1
        return -np.inf

    if not (a >= a1 and a <= a2): # Use >= and <= for range limits
        return -np.inf

    # Calculate eccentricity, ensure it's valid
    e_squared = 1. - j**2
    if e_squared < 0 or e_squared >= 1: # e must be < 1 for bound orbits (excluding parabolic)
         return -np.inf
    e = np.sqrt(e_squared)

    # Use modified t_coal
    t = t_coal(a, e, M1=M1, M2=M2)

    # Check if t calculation was valid and within sampling range
    if t <= 0 or not np.isfinite(t): return -np.inf

    if (t < tmin_sampling or t > tmax_sampling):
        return -np.inf

    return 0 # Log prior is 0 if within bounds

# Log-probability - MODIFIED FOR M1, M2
def lnprob(theta, f, M1, M2, PDF_unequal, a1, a2):
    """Log posterior probability, combining prior and likelihood (PDF)."""
    # Use modified lnprior
    lp = lnprior(theta, M1, M2, a1, a2) # Assuming default tmin/tmax are ok
    if not np.isfinite(lp):
        return -np.inf

    la, lj = theta
    a = 10**la
    j = 10**lj

    # !!! CRUCIAL: PDF_unequal MUST be the unequal mass probability distribution !!!
    # This function needs to be implemented based on Raidal et al. 2019 or similar.
    # It should take (la, lj, f, M1, M2) or (a, j, f, M1, M2) as arguments.
    # Example call assuming it takes log10 values:
    try:
        pdf_val = PDF_unequal(la, lj, f, M1, M2)
    except Exception as e:
        # Catch potential errors in the user-provided PDF
        # print(f"Warning: PDF function failed for la={la}, lj={lj}. Error: {e}")
        return -np.inf

    # Ensure PDF value is valid for log
    if pdf_val <= 0 or not np.isfinite(pdf_val):
        return -np.inf

    log_pdf_val = np.log(pdf_val)

    return lp + log_pdf_val


# Sampler - MODIFIED FOR M1, M2
def GetSamples_MCMC(N_samps, PDF_unequal, a1, a2, f, M1, M2, nwalkers=100, burn_in_steps=1000, thin_by=5):
    """
    Runs MCMC sampler to get samples from the unequal mass PDF.

    Args:
        N_samps (int): Number of desired samples *after* thinning.
        PDF_unequal (callable): The unequal mass probability distribution function.
                                Should accept (la, lj, f, M1, M2).
        a1, a2 (float): Min and max semi-major axis for prior.
        f (float): PBH fraction.
        M1, M2 (float): Component masses.
        nwalkers (int): Number of MCMC walkers.
        burn_in_steps (int): Number of steps to discard as burn-in.
        thin_by (int): Factor to thin the chain by.

    Returns:
        numpy.ndarray: Array of samples (shape: [N_samps, 2]).
    """
    ndim = 2
    total_steps = (N_samps * thin_by // nwalkers + 1) * nwalkers # Ensure enough steps for thinning
    if total_steps <= burn_in_steps:
         print(f"Warning: Total steps ({total_steps}) <= burn_in ({burn_in_steps}). Increasing total steps.")
         total_steps = burn_in_steps + (N_samps * thin_by // nwalkers + 1) * nwalkers


    # --- Initial Guess ---
    a0 = np.sqrt(a1 * a2) # Geometric mean for a
    # Use modified j_coal for initial guess, assuming merger at t0=13e9 years
    t0_guess = 13e9
    j0 = j_coal(a0, t0_guess, M1=M1, M2=M2)

    # Handle case where initial guess is invalid (e.g., merges too fast)
    if j0 <= 0 or j0 >= 1.0:
        print(f"Warning: Initial j0 calculation invalid or boundary value ({j0:.2e}) for a0={a0:.2e}, M1={M1}, M2={M2}. Trying different initial t.")
        # Try a different time, maybe closer to t_min or t_max of prior?
        j0_alt = j_coal(a0, 1e9, M1=M1, M2=M2) # Try 1 Gyr
        if j0_alt <= 0 or j0_alt >=1.0:
             j0 = 1e-3 # Fallback to a small value if still problematic
             print(f"Warning: Alternative j0 also invalid. Using default j0={j0:.1e}.")
        else:
             j0 = j0_alt
             print(f"Using alternative j0 = {j0:.2e}")


    # --- Initial positions for walkers ---
    p0 = []
    attempts = 0
    max_attempts = nwalkers * 100 # Limit attempts to avoid infinite loop

    while len(p0) < nwalkers and attempts < max_attempts:
        attempts += 1
        # Start close to a likely region, ensure j0 is > 0 before log10
        a_start = a0 * (1 + 0.1 * (np.random.rand() - 0.5)) # Increased spread for a
        j_start = j0 * (1 + 0.5 * (np.random.rand() - 0.5)) # Increased spread for j
        j_start = max(1e-9, min(j_start, 0.99999)) # Keep j within (0, 1) bounds

        # Check if this starting point has finite probability
        la_start, lj_start = np.log10(a_start), np.log10(j_start)
        initial_log_prob = lnprob([la_start, lj_start], f, M1, M2, PDF_unequal, a1, a2)

        if np.isfinite(initial_log_prob):
            p0.append([la_start, lj_start])
        # else:
            # Optional: print warning for rejected start points
            # print(f"Attempt {attempts}: Rejected start point a={a_start:.2e}, j={j_start:.2e}, log_prob={initial_log_prob}")


    if len(p0) < nwalkers:
        raise ValueError(f"Could not find {nwalkers} valid starting positions after {max_attempts} attempts. Check PDF or priors.")

    p0 = np.array(p0) # Convert to numpy array

    # --- Initialize and run sampler ---
    print(f"Starting MCMC for f={f:.2e}, M1={M1}, M2={M2}...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[f, M1, M2, PDF_unequal, a1, a2])

    # Run MCMC with progress bar
    sampler.run_mcmc(p0, total_steps, progress=True)

    # --- Extract samples ---
    # Discard burn-in, flatten, and thin the chain
    samples = sampler.get_chain(discard=burn_in_steps, flat=True, thin=thin_by)

    print(f"   Generated {len(samples)} samples after burn-in and thinning...")
    return samples
