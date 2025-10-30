from __future__ import division, print_function
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import warnings # To handle integration warnings

# --- Constants ---
alpha = 0.1
z_eq = 3375.0
rho_eq = 1512.0 #Solar masses per pc^3
lambda_max = 3.0 # Decouple at a maximum redshift of z_dec = z_eq (lambda = 3.0*z_dec) # Original comment mismatch? Assuming lambda_max=3 relates to X/0.85f limit
G_N = 4.302e-3 #Units of (pc/solar mass) (km/s)^2

# --- Global Cache for Interpolators ---
# Store interpolators in a dictionary keyed by mass
interpolator_cache = {}

# --- Halo Property Functions (Depend on single mass M_PBH - Keep signature) ---
# M_PBH in solar masses
def r_trunc(z, M_PBH):
    """Truncation radius at redshift z for a PBH of mass M_PBH."""
    if M_PBH <= 0: return 0.0
    r0 = 6.3e-3 # 1300 AU in pc
    return r0 * (M_PBH)**(1.0/3.0) * (1. + z_eq) / (1. + z)

def r_eq(M_PBH):
    """Truncation radius at equality for a PBH of mass M_PBH."""
    return r_trunc(z_eq, M_PBH)

def M_halo(z, M_PBH):
    """Mass of the DM halo accreted by redshift z for a PBH of mass M_PBH."""
    if M_PBH <= 0: return 0.0
    # Ensure r_eq(M_PBH) > 0 to avoid division by zero or NaN
    req = r_eq(M_PBH)
    if req <= 0: return 0.0 # No halo if r_eq is non-positive

    # Ensure z is non-negative
    z_phys = max(0.0, z)
    rt = r_trunc(z_phys, M_PBH)
    if rt <= 0: return 0.0 # Treat as no halo if truncation radius is non-positive

    # Original formula could give NaN if rt < 0, ensure positive base
    ratio = rt / req # rt and req should be positive now
    return M_PBH * (ratio)**1.5

# --- Density and Enclosed Mass (Depend on single mass M_PBH - Keep signature) ---
def rho(r, r_tr, M_PBH, gamma=3.0/2.0):
    """Density profile rho(r) for a halo truncated at r_tr."""
    if r_tr <= 0 or r < 0: return 0.0 # No density if no halo or invalid radius
    x = r / r_tr
    req = r_eq(M_PBH)
    if req <= 0: return 0.0 # Avoid division by zero, implies no standard halo formation

    # Avoid negative base for power if gamma is not integer and r_tr or req negative (checked)
    A_denom = (4 * np.pi * (r_tr**gamma) * (req**(3 - gamma)))
    if A_denom == 0: return 0.0 # Avoid division by zero

    A = (3 - gamma) * M_PBH / A_denom

    if (x <= 1):
        # Handle potential zero division if x=0 and gamma>0
        if x == 0 and gamma > 0: return np.inf # Density cusp
        return A * x**(-gamma)
    else:
        return 0

def Menc(r, r_tr, M_PBH, gamma=3.0/2.0):
    """Enclosed mass M(<r) including PBH and halo truncated at r_tr."""
    if M_PBH <= 0: return 0.0
    if r < 0: r = 0.0 # Enclosed mass at negative radius is zero/undefined

    if r_tr <= 0: return M_PBH # Assume only PBH mass if no halo radius
    x = r / r_tr
    req = r_eq(M_PBH)
    if req <= 0: return M_PBH # Assume only PBH mass if no r_eq

    # Ensure base of power is non-negative (already done for req, r_tr)
    r_over_req = r / req
    rtr_over_req = r_tr / req

    if (x <= 1):
        # Ensure r_over_req >= 0
        if r_over_req < 0: r_over_req = 0.0
        # Handle 0**(positive power) = 0, 0**(zero power)=1, 0**(negative power)=inf
        power_term = 0.0
        exponent = 3.0 - gamma
        if r_over_req == 0:
            if exponent > 0: power_term = 0.0
            elif exponent == 0: power_term = 1.0 # Should 1+1 = 2M? Check definition
            else: power_term = np.inf # Should not happen if r=0 used correctly
        else:
            power_term = r_over_req**exponent

        if not np.isfinite(power_term): power_term = 0.0 # If something went wrong

        return M_PBH * (1. + power_term)
    else:
        # Ensure rtr_over_req >= 0 (already done)
        power_term = 0.0
        exponent = 3.0 - gamma
        if rtr_over_req == 0:
             if exponent > 0: power_term = 0.0
             elif exponent == 0: power_term = 1.0
             else: power_term = np.inf
        else:
            power_term = rtr_over_req**exponent

        if not np.isfinite(power_term): power_term = 0.0

        return M_PBH * (1. + power_term)


# --- Binding Energy Calculation (Depend on single mass M_PBH - Keep signature) ---
def calcBindingEnergy(r_tr, M_PBH):
    """Calculates gravitational binding energy of the halo."""
    if r_tr <= 0: return 0.0 # No binding energy if no halo

    # Define integrand, handle potential issues inside
    def integrand(r):
        density = rho(r, r_tr, M_PBH)
        enclosed_mass = Menc(r, r_tr, M_PBH)
        if not (np.isfinite(density) and np.isfinite(enclosed_mass)):
            return 0.0 # Return 0 if calculation fails
        return enclosed_mass * density * r

    lower_bound = 1e-9 * r_tr # Use a fraction of r_tr as lower bound
    upper_bound = r_tr

    if upper_bound <= lower_bound: return 0.0

    try:
        # Suppress integration warnings which might occur near r=0 cusp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, error = quad(integrand, lower_bound, upper_bound, epsrel=1e-3, limit=100) # Increased limit
        # Check if result is finite
        if not np.isfinite(result):
             # print(f"Warning: quad result non-finite in calcBindingEnergy for r_tr={r_tr}, M_PBH={M_PBH}.")
             return 0.0
        # Binding energy is negative
        binding_energy = -G_N * 4 * np.pi * result
        return binding_energy

    except Exception as e:
        print(f"Warning: quad failed in calcBindingEnergy for r_tr={r_tr}, M_PBH={M_PBH}. Error: {e}")
        return 0.0 # Return 0 or handle error

# --- Formation Physics Primitives (!!! PLACEHOLDERS - NEED REPLACEMENT !!!) ---
# These depend on the specific Ali-Haimoud model for equal masses.
# Signatures updated, but implementations MUST BE REPLACED with unequal mass theory.

def xbar(f, M1, M2):
    """Mean comoving separation. !!! PLACEHOLDER - NEEDS UNEQUAL MASS DEFINITION !!!"""
    # Using total mass M as a placeholder. This is incorrect. See Raidal+(2019) section 2.1 [cite: 1066-1157]
    print("Warning: Using placeholder 'xbar' function based on total mass.")
    M_eff = M1 + M2 # Simplistic assumption
    if rho_eq <= 0 or f <= 0 or M_eff <= 0: return 0.0
    return (3.0 * M_eff / (4 * np.pi * rho_eq * (0.85 * f)))**(1.0 / 3.0)

def bigX(x, f, M1, M2):
     """Dimensionless separation cubed. !!! PLACEHOLDER - NEEDS UNEQUAL MASS DEFINITION !!!"""
     print("Warning: Using placeholder 'bigX' function.")
     xb = xbar(f, M1, M2) # Uses placeholder xbar
     if xb == 0: return np.inf # Avoid division by zero
     # Ensure x is non-negative
     x_phys = max(0.0, x)
     return (x_phys / xb)**3.0

def x_of_a(a, f, M1, M2):
     """Comoving separation corresponding to semi-major axis a. !!! PLACEHOLDER !!!"""
     print("Warning: Using placeholder 'x_of_a' function.")
     xb = xbar(f, M1, M2) # Uses placeholder xbar
     if alpha == 0 or f <= 0 or xb == 0 or a < 0: return 0.0 # Avoid issues
     term = (a * (0.85 * f) * xb**3.) / alpha
     if term < 0: return 0.0 # Ensure non-negative base for power
     return (term)**(1./4.)

def z_decoupling(a, f, M_calc1, M_calc2):
     """Decoupling redshift. !!! PLACEHOLDER - Needs careful implementation !!!"""
     # This function's role in the iteration is complex with two masses + halos.
     # Original code passed M_PBH + M_halo implicitly.
     # How does decoupling scale depend on M1, M2, M_halo1, M_halo2?
     # Simplistic Placeholder: Use effective mass M1+M2 for the calculation. Needs verification.
     # print("Warning: Using placeholder 'z_decoupling' function.")

     # Use the masses passed (which might be M_PBH or M_PBH+M_halo during iteration)
     M_eff1 = M_calc1
     M_eff2 = M_calc2

     if a <= 0 or f <= 0 or M_eff1 <= 0 or M_eff2 <= 0: return -1.0

     x_val = x_of_a(a, f, M_eff1, M_eff2) # Uses placeholder x_of_a
     X_val = bigX(x_val, f, M_eff1, M_eff2) # Uses placeholder bigX

     denom_term = (0.85 * f)
     if denom_term == 0: return -1.0
     denom = (1./3. * X_val / denom_term)
     if denom <= 0: return -1.0 # Invalid redshift (e.g., if X_val is non-positive)

     z_dec = (1. + z_eq) / denom - 1.
     # Ensure redshift is not excessively large or negative if calculations were unstable
     return max(-1.0, z_dec) # Cap at -1 (Big Bang)


# --- Interpolator Generation (Now specific to a given mass M_PBH) ---
def get_rtr_interpolator(M_PBH):
    """Generates the r_tr(a) interpolation function for a specific mass M_PBH."""
    print(f"   Generating r_tr(a) interpolator for M_PBH = {M_PBH}...")
    # NB: Using f = 0.01 as in original, assuming cancellation holds for this specific calculation
    f_calc = 0.01

    # --- a_max needs modification if it depends on unequal mass physics ---
    # Placeholder: Assuming a_max_with_Halo exists and takes single mass
    try:
         am = a_max_with_Halo(f_calc, M_PBH) # Needs definition/check based on unequal mass theory
         if not np.isfinite(am) or am <= 0: raise ValueError("a_max calculation failed.")
    except (NameError, ValueError):
         print("Warning: a_max_with_Halo failed or not defined. Using fallback calculation.")
         # Fallback using xbar with placeholder total mass M=2*M_PBH
         try:
              M_eff_am = M_PBH + M_halo(z_eq, M_PBH) # Approx mass for a_max calc
              xb_am = xbar(f_calc, M_eff_am, M_eff_am) # Use placeholder xbar
              am = alpha * xb_am * (f_calc * 0.85)**(1.0/3.0) * (lambda_max**(4.0/3.0))
              if not np.isfinite(am) or am <= 0: raise ValueError("Fallback a_max failed.")
         except ValueError:
              print("Error: Fallback a_max failed. Using default am=0.1.")
              am = 0.1 # Default if calculation fails

    a_list = np.logspace(-8, np.log10(am * 1.1), 101) # Use log10 for endpoint

    # Calculate z_decoupling iteratively, accounting for halo mass accretion
    # Initial guess using just M_PBH
    z_decoupling_current = np.array([z_decoupling(a, f_calc, M_PBH, M_PBH) for a in a_list])
    # Ensure non-negative before M_halo calculation
    z_decoupling_current[z_decoupling_current < 0] = 0.0
    M_halo_current = M_halo(z_decoupling_current, M_PBH)

    for _ in range(3): # Iterate 3 times as in original code
        M_total_iter = M_PBH + M_halo_current
        # Ensure M_total_iter is positive for z_decoupling call
        M_total_iter[M_total_iter <= M_PBH] = M_PBH # Fallback if halo mass negative/zero

        # Assuming M1=M2=M_total_iter for placeholder z_decoupling call during iteration
        z_decoupling_next = np.array([z_decoupling(a, f_calc, mt, mt) for a, mt in zip(a_list, M_total_iter)])

        # Ensure z_decoupling_next is non-negative before passing to M_halo
        z_decoupling_next[z_decoupling_next < 0] = 0.0
        M_halo_next = M_halo(z_decoupling_next, M_PBH)

        # Update for next iteration
        z_decoupling_current = z_decoupling_next
        M_halo_current = M_halo_next

    final_z_decoupling = z_decoupling_current
    # Ensure final redshift is non-negative for r_trunc
    final_z_decoupling[final_z_decoupling < 0] = 0.0

    r_list = r_trunc(final_z_decoupling, M_PBH)

    # --- Create Interpolator ---
    # Ensure a_list and r_list are sorted and finite for interpolation
    valid_indices = np.isfinite(a_list) & np.isfinite(r_list) & (a_list > 0) & (r_list > 0)
    if not np.any(valid_indices):
         print(f"Error: No valid data points for rtr interpolation for M_PBH={M_PBH}")
         return None # Failed

    a_list_valid = a_list[valid_indices]
    r_list_valid = r_list[valid_indices]

    # Sort by a_list_valid
    sort_indices = np.argsort(a_list_valid)
    a_list_sorted = a_list_valid[sort_indices]
    r_list_sorted = r_list_valid[sort_indices]

    # Remove duplicates in a_list_sorted if any, keeping the first occurrence
    unique_a, unique_indices = np.unique(a_list_sorted, return_index=True)
    if len(unique_a) < 2:
         print(f"Error: Need at least 2 unique points for interpolation for M_PBH={M_PBH}")
         return None # Failed

    try:
        # Use linear interpolation, allow extrapolation
        rtr_interp = interp1d(unique_a, r_list_sorted[unique_indices], kind='linear', fill_value="extrapolate", bounds_error=False)
        print(f"   Interpolator r_tr(a) created for M_PBH = {M_PBH}.")
        return rtr_interp
    except ValueError as e:
        print(f"Error creating rtr interpolator for M_PBH={M_PBH}: {e}")
        return None # Failed

def get_Ubind_interpolator(M_PBH):
    """Generates the U_bind(r_tr) interpolation function for a specific mass M_PBH."""
    print(f"   Generating U_bind(r_tr) interpolator for M_PBH = {M_PBH}...")
    req = r_eq(M_PBH)
    if req <= 0:
        print(f"Error: r_eq <= 0 for M_PBH={M_PBH}, cannot generate Ubind interpolator.")
        return None # Failed

    # Define range for r_tr, ensure lower bound is positive and less than upper
    r_tr_min = req * 1e-6 # Go down significantly from r_eq
    r_tr_max = req * 1.0
    if r_tr_min >= r_tr_max:
        print(f"Error: r_tr_min >= r_tr_max for Ubind interpolation (M={M_PBH}). Adjust range.")
        return None

    rtr_vals = np.logspace(np.log10(r_tr_min), np.log10(r_tr_max), 100) # Reduced points for speed

    # Calculate binding energies, ensure they are finite
    Ubind_vals = np.array([calcBindingEnergy(r1, M_PBH) for r1 in rtr_vals])

    # --- Create Interpolator ---
    valid_indices = np.isfinite(rtr_vals) & np.isfinite(Ubind_vals) & (rtr_vals > 0)
    if not np.any(valid_indices):
         print(f"Error: No valid data points for Ubind interpolation for M_PBH={M_PBH}")
         return None # Failed

    rtr_vals_valid = rtr_vals[valid_indices]
    Ubind_vals_valid = Ubind_vals[valid_indices]

    # Sort by rtr_vals_valid
    sort_indices = np.argsort(rtr_vals_valid)
    rtr_vals_sorted = rtr_vals_valid[sort_indices]
    Ubind_vals_sorted = Ubind_vals_valid[sort_indices]

    # Remove duplicates if any
    unique_rtr, unique_indices = np.unique(rtr_vals_sorted, return_index=True)
    if len(unique_rtr) < 2:
         print(f"Error: Need at least 2 unique points for Ubind interpolation for M_PBH={M_PBH}")
         return None # Failed

    try:
        # Use linear interpolation, allow extrapolation
        Ubind_interp = interp1d(unique_rtr, Ubind_vals_sorted[unique_indices], kind='linear', fill_value="extrapolate", bounds_error=False)
        print(f"   Interpolator U_bind(r_tr) created for M_PBH = {M_PBH}.")
        return Ubind_interp
    except ValueError as e:
        print(f"Error creating Ubind interpolator for M_PBH={M_PBH}: {e}")
        return None # Failed


# --- Interpolator Management ---
def setup_interpolators(M1, M2):
    """Generates and caches interpolators for M1 and M2 if not already done."""
    global interpolator_cache
    success = True
    masses_to_setup = {M1, M2} # Use a set to handle M1=M2 case efficiently

    for M_val in masses_to_setup:
        if M_val not in interpolator_cache or \
           interpolator_cache[M_val]['rtr'] is None or \
           interpolator_cache[M_val]['Ubind'] is None:

            print(f"Setting up interpolators for M_PBH = {M_val}...")
            rtr_interp = get_rtr_interpolator(M_val)
            Ubind_interp = get_Ubind_interpolator(M_val)

            if rtr_interp is None or Ubind_interp is None:
                 print(f"Error: Failed to generate interpolators for M={M_val}")
                 # Store None to avoid re-attempting constantly, but mark failure
                 interpolator_cache[M_val] = {'rtr': None, 'Ubind': None}
                 success = False
            else:
                 interpolator_cache[M_val] = {'rtr': rtr_interp, 'Ubind': Ubind_interp}
                 print(f"Interpolators ready for M_PBH = {M_val}.")
        # else: already cached and valid

    return success # Indicate overall success/failure


# --- Remapping Functions - MODIFIED FOR M1, M2 ---

def calc_af(ai, M1, M2):
    """Calculates final semi-major axis after halo ejection for binary (M1, M2)."""
    global G_N, interpolator_cache

    # Ensure valid inputs
    if ai <= 0 or M1 <= 0 or M2 <= 0:
        # print(f"Warning: Invalid input to calc_af (ai={ai}, M1={M1}, M2={M2}). Returning ai.")
        return ai if ai > 0 else np.inf # Return input a or indicate error

    # Ensure interpolators are ready
    if not setup_interpolators(M1, M2):
        print("Error: Could not setup interpolators. Cannot calculate af.")
        return ai # Return initial value or indicate error

    # Retrieve interpolators
    cache_M1 = interpolator_cache.get(M1)
    cache_M2 = interpolator_cache.get(M2)

    if cache_M1 is None or cache_M2 is None or \
       cache_M1['rtr'] is None or cache_M1['Ubind'] is None or \
       cache_M2['rtr'] is None or cache_M2['Ubind'] is None:
        print("Error: Interpolators not valid in cache. Cannot calculate af.")
        return ai

    rtr_interp_M1 = cache_M1['rtr']
    Ubind_interp_M1 = cache_M1['Ubind']
    rtr_interp_M2 = cache_M2['rtr']
    Ubind_interp_M2 = cache_M2['Ubind']

    # --- Get properties for M1 ---
    try:
        r_tr_1 = float(rtr_interp_M1(ai))
        # Ensure r_tr is physically plausible (e.g., positive)
        if r_tr_1 <= 0: raise ValueError("Interpolated r_tr_1 non-positive")
    except ValueError as e:
        # print(f"Warning: Interpolation/Extrapolation failed for rtr_interp_M1 at ai={ai}. Error: {e}. Using boundary/min value.")
        # Use smallest valid r_tr from interpolator domain if ai is too small, or a tiny default
        r_tr_1 = max(rtr_interp_M1.x[0], 1e-9) if len(rtr_interp_M1.x) > 0 else 1e-9

    M_tot_1 = Menc(r_tr_1, r_tr_1, M1)
    try:
         U_bind_1 = float(Ubind_interp_M1(r_tr_1))
         if not np.isfinite(U_bind_1): raise ValueError("Interpolated U_bind_1 non-finite")
    except ValueError as e:
         # print(f"Warning: Interpolation/Extrapolation failed for Ubind_interp_M1 at r_tr_1={r_tr_1}. Error: {e}. Using boundary/zero.")
         # Use boundary value or assume zero binding energy if extrapolation fails badly
         U_bind_1 = 0.0 # Safer fallback? Or use boundary:
         # U_bind_1 = Ubind_interp_M1.y[0] if r_tr_1 < Ubind_interp_M1.x[0] else Ubind_interp_M1.y[-1]


    # --- Get properties for M2 ---
    try:
        r_tr_2 = float(rtr_interp_M2(ai))
        if r_tr_2 <= 0: raise ValueError("Interpolated r_tr_2 non-positive")
    except ValueError as e:
        # print(f"Warning: Interpolation/Extrapolation failed for rtr_interp_M2 at ai={ai}. Error: {e}. Using boundary/min value.")
        r_tr_2 = max(rtr_interp_M2.x[0], 1e-9) if len(rtr_interp_M2.x) > 0 else 1e-9

    M_tot_2 = Menc(r_tr_2, r_tr_2, M2)
    try:
        U_bind_2 = float(Ubind_interp_M2(r_tr_2))
        if not np.isfinite(U_bind_2): raise ValueError("Interpolated U_bind_2 non-finite")
    except ValueError as e:
        # print(f"Warning: Interpolation/Extrapolation failed for Ubind_interp_M2 at r_tr_2={r_tr_2}. Error: {e}. Using boundary/zero.")
        U_bind_2 = 0.0 # Safer fallback


    # --- Energy Calculation ---
    U_orb_before = -G_N * M_tot_1 * M_tot_2 / (2.0 * ai)

    # Final orbital energy = initial orb + released binding energy.
    # calcBindingEnergy returns negative value, so add them.
    U_orb_final = U_orb_before + U_bind_1 + U_bind_2

    # Ensure final energy corresponds to a bound state (U_orb_final < 0)
    if U_orb_final >= -1e-15: # Use small tolerance instead of exact zero
        # print(f"Warning: Final orbit unbound or zero energy (U_orb_final={U_orb_final:.2e} >= 0). ai={ai:.2e}, M1={M1}, M2={M2}")
        return np.inf # Indicate unbound orbit

    # Calculate final semi-major axis (bare PBHs: M1, M2)
    af_denom = (2.0 * U_orb_final)
    # Denominator should be negative and non-zero here
    if af_denom == 0: return np.inf # Should not happen if U_orb_final < 0

    af = -G_N * M1 * M2 / af_denom

    # Ensure af is positive
    if af <= 0:
         # print(f"Warning: Calculated af={af:.2e} is non-positive. U_final={U_orb_final:.2e}. Returning inf.")
         return np.inf # Indicate an issue

    return af


def calc_jf(ji, ai, M1, M2):
    """Calculates final dimensionless angular momentum jf after halo ejection."""
    global interpolator_cache

    # Ensure valid inputs
    if not (0 < ji <= 1.0) or ai <= 0 or M1 <= 0 or M2 <= 0:
        # print(f"Warning: Invalid input to calc_jf (ji={ji}, ai={ai}, M1={M1}, M2={M2}). Returning ji.")
        return ji if (0 < ji <= 1.0) else 0.0

    # Ensure interpolators are ready
    if not setup_interpolators(M1, M2):
        print("Error: Could not setup interpolators. Cannot calculate jf.")
        return ji

    cache_M1 = interpolator_cache.get(M1)
    cache_M2 = interpolator_cache.get(M2)
    if cache_M1 is None or cache_M2 is None or cache_M1['rtr'] is None or cache_M2['rtr'] is None:
        print("Error: Interpolators not valid in cache. Cannot calculate jf.")
        return ji

    rtr_interp_M1 = cache_M1['rtr']
    rtr_interp_M2 = cache_M2['rtr']

    # --- Get Total Masses (repetitive block, consider refactoring) ---
    try:
        r_tr_1 = float(rtr_interp_M1(ai))
        if r_tr_1 <= 0: raise ValueError("Interpolated r_tr_1 non-positive")
    except ValueError:
        r_tr_1 = max(rtr_interp_M1.x[0], 1e-9) if len(rtr_interp_M1.x) > 0 else 1e-9
    M_tot_1 = Menc(r_tr_1, r_tr_1, M1)

    try:
        r_tr_2 = float(rtr_interp_M2(ai))
        if r_tr_2 <= 0: raise ValueError("Interpolated r_tr_2 non-positive")
    except ValueError:
        r_tr_2 = max(rtr_interp_M2.x[0], 1e-9) if len(rtr_interp_M2.x) > 0 else 1e-9
    M_tot_2 = Menc(r_tr_2, r_tr_2, M2)
    # --- End repetitive block ---

    af = calc_af(ai, M1, M2) # Call new function

    # Check if af indicates an unbound state or calculation failed
    if not np.isfinite(af) or af <= 0:
        # print(f"Warning: Cannot calculate jf because af={af} is invalid/unbound.")
        return 0.0 # Angular momentum j is ill-defined

    # --- Full formula for jf ---
    M_i = M_tot_1 + M_tot_2
    if M_i <= 0: return ji # Avoid division by zero
    # Ensure individual masses are non-negative for mu_i calculation
    if M_tot_1 < 0 or M_tot_2 < 0: return ji
    mu_i = (M_tot_1 * M_tot_2) / M_i

    M_f = M1 + M2
    # M_f should be > 0 if M1, M2 > 0
    mu_f = (M1 * M2) / M_f

    # Ensure mu_f is not zero
    if mu_f <= 0:
        # print("Warning: Final reduced mass mu_f <= 0. Returning initial ji.")
        return ji

    # Ensure term under square root is non-negative
    sqrt_term_num = M_i * ai
    sqrt_term_den = M_f * af
    if sqrt_term_den == 0: return 0.0 # Avoid division by zero
    sqrt_term = sqrt_term_num / sqrt_term_den
    if sqrt_term < 0:
         # print(f"Warning: Term under square root in jf calculation is negative ({sqrt_term:.2e}). Returning 0.")
         return 0.0

    jf = ji * (mu_i / mu_f) * np.sqrt(sqrt_term)

    # Ensure jf is physically valid (0 < jf <= 1)
    # Allow jf=0 if ji was 0 or other terms led to it
    return max(0.0, min(jf, 1.0))


def calc_Tf(Ti, ai, M1, M2):
    """Calculates final orbital period Tf."""
    af = calc_af(ai, M1, M2)
    # Handle cases where af is invalid or unbound
    if not np.isfinite(af) or af <= 0 or ai <= 0:
         return np.inf # Period is infinite if unbound or parameters invalid

    # Ensure base of sqrt is non-negative
    ratio = af / ai
    if ratio < 0: return np.inf # Should not happen if af, ai > 0

    # Ensure Ti is valid
    if Ti < 0 or not np.isfinite(Ti): return np.inf

    return Ti * np.sqrt(ratio)

# --- Placeholder definitions for functions needed by get_rtr_interpolator ---
# !!! These MUST BE REPLACED with proper implementations based on unequal mass theory !!!
def a_max(f, M_PBH, withHalo=False):
     """Max semi-major axis. !!! PLACEHOLDER - NEEDS UNEQUAL MASS DEFINITION !!!"""
     print("Warning: Using placeholder 'a_max' function.")
     # Need to decide how M_PBH relates to M1, M2 here. Using M_PBH for both.
     xb = xbar(f, M_PBH, M_PBH) # Uses placeholder xbar
     if alpha == 0 or f <= 0 or xb == 0 : return 0.0
     M_eff = M_PBH # Simplistic
     if withHalo: M_eff += M_halo(z_eq, M_PBH) # Very simplistic halo addition
     return alpha * xbar(f, M_eff, M_eff) * (f * 0.85)**(1.0/3.0) * (lambda_max**(4.0/3.0))

def a_max_with_Halo(f, M_PBH):
     """Max semi-major axis with halo effects. !!! PLACEHOLDER !!!"""
     print("Warning: Using placeholder 'a_max_with_Halo' function.")
     # Needs proper derivation for unequal mass + halos
     M_eff_am = M_PBH + M_halo(z_eq, M_PBH) # Approx mass for a_max calc
     if alpha == 0 or f <= 0 : return 0.0
     xb_am = xbar(f, M_eff_am, M_eff_am) # Use placeholder xbar
     if xb_am == 0: return 0.0
     return alpha * xb_am * (f * 0.85)**(1.0/3.0) * (lambda_max**(4.0/3.0))
