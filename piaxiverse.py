import numpy as np
from piaxi_utils import init_params
from piaxi_numerics import solve_piaxi_system, piaxi_system

## Parameters of model
manual_set = False       # Toggle override mass definitions
unitful_masses = True    # Toggle whether masses are defined in units of [eV] vs. units of mass-ratio [m_unit] (Currently always True)
unitful_k = False        # Toggle whether k values are defined unitfully [eV] vs. units of mass-ratio [m_unit] (Default: False)

## TODO
def main():

    #init_phases()
    #sample_phases()

    return None

trim_masked_arrays = lambda arr: np.array([np.array(arr_i, dtype=float) if len(arr_i) > 0 else np.array([], dtype=float) for arr_i in arr], dtype=object)

## Pi-axion Species Mass Definitions
def define_mass_species(qc, qm, F, e, eps, eps_c, xi):
    m_r = np.array([0., 0., 0., 0., 0.], dtype=float)                   # real neutral
    m_n = np.array([0., 0., 0., 0., 0., 0.], dtype=float)               # complex neutral
    m_c = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float)   # charged
    # Pi-axion Species Labels
    s_l = np.array([np.full_like(m_r, '', dtype=str), np.full_like(m_n, '', dtype=str), np.full_like(m_c, '', dtype=str)], dtype=object)

    ## Real Neutral Masses
    # pi_3
    m_r[0]    = np.sqrt(((qc[0] + qc[1])*qm[0])*F) if 0. not in [qc[0], qc[1]] else 0.
    s_l[0][0] = '$\pi_{3}$'
    # pi_8
    m_r[1]    = np.sqrt(((qc[0] + qc[1])*qm[0] + qc[2]*qm[1])*F) if 0. not in [qc[0], qc[1], qc[2]] else 0.
    s_l[0][1] = '$\pi_{8}$'
    # pi_29
    m_r[2]    = np.sqrt(((qc[3]*qm[1]) + (qc[4]*qm[2]))*F) if 0. not in [qc[3], qc[4]] else 0.
    s_l[0][2] = '$\pi_{29}$'
    # pi_34
    m_r[3]    = np.sqrt(((qc[3]*qm[1]) + ((qc[4] + qc[5])*qm[2]))*F) if 0. not in [qc[3], qc[4], qc[5]] else 0.
    s_l[0][3] = '$\pi_{34}$'
    # pi_35
    m_r[4]    = np.sqrt(((qc[0]+qc[1])*qm[0] + (qc[2]+qc[3])*qm[1] + (qc[4]+qc[5])*qm[2])*F) if 0. not in [qc[0], qc[1], qc[2], qc[3], qc[4], qc[5]] else 0.
    s_l[0][4] = '$\pi_{35}$'

    ## Complex Neutral Masses
    # pi_6  +/- i*pi_7
    m_n[0]    = np.sqrt((qc[1]*qm[0] + qc[2]*qm[1]) * F) if 0. not in [qc[1], qc[2]] else 0.
    s_l[1][0] = '$\pi_6 \pm i\pi_7$'
    # pi_9  +/- i*pi_10
    m_n[1]    = np.sqrt((qc[0]*qm[0] + qc[3]*qm[1]) * F) if 0. not in [qc[0], qc[3]] else 0.
    s_l[1][1] = '$\pi_9 \pm i\pi_{10}$'
    # pi_17 +/- i*pi_18
    m_n[2]    = np.sqrt((qc[1]*qm[0] + qc[4]*qm[2]) * F) if 0. not in [qc[1], qc[4]] else 0.
    s_l[1][2] = '$\pi_{17} \pm i\pi_{18}$'
    # pi_19 +/- i*pi_20
    m_n[3]    = np.sqrt((qc[2]*qm[1] + qc[4]*qm[2]) * F) if 0. not in [qc[2], qc[4]] else 0.
    s_l[1][3] = '$\pi_{19} \pm i\pi_{20}$'
    # pi_21 +/- i*pi_22
    m_n[4]    = np.sqrt((qc[0]*qm[0] + qc[5]*qm[2]) * F) if 0. not in [qc[0], qc[5]] else 0.
    s_l[1][4] = '$\pi_{21} \pm i\pi_{22}$'
    # pi_30 +/- i*pi_31
    m_n[5]    = np.sqrt((qc[3]*qm[1] + qc[5]*qm[2]) * F) if 0. not in [qc[3], qc[5]] else 0.
    s_l[1][5] = '$\pi_{30} \pm i\pi_{31}$'

    ## Charged Masses
    # pi_1  +/- i*pi_2
    m_c[0]    = np.sqrt((qc[0]*qm[0] + qc[1]*qm[0])*F + 2*xi[0]*(e*eps*F)**2) if 0. not in [qc[0], qc[1]] and abs(eps) < 0.1 else 0.
    s_l[2][0] = '$\pi_1 \pm i\pi_2$'
    # pi_4  +/- i*pi_5
    m_c[1]    = np.sqrt((qc[0]*qm[0] + qc[2]*qm[1])*F + 2*xi[1]*(e*eps*F)**2) if 0. not in [qc[0], qc[2]] and abs(eps) < 0.1 else 0.
    s_l[2][1] = '$\pi_4 \pm i\pi_5$'
    # pi_15 +/- i*pi_16
    m_c[2]    = np.sqrt((qc[0]*qm[0] + qc[4]*qm[2])*F + 2*xi[2]*(e*eps*F)**2) if 0. not in [qc[0], qc[4]] and abs(eps) < 0.1 else 0.
    s_l[2][2] = '$\pi_{15} \pm i\pi_{16}$'
    # pi_11 +/- i*pi_12
    m_c[3]    = np.sqrt((qc[1]*qm[0] + qc[3]*qm[2])*F + 2*xi[3]*(e*eps*F)**2) if 0. not in [qc[1], qc[3]] and abs(eps) < 0.1 else 0.
    s_l[2][3] = '$\pi_{11} \pm i\pi_{12}$'
    # pi_23 +/- i*pi_24
    m_c[4]    = np.sqrt((qc[1]*qm[0] + qc[5]*qm[2])*F + 2*xi[4]*(e*eps*F)**2) if 0. not in [qc[1], qc[5]] and abs(eps) < 0.1 else 0.
    s_l[2][4] = '$\pi_{23} \pm i\pi_{24}$'
    # pi_13 +/- i*pi_14
    m_c[5]    = np.sqrt((qc[2]*qm[1] + qc[3]*qm[1])*F + 2*xi[5]*(e*eps*F)**2) if 0. not in [qc[2], qc[3]] and abs(eps) < 0.1 else 0.
    s_l[2][5] = '$\pi_{13} \pm i\pi_{14}$'
    # pi_25 +/- i*pi_26
    m_c[6]    = np.sqrt((qc[2]*qm[1] + qc[5]*qm[2])*F + 2*xi[6]*(e*eps*F)**2) if 0. not in [qc[2], qc[5]] and abs(eps) < 0.1 else 0.
    s_l[2][6] = '$\pi_{25} \pm i\pi_{26}$'
    # pi_27 +/- i*pi_28
    m_c[7]    = np.sqrt((qc[3]*qm[1] + qc[4]*qm[2])*F + 2*xi[7]*(e*eps*F)**2) if 0. not in [qc[3], qc[4]] and abs(eps) < 0.1 else 0.
    s_l[2][7] = '$\pi_{27} \pm i\pi_{28}$'
    # pi_32 +/- i*pi_33
    m_c[8]    = np.sqrt((qc[4]*qm[2] + qc[5]*qm[2])*F + 2*xi[8]*(e*eps*F)**2) if 0. not in [qc[4], qc[5]] and abs(eps) < 0.1 else 0.
    s_l[2][8] = '$\pi_{32} \pm i\pi_{33}$'

    # Mask zero-valued / disabled species in arrays
    m_r = np.ma.masked_where(m_r == 0.0, m_r, copy=True)
    m_n = np.ma.masked_where(m_n == 0.0, m_n, copy=True)
    m_c = np.ma.masked_where(m_c == 0.0, m_c, copy=True)
    r_mask = np.ma.getmask(m_r)
    n_mask = np.ma.getmask(m_n)
    c_mask = np.ma.getmask(m_c)
    masks = np.array([r_mask, n_mask, c_mask], dtype=object)
    N_r = m_r.count()
    N_n = m_n.count()
    N_c = m_c.count()
    counts = np.array([N_r, N_n, N_c])
    
    return m_r, m_n, m_c, counts, masks

def init_masses(m_r: np.ma, m_n: np.ma, m_c: np.ma, natural_units=True, c=1, verbosity=0):

    #m_unit = np.min([np.min(m_i) for m_i in (m_r.compressed(), m_n.compressed(), m_c.compressed()) if len(m_i) > 0])

    m = np.array([m_r.compressed(), m_n.compressed(), m_c.compressed()], dtype=object, copy=True)
    m_unit = np.min([np.min(m_i) for m_i in m if len(m_i) > 0])
    m_raw = m

    m *= (1. if unitful_masses else 1./m_unit) # Ensure m is provided in desired units
    if unitful_masses and not natural_units: 
        m      *= (1./c**2) # WIP
        m_unit *= (1./c**2)

    if verbosity > 2:
        print('m_unit:  ', m_unit)
        if verbosity > 3:
            print('m (raw):\n', m_raw)
            print('m (out):\n', m)
        else:
            print('m:\n', m)

    return m, m_unit

def init_densities(masks, p_t, normalized_subdens=True, densities_in=None):
    ## local DM densities for each species, assume equal mix for now.
    # TODO: More granular / nontrivial distribution of densities? Spacial dependence? Sampling?
    N_r, N_n, N_c = [len(mask) for mask in masks]
    
    if normalized_subdens:
        p_loc = p_t/3.
    else:
        p_loc = p_t

    ## Accept more specific density profiles
    if densities_in != None:
        if all([len(dens_in) <= 1 for dens_in in densities_in]):
            p_r  = np.ma.masked_where(masks[0], np.full(densities_in[0], dtype=float), copy=True)
            p_n  = np.ma.masked_where(masks[1], np.full(densities_in[1], dtype=float), copy=True)
            p_c  = np.ma.masked_where(masks[2], np.full(densities_in[2], dtype=float), copy=True)
            np.ma.set_fill_value(p_r, 0.0)
            np.ma.set_fill_value(p_n, 0.0)
            np.ma.set_fill_value(p_c, 0.0)

            p = np.array([p_r.compressed()*p_loc, p_n.compressed()*p_loc, p_c.compressed()*p_loc], dtype=object, copy=True)
        else:
            p_r = np.array(densities_in[0])
            p_n = np.array(densities_in[1])
            p_c = np.array(densities_in[2])
            
            p = np.array([p_r*p_loc, p_n*p_loc, p_c*p_loc], dtype=object)
    else:
        p_r  = np.ma.masked_where(masks[0], np.full(N_r, 1./max(1., N_r), dtype=float), copy=True)
        p_n  = np.ma.masked_where(masks[1], np.full(N_n, 1./max(1., N_n), dtype=float), copy=True)
        p_c  = np.ma.masked_where(masks[2], np.full(N_c, 1./max(1., N_c), dtype=float), copy=True)
        #p_r  = np.ma.masked_where(masks[0], np.full(N_r, 1./max(1., N_r), dtype=float), copy=True)
        #p_n  = np.ma.masked_where(masks[1], np.full(N_n, 1./max(1., N_n), dtype=float), copy=True)
        #p_c  = np.ma.masked_where(masks[2], np.full(N_c, 1./max(1., N_c), dtype=float), copy=True)
        #np.ma.set_fill_value(p_r, 0.0)
        #np.ma.set_fill_value(p_n, 0.0)
        #np.ma.set_fill_value(p_c, 0.0)

        p = np.array([p_r.compressed()*p_loc, p_n.compressed()*p_loc, p_c.compressed()*p_loc], dtype=object, copy=True)

    return p

## Define (initial) pi-axion dark matter mass-amplitudes for each species, optional units of [eV/c]
def init_amplitudes(m, p, m_unit=1, mass_units=True, natural_units=True, unitful_amps=unitful_masses, rescale_amps=False, h=1, c=1, verbosity=0):
    amps = np.array([np.array([np.sqrt(2 * p[s][i]) / m[s][i] if np.abs(m[s][i]) > 0. else 0. for i in range(len(m[s]))]) for s in range(len(m))], dtype=object, copy=True)
    amps_raw = np.copy(amps)

    #rescale_amps = mass_units if unitful_amps else not(mass_units)
    # normalize/rescale amplitudes by dividing by amp of pi_0?
    amps_to_units = np.sqrt((h/c)**3) # (TODO: WIP, double check these units)
    if rescale_amps:
        if unitful_amps:
            amps *= (1. / m_unit)          # convert to units of [m_unit] instead of [eV]
        else:
            amps *= m_unit                 # convert to units of [eV] instead of [m_unit]
            if not(natural_units):
                amps *= amps_to_units
    else:
        if unitful_amps and not(natural_units):
            amps *= amps_to_units

    if verbosity > 3:
        print('amps (raw):\n', amps_raw)
        print('amps (out):\n', amps)
    elif verbosity > 0:
        print('amps:\n', amps)
    return amps

# Sample global and local phases from normal distribution, between 0 and 2pi
def init_phases(masks, rng=None, sample_delta=True, sample_Theta=True, delta_in=None, Theta_in=None, cosine_form=True, verbosity=0):
    #global d, Th
    N_r, N_n, N_c = [len(mask) for mask in masks]
    ## local phases for each species (to be sampled)
    if delta_in == None:
        d_raw = np.array([np.ma.masked_where(masks[0], np.zeros(N_r, dtype=float)).compressed(),
                          np.ma.masked_where(masks[1], np.zeros(N_n, dtype=float)).compressed(),
                          np.ma.masked_where(masks[2], np.zeros(N_c, dtype=float)).compressed()], dtype=object)
    else:
        d_raw = delta_in
    d = d_raw

    ## global phase for neutral complex species (to be sampled)
    if Theta_in == None:
        Th_raw = np.array([np.ma.masked_where(masks[0], np.zeros(N_r, dtype=float)).compressed(),
                           np.ma.masked_where(masks[1], np.zeros(N_n, dtype=float)).compressed(),
                           np.ma.masked_where(masks[2], np.zeros(N_c, dtype=float)).compressed()], dtype=object)
    else:
        Th_raw = Theta_in
    Th = Th_raw

    if verbosity > 3:
        print('delta (raw):\n', d_raw)
        print('Theta (raw):\n', Th_raw)

    d, Th = sample_phases(rng, d, Th, sample_delta, sample_Theta, verbosity=verbosity)

    return d, Th

mu  = np.pi      # mean
sig = np.pi / 3  # standard deviation
def sample_phases(rng, d_in, Th_in, sample_delta=True, sample_Theta=True, mean_delta=mu, mean_theta=mu, stdev_delta=sig, stdev_theta=sig, cosine_form=True, verbosity=0):

    mu_delta  = mu  if mean_delta  is None else mean_delta
    mu_theta  = mu  if mean_theta  is None else mean_theta
    sig_delta = sig if stdev_delta is None else stdev_delta
    sig_theta = sig if stdev_theta is None else stdev_theta

    # Modulo 2pi to ensure value is within range
    shift_val = np.pi if cosine_form else 0.
    if sample_delta:
        #d  = np.array([np.ma.masked_where(mask[i], np.mod(rng.normal(mu_delta, sig_delta, len(mask)) + shift_val, 2*np.pi)).compressed() for i,mask in enumerate(masks)], dtype=object)
        d  = np.array([np.mod(rng.normal(mu_delta, sig_delta, len(d_i)) + shift_val, 2*np.pi) for d_i in d_in], dtype=object)
    else:
        d = d_in
    if sample_Theta:
        #Th = np.array([np.ma.masked_where(mask[i], np.mod(rng.normal(mu_theta, sig_theta, len(mask)) + shift_val, 2*np.pi)).compressed() for i,mask in enumerate(masks)], dtype=object)
        Th = np.array([np.mod(rng.normal(mu_theta, sig_theta, len(Th_i)) + shift_val, 2*np.pi) for Th_i in Th_in], dtype=object)
    else:
        Th = Th_in

    if verbosity > 3:
        print('Sample delta?  ', sample_delta)
        if sample_delta: print('  mu: %.2f  |  sigma: %.2f' % (mu_delta, sig_delta))
        print('delta (out):\n', d)
        print('Sample Theta?  ', sample_Theta)
        if sample_Theta: print('  mu: %.2f  |  sigma: %.2f' % (mu_theta, sig_theta))
        print('Theta (out):\n', Th)
    elif verbosity > 0:
        print('delta:\n', d)
        print('Theta:\n', Th)

    return d, Th