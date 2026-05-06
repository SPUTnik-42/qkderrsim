import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.ticker import ScalarFormatter

class Source:
    """
    Represents the photon source.
    Ideally a single photon source (q=1).
    
    Attributes:
        freq (float): Repetition rate in Hz.
        mu (float): Mean photon number (1.0 for ideal SPS).
        q (float): Preparation quality parameter (1.0 for ideal qubits).
        e_mis (float): Intrinsic misalignment error probability.
    """
    def __init__(self, freq=1e7, mean_photon_num=1.0, q=1.0, alignment_error=0.0):
        self.freq = freq
        self.mu = mean_photon_num
        self.q = q
        self.e_mis = alignment_error

class Channel:
    """
    Represents the fiber optic channel.
    
    Attributes:
        L (float): Length of the fiber in km.
        alpha (float): Attenuation coefficient in dB/km.
    """
    def __init__(self, length_km=0.0, att_db_km=0.2, channel_mode='fiber',atmos_attenuation=0.2,
                transmitter_diameter=0.1, receiver_diameter=0.3, beam_divergence=0.001):
        self.L = length_km
        self.alpha = att_db_km
        self.atmospheric_attenuation = atmos_attenuation
        self.transmitter_diameter = transmitter_diameter
        self.receiver_diameter = receiver_diameter
        self.beam_divergence = beam_divergence
        self.channel_mode = channel_mode

    @property
    def transmittance(self):
        if self.channel_mode == 'fiber':
            """Calculates channel transmittance T = 10^(-alpha * L / 10)."""
            return 10 ** (-self.alpha * self.L / 10.0)
        if self.channel_mode == 'fso':
            beam_diameter_at_receiver = self.transmitter_diameter + (self.L * 1000 * self.beam_divergence)
            geo_factor = min(1.0, (self.receiver_diameter / beam_diameter_at_receiver)**2)

            # Calculate atmospheric loss factor
            atmos_loss = np.exp(-self.atmospheric_attenuation * self.L)
            
            # Calculate overall transmission efficiency
            total_efficiency =  geo_factor * atmos_loss
            return min(1.0, max(0.0, total_efficiency))  # Ensure efficiency is between 0 and 1


class Detector:
    """
    Represents the single photon detector.
    
    Attributes:
        eta (float): Detection efficiency.
        pd (float): Dark count probability per gate.
    """
    def __init__(self, efficiency=0.1, dark_count_rate=1e6, time_window=1e-9):
        self.eta = efficiency
        self.pd = 1 - np.exp(-dark_count_rate * time_window)

class Protocol:
    """
    Implements the 'Tight Finite-Key Analysis for Quantum Cryptography' 
    for Asymmetric BB84.
    
    Formulas extracted from context:
    
    Eq (1) Key Length:
    ell <= n * (q - h(Q_tol + mu)) - leak_EC - log2(2 / (es^2 * ec))
    
    where:
    mu = sqrt( (n + k)/(n * k) * (k+1)/k * ln(4/es) )
    
    Eq (2) Expected Secret Key Rate:
    r = (1 - er) * ell / M
    """
    def __init__(self, source, channel, detector, 
                 epsilon_sec=1e-10, epsilon_cor=1e-10, f_ec=1.16):
        self.source = source
        self.channel = channel
        self.detector = detector
        self.es = epsilon_sec
        self.ec = epsilon_cor
        self.xi = f_ec  # Error correction leakage efficiency (f_ec > 1.0)
    
    def h(self, x):
        """Binary entropy function: -x log2(x) - (1-x) log2(1-x)."""
        if x >= 0.5: return 1.0
        if x <= 0: return 0.0
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    def calculate_system_params(self):
        """
        Calculates the basic experimental parameters.
        
        Returns:
            p_click (float): Probability of a detection event per pulse.
            Q_exp (float): Expected Quantum Bit Error Rate (QBER).
            q_1 (float): Fraction of detection events from single-photon emissions.
        """
        T = self.channel.transmittance
        eta = self.detector.eta
        mu = self.source.mu
        
        # Exact Poissonian probability of signal click
        p_sig = 1.0 - np.exp(-mu * T * eta)
        
        # Probability of dark count (2 detectors)
        p_dark = 2 * self.detector.pd
        
        # Total click probability
        p_click = p_sig + p_dark - (p_sig * p_dark)
        
        if p_click <= 0:
            return 0.0, 0.5, 0.0

        # QBER Calculation
        err_sig = p_sig * self.source.e_mis
        err_dark = p_dark * 0.5
        
        Q_exp = (err_sig + err_dark) / p_click
        
        # Multi-photon penalty (PNS bounding): calculate fraction of secure single photons
        Q_1 = (mu * np.exp(-mu)) * (T * eta)
        q_1 = Q_1 / p_click
        
        return p_click, Q_exp, q_1
    
    def _optimize_params(self, N, p_click, Q_exp, q_1, initial_guess=None):
        """
        Internal optimization function with improved stability.
        """
        
        # Scaling factor to help optimizer with small numbers
        SCALE = 1e7

        def objective(params):
            k_frac, delta = params
            
            # Softer constraint handling near boundaries
            penalty = 0.0
            
            # Apply soft penalties instead of hard cutoffs
            if k_frac < 0.001:
                penalty += 1e6 * (0.001 - k_frac)**2
            if k_frac > 0.95:
                penalty += 1e6 * (k_frac - 0.95)**2
            if delta < 0.0001:
                penalty += 1e6 * (0.0001 - delta)**2
            if delta > 0.3:
                penalty += 1e6 * (delta - 0.3)**2
                
            k = k_frac * N
            n = N - k
            
            if k < 1 or n < 1: 
                return 1e9 + penalty
            
            # 1. Calculate mu
            term1 = N / (n * k)
            term2 = (k + 1) / k
            term3 = np.log(4 / self.es)
            mu = np.sqrt(term1 * term2 * term3)
            
            Q_tol = Q_exp + delta
            
            # Softer penalty for approaching 0.5
            if Q_tol + mu >= 0.48:
                penalty += 1e7 * ((Q_tol + mu) - 0.48)**2
                if Q_tol + mu >= 0.5:
                    return 1e9 + penalty
            
            # 2. Calculate Robustness (er)
            er = np.exp(-2 * k * (delta**2))
            
            if er > 0.99: 
                penalty += 1e6 * (er - 0.99)**2
            
            # 3. Calculate Key Length (ell)
            leak_ec = n * self.xi * self.h(Q_tol)
            # Use dynamically calculated single-photon fraction q_1 to penalize high mu
            frac_q = q_1 if self.source.q == 1.0 else self.source.q
            term_privacy = n * (frac_q - self.h(Q_tol + mu))
            term_security = np.log2(2 / (self.es**2 * self.ec))
            
            ell = term_privacy - leak_ec - term_security
            
            if ell <= 0: 
                return 1e9 + penalty + 1e6 * abs(ell)
            
            # 4. Calculate Expected Rate
            p_x = 1.0 / (1.0 + np.sqrt(k/n))
            M = n / (p_click * p_x**2)
            
            skr_pulse = (1 - er) * ell / M
            
            # Add penalty to the objective
            return -skr_pulse * SCALE + penalty

        # Use provided initial guess or default
        x0 = initial_guess if initial_guess is not None else [0.1, 0.02]
        
        # Bounds
        bounds = [(0.0001, 0.95), (0.0001, 0.3)]
        
        # Try multiple optimization attempts for robustness
        best_result = None
        best_value = 1e9
        
        # First attempt with warm start
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                    options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.fun < best_value:
            best_value = res.fun
            best_result = res
        
        # If first attempt failed, try with different starting points
        if best_value > 0:  # Negative values are good (negative of SKR)
            for k_init in [0.05, 0.15, 0.3]:
                for d_init in [0.01, 0.03, 0.05]:
                    res = minimize(objective, [k_init, d_init], method='L-BFGS-B', 
                                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
                    if res.fun < best_value:
                        best_value = res.fun
                        best_result = res
        
        if best_result is not None and best_value < 1e8:
            rate = -best_result.fun / SCALE
            # Only return positive rates
            if rate > 0:
                return rate, best_result.x
        
        return 0.0, x0


    def calculate_skr(self, N, x0=None):
        """
        Calculates the optimized Secret Key Rate (bits/sec).
        Returns (rate, optimal_params)
        """
        p_click, Q_exp, q_1 = self.calculate_system_params()
        
        # More conservative cutoff with smoother transition
        if Q_exp >= 0.35 or p_click <= 0:
            return 0.0, (x0 if x0 is not None else [0.1, 0.02])
            
        rate_per_pulse, best_params = self._optimize_params(N, p_click, Q_exp, q_1, x0)
        
        # Rate in bits/sec
        return rate_per_pulse * self.source.freq, best_params


    def skr_vs_distance(self, dist_values, fixed_N):
        """
        Improved distance sweep with better warm-start handling.
        """
        rates = []
        original_L = self.channel.L
        
        # Better initialization
        current_params = [0.1, 0.02]
        last_valid_rate = 0.0
        
        for i, L in enumerate(dist_values):
            self.channel.L = L
            r, params = self.calculate_skr(fixed_N, x0=current_params)
            
            # Smoothness check: detect unrealistic jumps
            if i > 0 and r > 0 and last_valid_rate > 0:
                ratio = r / last_valid_rate
                # If rate jumps by more than 10x or drops by more than 90%, be suspicious
                if ratio > 10.0 or ratio < 0.1:
                    # Try re-optimizing with previous parameters
                    r_retry, params_retry = self.calculate_skr(fixed_N, x0=current_params)
                    if r_retry > 0:
                        r, params = r_retry, params_retry
            
            rates.append(r)
            
            # Update warm start only if we have a valid, reasonable result
            if r > 0:
                current_params = params
                last_valid_rate = r
            
        self.channel.L = original_L
        return np.array(rates)

    # def _optimize_params(self, N, p_click, Q_exp, initial_guess=None):
    #     """
    #     Internal optimization function.
    #     """
        
    #     # Scaling factor to help optimizer with small numbers
    #     SCALE = 1e7

    #     def objective(params):
    #         k_frac, delta = params
            
    #         # Constraints check (soft boundaries handled by bounds, hard checks here)
    #         if k_frac <= 0.0001 or k_frac >= 0.999: return 1e9
    #         if delta <= 0.0 or delta >= 0.5: return 1e9
            
    #         k = k_frac * N
    #         n = N - k
            
    #         if k < 1 or n < 1: return 1e9
            
    #         # 1. Calculate mu
    #         term1 = N / (n * k)
    #         term2 = (k + 1) / k
    #         term3 = np.log(4 / self.es)
    #         mu = np.sqrt(term1 * term2 * term3)
            
    #         Q_tol = Q_exp + delta
            
    #         if Q_tol + mu >= 0.5: return 1e9
            
    #         # 2. Calculate Robustness (er)
    #         # er = exp(-2 * k * delta^2)
    #         er = np.exp(-2 * k * (delta**2))
            
    #         if er > 0.99: return 1e9
            
    #         # 3. Calculate Key Length (ell)
    #         leak_ec = n * self.xi * self.h(Q_tol)
    #         term_privacy = n * (self.source.q - self.h(Q_tol + mu))
    #         term_security = np.log2(2 / (self.es**2 * self.ec))
            
    #         ell = term_privacy - leak_ec - term_security
            
    #         if ell <= 0: return 1e9
            
    #         # 4. Calculate Expected Rate
    #         # Optimal probability p_x
    #         p_x = 1.0 / (1.0 + np.sqrt(k/n))
            
    #         # M is total pulses sent
    #         M = n / (p_click * p_x**2)
            
    #         skr_pulse = (1 - er) * ell / M
            
    #         return -skr_pulse * SCALE

    #     # Use provided initial guess or default
    #     x0 = initial_guess if initial_guess is not None else [0.1, 0.02]
        
    #     # Relaxed bounds: k_frac down to 0.0001
    #     bounds = [(0.0001, 0.9), (0.001, 0.2)]
        
    #     res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, tol=1e-9)
        
    #     if res.success:
    #         return (-res.fun / SCALE), res.x
    #     else:
    #         return 0.0, x0

    # def calculate_skr(self, N, x0=None):
    #     """
    #     Calculates the optimized Secret Key Rate (bits/sec).
    #     Returns (rate, optimal_params)
    #     """
    #     p_click, Q_exp = self.calculate_system_params()
        
    #     if Q_exp >= 0.4: # Safety cutoff
    #         return 0.0, x0
            
    #     rate_per_pulse, best_params = self._optimize_params(N, p_click, Q_exp, x0)
        
    #     # Rate in bits/sec
    #     return rate_per_pulse * self.source.freq, best_params

    # def skr_vs_block_size(self, N_values):
    #     rates = []
    #     # Warm start initialization
    #     current_params = [0.1, 0.02] 
        
    #     for N in N_values:
    #         r, params = self.calculate_skr(N, x0=current_params)
    #         rates.append(r)
    #         # Only update warm start if we found a valid key rate
    #         if r > 0:
    #             current_params = params
    #     return np.array(rates)

    # def skr_vs_distance(self, dist_values, fixed_N):
    #     rates = []
    #     original_L = self.channel.L
        
    #     # Warm start initialization
    #     current_params = [0.1, 0.02]
        
    #     for L in dist_values:
    #         self.channel.L = L
    #         r, params = self.calculate_skr(fixed_N, x0=current_params)
    #         rates.append(r)
    #         # Update warm start parameters to follow the curve
    #         if r > 0:
    #             current_params = params
            
    #     self.channel.L = original_L
    #     return np.array(rates)


def main():
    # --- Simulation Parameters ---
    freq = 1e7 
    channel_mode = 'fiber'  # 'fiber' or 'fso'

    source = Source(freq=freq, q=1.0, alignment_error=0.005)
    detector = Detector(efficiency=0.1, dark_count_rate=1e4, time_window=1e-9)
    channel = Channel(att_db_km=0.2, channel_mode=channel_mode)
    
    protocol = Protocol(source, channel, detector, epsilon_sec=1e-9, epsilon_cor=1e-9)
    
    # --- Plot 1: SKR vs Block Size ---
    print("Simulating SKR vs Block Size...")
    channel.L = 20.0 
    N_values = np.logspace(4, 9, 40) # Increased resolution
    
    rates_n = protocol.skr_vs_block_size(N_values)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(N_values, rates_n, 'b-', linewidth=2)
    plt.xlabel('Sifted Block Size $N$')
    plt.ylabel('Secret Key Rate (bits/sec)')
    plt.title(f'SKR vs Block Size (L={channel.L}km, Freq={freq/1e6} MHz ,{channel.channel_mode})')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()
    
    # --- Plot 2: SKR vs Distance ---
    print("Simulating SKR vs Distance...")
    fixed_N = 1e8 
    distances = np.linspace(0, 250, 100)
    
    rates_dist = protocol.skr_vs_distance(distances, fixed_N)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(distances, rates_dist, 'r-', linewidth=2)
    plt.xlabel('Distance (km)')
    plt.ylabel(f'Secret Key Rate (bits/sec)')
    plt.title(f'SKR vs Distance ($N=10^8$, Freq={freq/1e6} MHz ,{channel.channel_mode})')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    plt.show()

if __name__ == "__main__":
    main()