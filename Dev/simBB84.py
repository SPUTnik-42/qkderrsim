import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import binom
import math

matplotlib.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",   # Use Computer Modern for math
})


class WeakCoherentSource:
    """
    Simulates a weak coherent photon source for BB84 protocol.
    """
    def __init__(self, mu):
        """
        Initialize the photon source with mean photon number mu.
        
        Args:
            mu (float): Mean photon number per pulse
        """
        self.mu = mu
    
    def photon_distribution(self, n_max=20):
        """
        Calculate the Poisson photon number distribution for weak coherent states.
        
        Returns:
            np.array: Probability distribution of photon numbers
        """
        n_values = np.arange(n_max + 1)
        # Poisson distribution: P(n) = e^(-μ) * μ^n / n!
        p_n = np.exp(-self.mu) * (self.mu**n_values) / np.array([math.factorial(n) for n in n_values])
        return p_n


class Channel:
    """
    Represents the quantum channel between Alice and Bob.
    Includes both fiber and FSO (Free Space Optical) channel modeling options.
    """
    def __init__(self, base_efficiency, distance=0, attenuation=0.2, mode="fiber", atmos_attenuation=0.2,
                 transmitter_diameter=0.1, receiver_diameter=0.3, beam_divergence=0.001,
                 misalignment_base=0.015, misalignment_factor=0.0002):
        """
        Initialize the channel with distance-dependent efficiency.
        
        Args:
            base_efficiency (float): Base channel transmission efficiency without distance (0-1)
            distance (float): Channel distance in kilometers
            attenuation (float): Fiber attenuation coefficient in dB/km
            mode (str): Channel mode - "fiber" or "fso"
        """
        self.base_efficiency = base_efficiency
        self.distance = distance
        self.attenuation = attenuation
        
        # FSO specific parameters with default values
        self.transmitter_efficiency = transmitter_diameter  # Efficiency of transmitter optics
        self.receiver_efficiency = receiver_diameter         # Efficiency of receiver optics
        self.transmitter_diameter = transmitter_diameter     # Diameter of transmitter aperture in meters
        self.receiver_diameter = receiver_diameter           # Diameter of receiver aperture in meters
        self.atmos_attenuation = atmos_attenuation           # Atmospheric attenuation coefficient in dB/km
        self.beam_divergence = beam_divergence       # Beam divergence angle in radians
        # Optical misalignment that increases with distance
        self.misalignment_base = misalignment_base           # 1.5% base misalignment error
        self.misalignment_factor = misalignment_factor       # Increase per km

        # Set mode and calculate efficiency
        self.mode = mode
        self.efficiency = self.calculate_efficiency()
    
    def calculate_efficiency(self):
        """
        Calculate the actual channel efficiency based on distance and mode.
        
        Returns:
            float: Actual channel efficiency after distance attenuation
        """
        if self.mode == "fiber":
            return self._calculate_fiber_efficiency()
        elif self.mode == "fso":
            return self._calculate_fso_efficiency()
        else:
            raise ValueError(f"Unknown channel mode: {self.mode}")
    
    def _calculate_fiber_efficiency(self):
        """
        Calculate efficiency for fiber optic channel.
        
        Returns:
            float: Channel efficiency for fiber
        """
        # Calculate attenuation in dB
        attenuation_db = self.distance * self.attenuation
        
        # Convert to transmission efficiency: 10^(-attenuation_db/10)
        distance_factor = 10**(-attenuation_db/10)
        
        # Total efficiency is base efficiency times distance factor
        return self.base_efficiency * distance_factor
    
    def _calculate_fso_efficiency(self):
        """
        Calculate efficiency for FSO channel based on provided model.
        
        Returns:
            float: Channel efficiency for FSO
        """
        # For zero distance, return direct efficiency without atmospheric effects
        if self.distance <= 1e-6:  # Effectively zero
            return self.base_efficiency 
    
        # Calculate geometrical loss factor
        beam_diameter_at_receiver = self.transmitter_diameter + (self.distance * 1000 * self.beam_divergence)
        geo_factor = min(1.0, (self.receiver_diameter / beam_diameter_at_receiver)**2)

        #calculate atmospheric loss factor , beer-lambert law
        atmos_loss = np.exp(-self.atmos_attenuation * self.distance)
        

        
        # Calculate overall transmission efficiency
        total_efficiency = (self.base_efficiency * geo_factor * atmos_loss)
        
        return min(1.0, max(0.0, total_efficiency))  # Ensure efficiency is between 0 and 1
    
    def update_distance(self, distance):
        """
        Update the channel distance and recalculate efficiency.
        
        Args:
            distance (float): New channel distance in kilometers
        """
        self.distance = distance
        self.efficiency = self.calculate_efficiency()
    
    def update_mode(self, mode):
        """
        Update the channel mode and recalculate efficiency.
        Default FSO parameters are automatically used when switching to FSO mode.
        
        Args:
            mode (str): New channel mode ("fiber" or "fso")
        """
        if mode not in ["fiber", "fso"]:
            raise ValueError(f"Unsupported channel mode: {mode}. Use 'fiber' or 'fso'.")
            
        self.mode = mode
        self.efficiency = self.calculate_efficiency()
    
    def set_fso_parameters(self, transmitter_diameter=None, receiver_diameter=None, atmos_attenuation=None,
                          beam_divergence=None):
        """
        Update FSO-specific parameters. Only updates the parameters that are provided.
        
        Args:
            transmitter_diameter (float, optional): Diameter of transmitter aperture in meters
            receiver_diameter (float, optional): Diameter of receiver aperture in meters
            beam_divergence (float, optional): Beam divergence angle in radians
        """
        if transmitter_diameter is not None:
            self.transmitter_diameter = transmitter_diameter
        if receiver_diameter is not None:
            self.receiver_diameter = receiver_diameter
        if beam_divergence is not None:
            self.beam_divergence = beam_divergence
        if atmos_attenuation is not None:
            self.atmos_attenuation = atmos_attenuation

        # Recalculate efficiency if in FSO mode
        if self.mode == "fso":
            self.efficiency = self.calculate_efficiency()
    
    def transmission_probability(self, sent_photons, received_photons):
        """
        Calculate probability of receiving photons given sent photons.
        
        Args:
            sent_photons (int): Number of photons sent
            received_photons (int): Number of photons received
            
        Returns:
            float: Probability of receiving the specified number of photons
        """
        if received_photons > sent_photons:
            return 0.0
        
        return binom.pmf(received_photons, sent_photons, self.efficiency)
    
    def calculate_misalignment_error(self):
        """
        Calculate optical misalignment error based on distance.
        
        Returns:
            float: Misalignment error probability (0-1)
        """
        # Error increases with distance but saturates
        return min(0.1, self.misalignment_base + self.misalignment_factor * self.distance)


class Detector:
    """
    Represents a single-photon detector with noise characteristics.
    """
    def __init__(self, efficiency, dark_count_rate, time_window, afterpulsing_prob=0.02, timing_jitter_error=0.01):
        """
        Initialize detector with its characteristics.
        
        Args:
            efficiency (float): Detector efficiency (0-1)
            dark_count_rate (float): Dark count rate in counts per second
            time_window (float): Detection time window in seconds
        """
        self.efficiency = efficiency
        self.dark_count_rate = dark_count_rate
        self.time_window = time_window
        self.p_dark = 1 - np.exp(-dark_count_rate * time_window)
        
        # Detector afterpulsing probability
        self.afterpulsing_prob = afterpulsing_prob

        # Detector timing jitter (as error probability)
        self.timing_jitter_error = timing_jitter_error

    def detect_probability(self, photons):
        """
        Calculate the probability of detection given number of photons.
        
        Args:
            photons (int): Number of photons arriving at detector
            
        Returns:
            float: Probability of detection
        """
        # Probability of at least one photon being detected
        if photons > 0:
            # 1 - probability that none are detected
            p_detect_signal = 1 - (1 - self.efficiency)**photons
            
            # Add saturation effect for multiple photons (models crosstalk and other non-linearities)
            saturation_factor = 1.0
            if photons > 1:
                # Detector saturation for multi-photon pulses
                saturation_factor = 1.0 + 0.02 * (photons - 1)
                
            return min(1.0, p_detect_signal * saturation_factor)
        return 0
    
    def dark_count_probability(self):
        """
        Calculate the probability of a dark count in the detection window.
        
        Returns:
            float: Dark count probability
        """
        return self.p_dark


class BB84Simulator:
    """
    Simulates the BB84 QKD protocol with weak coherent source.
    Supports both fiber and FSO channels.
    """
    def __init__(self, mu, detector_efficiency, channel_base_efficiency,
                 dark_count_rate, time_window, distance=1, attenuation=0.2, 
                 channel_mode="fiber",atmos_attenuation=0.2,
                 transmitter_diameter=0.1, receiver_diameter=0.3, beam_divergence=0.001,
                 misalignment_base=0.015, misalignment_factor=0.0004, ec_eff_factor=1.1, e1_factor=1.05):
        """
        Initialize the BB84 simulator.
        
        Args:
            mu (float): Mean photon number
            detector_efficiency (float): Bob's detector efficiency
            channel_base_efficiency (float): Base efficiency of quantum channel
            dark_count_rate (float): Dark count rate in counts per second
            time_window (float): Detection time window in seconds
            distance (float): Distance between Alice and Bob in kilometers
            attenuation (float): Fiber attenuation coefficient in dB/km
            channel_mode (str): Channel mode - "fiber" or "fso"
            ec_eff_factor (float): Efficiency factor for error correction
            e1_factor (float): error rate for privacy amplification calculation estimates how much 
                worse (or better) the error rate is on single-photon events compared to the total.
        """
        self.source = WeakCoherentSource(mu)
        self.mu = mu
        self.channel = Channel(channel_base_efficiency, distance, attenuation, channel_mode,
                               atmos_attenuation=atmos_attenuation,
                               transmitter_diameter=transmitter_diameter,
                               receiver_diameter=receiver_diameter,
                               misalignment_base=misalignment_base,
                               misalignment_factor=misalignment_factor,
                               beam_divergence=beam_divergence)
        self.detector = Detector(detector_efficiency, dark_count_rate, time_window)
        self.distance = distance
        self.attenuation = attenuation
        self.n_max = 10  # Maximum photon number to consider in calculations
        self.time_window = time_window
        self.repetition_rate = 1e6  # Default pulse rate: 1 MHz
        self.channel_mode = channel_mode
        self.ec_eff_factor = ec_eff_factor  # Efficiency factor for error correction
        self.e1_factor = e1_factor  # Error rate factor for privacy amplification
        # Additional error sources
        self.optical_error_base = 0.01  # Base optical error rate (1%)
        self.multi_photon_error_factor = 0.08  # Additional error per photon for multi-photon pulses
    
    def update_distance(self, distance):
        """
        Update the distance between Alice and Bob and recalculate channel efficiency.
        
        Args:
            distance (float): New distance in kilometers
        """
        self.distance = distance
        self.channel.update_distance(distance)
    
    def update_channel_mode(self, mode):
        """
        Update the channel mode (fiber or FSO).
        
        Args:
            mode (str): New channel mode ("fiber" or "fso")
        """
        self.channel_mode = mode
        self.channel.update_mode(mode)
    
    def set_fso_parameters(self, transmitter_diameter=None, receiver_diameter=None, 
                          beam_divergence=None, pointing_error=None):
        """
        Update FSO-specific parameters in the channel.
        
        Args:
            transmitter_diameter (float, optional): Diameter of transmitter aperture in meters
            receiver_diameter (float, optional): Diameter of receiver aperture in meters
            beam_divergence (float, optional): Beam divergence angle in radians
            wavelength (float, optional): Wavelength in meters
            pointing_error (float, optional): Pointing error in radians
        """
        self.channel.set_fso_parameters(
            transmitter_diameter, receiver_diameter, 
            beam_divergence, pointing_error
        )
    
    def update_mu(self, mu):
        """
        Update the mean photon number.
        
        Args:
            mu (float): New mean photon number
        """
        self.mu = mu
        self.source = WeakCoherentSource(mu)
    
    def calculate_raw_key_rate(self):
        """
        Calculate the raw key rate (before sifting).
        
        Returns:
            float: Raw key rate in bits per pulse
        """
        # Probability that at least one photon is received and detected
        p_distribution = self.source.photon_distribution(self.n_max)
        
        # Raw key rate before sifting - probability of detection per pulse
        raw_rate = 0
        for n in range(1, self.n_max + 1):  # Start from n=1 (at least one photon)
            # Probability that n photons are sent
            p_n = p_distribution[n]
            
            # Calculate probability of receiving and detecting at least one photon
            p_detect_n = 0
            for k in range(1, n + 1):  # k photons reach Bob
                # Probability that k out of n photons reach Bob
                p_trans = self.channel.transmission_probability(n, k)
                # Probability that at least one of k photons is detected
                p_detect = self.detector.detect_probability(k)
                p_detect_n += p_trans * p_detect
            
            raw_rate += p_n * p_detect_n
        
        # Add probability of dark count when no photon is detected
        p_no_photon = p_distribution[0]  # Probability of sending zero photons
        p_dark = self.detector.dark_count_probability()
        raw_rate += p_no_photon * p_dark
        
        # For non-zero photons that aren't detected, there's still a chance of dark count
        for n in range(1, self.n_max + 1):
            p_n = p_distribution[n]
            
            # Probability no photons are detected (includes all transmission possibilities)
            p_no_detect = 0
            for k in range(n + 1):  # 0 to n photons reach Bob
                p_trans = self.channel.transmission_probability(n, k)
                p_not_detect = (1 - self.detector.detect_probability(k))
                p_no_detect += p_trans * p_not_detect
            
            # Add probability of dark count when signal photons aren't detected
            raw_rate += p_n * p_no_detect * p_dark
        
        return raw_rate
    
    def calculate_sifted_key_rate(self):
        """
        Calculate the sifted key rate (after basis reconciliation).
        
        Returns:
            float: Sifted key rate in bits per pulse
        """
        # After basis reconciliation, approximately half of the raw bits remain
        return self.calculate_raw_key_rate() * 0.5
    
    def calculate_quantum_bit_error_rate(self):
        """
        Calculate the quantum bit error rate (QBER) for the BB84 protocol.
        
        Returns:
            float: QBER as a percentage
        """
        p_distribution = self.source.photon_distribution(self.n_max)
        
        # Channel efficiency decreases exponentially with distance
        channel_efficiency = self.channel.efficiency
        
        # Misalignment error inversely proportional to channel efficiency
        # (harder to maintain alignment as signal weakens)
        misalignment_error = self.channel.calculate_misalignment_error() # self.channel.misalignment_base + self.channel.misalignment_factor * self.distance
        
        # Calculate error sources
        p_dark = self.detector.dark_count_probability()
        p_sig_correct = 0
        p_sig_error = 0
        
        # Calculate signal error probabilities
        for n in range(1, self.n_max + 1):
            p_n = p_distribution[n]
            multi_photon_error = self.optical_error_base
            if n > 1:
                multi_photon_error += self.multi_photon_error_factor * (n - 1)
            
            for k in range(1, n + 1):
                p_trans = self.channel.transmission_probability(n, k)
                p_detect = self.detector.detect_probability(k)
                
                # Total optical error without artificial caps
                p_optical_error = misalignment_error + self.detector.timing_jitter_error + multi_photon_error
                
                
                p_sig_correct += p_n * p_trans * p_detect * (1 - p_optical_error)
                p_sig_error += p_n * p_trans * p_detect * p_optical_error
        
        # Dark count errors (50% chance of wrong bit)
        p_dark_error = 0.5 * p_dark * (1 - p_sig_correct - p_sig_error)
        
        # Afterpulsing errors
        p_afterpulse_error = 0.5 * (p_sig_correct + p_sig_error) * self.detector.afterpulsing_prob
        
        # Total error probability
        p_error = p_sig_error + p_dark_error + p_afterpulse_error
        
        # QBER calculation - no artificial scaling needed
        p_detect_total = self.calculate_sifted_key_rate() * 2
        
        # As channel efficiency decreases, dark count contribution naturally increases
        qber = (p_error / p_detect_total) * 100 if p_detect_total > 0 else 100

        return qber

    
    def error_correction_efficiency(self, error_rate):
        """
        Calculate the fraction of bits lost due to error correction.
        
        Args:
            error_rate (float): Error rate (δ)
            
        Returns:
            float: Fraction of bits lost in error correction
        """
        # if error_rate <= 0:
        #     return 0

        # r_ec = ec_eff_factor × h_binary(error_rate)
        r_ec = self.ec_eff_factor * self.h_binary(error_rate)

        return r_ec
    
    def h_binary(self, p):
        """
        Binary entropy function H(p) = -p*log2(p) - (1-p)*log2(1-p).
        
        Args:
            p (float): Probability (0 <= p <= 1)
            
        Returns:
            float: Binary entropy value
        """
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    def calculate_skr(self):
        """
        Calculate the secret key rate (SKR).
        
        Returns:
            float: Secret key rate in bits per second
        """
        # Get photon number distribution
        p_distribution = self.source.photon_distribution(self.n_max)
        
        # Calculate QBER (E_μ)
        qber = self.calculate_quantum_bit_error_rate() / 100  # Convert from percentage to fraction
        
        # Basis reconciliation factor for BB84
        q = 0.5
        
        # Calculate overall gain (Q_μ) - probability of detection per pulse
        Q_mu = self.calculate_raw_key_rate()
        
        # Calculate gain from single-photon pulses (Q_1)
        Q_1 = 0
        p_single = p_distribution[1]  # Probability of single-photon pulses
        for k in range(1, 2):  # For single-photon pulses, only k=1 is possible
            p_trans = self.channel.transmission_probability(1, k)
            p_detect = self.detector.detect_probability(k)
            Q_1 += p_single * p_trans * p_detect
        

        error_correction_term = -Q_mu * self.error_correction_efficiency(qber)
        privacy_amplification_term = Q_1 * (1 - self.h_binary(self.e1_factor * qber))
        
        # Calculate the final secret key rate 
        skr = q * (error_correction_term + privacy_amplification_term) * self.repetition_rate
        
        # No negative key rates
        return max(0, skr)
    
def plot_qber_vs_mu(mu_values=None, time_window=10e-9, distance=50,
                   detector_efficiency=0.15, channel_base_efficiency=1, 
                   dark_count_rate=2000, channel_mode="fiber",
                   fso_params=None, save_fig=False):
    """
    Plot QBER vs mean photon number μ.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        time_window (float, optional): Detection time window in seconds
        distance (float, optional): Distance in kilometers
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
        fso_params (dict, optional): Custom FSO parameters if needed
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.0, 40)
    
    qber_values = []
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            detector_efficiency=detector_efficiency,
            channel_base_efficiency=channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            channel_mode=channel_mode
        )
        
        # Set custom FSO parameters if provided
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
            
        qber = simulator.calculate_quantum_bit_error_rate()
        qber_values.append(qber)
    
    plt.figure(figsize=(12, 8))
    plt.plot(mu_values, qber_values, 'b', linewidth=3.5,label=f'QBER ({channel_mode} channel)')
    #plt.axhline(y=7, color='magenta', linestyle='--', linewidth=3.5, label='QBER 7% Threshold')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)', fontsize=25)
    plt.ylabel('QBER (%)', fontsize=25)
    #plt.title(f'Quantum Bit Error Rate vs Mean Photon Number ({channel_mode.upper()} channel)', fontsize=25)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    if save_fig:
        plt.savefig(f"Figures/BB84/{channel_mode.capitalize()}/qbervsmu{channel_mode.capitalize()}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    
    return mu_values, qber_values


def plot_skr_vs_mu(mu_values=None, time_window=10e-9, distance=50,
                  detector_efficiency=0.15, channel_base_efficiency=1, 
                  dark_count_rate=2000,
                  channel_mode="fiber", fso_params=None, save_fig=False):
    """
    Plot Secret Key Rate vs mean photon number μ.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        time_window (float, optional): Detection time window in seconds
        distance (float, optional): Distance in kilometers
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
        fso_params (dict, optional): Custom FSO parameters if needed
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1, 40)
    
    skr_values = []
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            detector_efficiency=detector_efficiency,
            channel_base_efficiency=channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            channel_mode=channel_mode
        )
        
        # Set custom FSO parameters if provided
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
            
        skr_per_pulse = simulator.calculate_skr()
        skr_per_second = skr_per_pulse # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(12, 8))
    plt.plot(mu_values, skr_values, 'g', linewidth=3.5, label=f'SKR ({channel_mode} channel) ')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)', fontsize=25)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=25)
    #plt.title(f'Secret Key Rate vs Mean Photon Number ({channel_mode.upper()} channel)', fontsize=25)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    if save_fig:
        plt.savefig(f"Figures/BB84/{channel_mode.capitalize()}/skrvsmu{channel_mode.capitalize()}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    
    return mu_values, skr_values


def plot_qber_vs_distance(distance_values=None, time_window=10e-9, mu=0.5,
                         detector_efficiency=0.15, channel_base_efficiency=1, 
                         dark_count_rate=2000, channel_mode="fiber",
                         fso_params=None, save_fig=False):
    """
    Plot QBER vs distance.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        time_window (float, optional): Detection time window in seconds
        mu (float, optional): Mean photon number
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
        fso_params (dict, optional): Custom FSO parameters if needed
    """
    if distance_values is None:
        distance_values = np.linspace(0, 120, 40)
    
    qber_values = []
    
    simulator = BB84Simulator(
        mu=mu,
        detector_efficiency=detector_efficiency,
        channel_base_efficiency=channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        channel_mode=channel_mode
    )
    
    # Set custom FSO parameters if provided
    if channel_mode == "fso" and fso_params is not None:
        simulator.set_fso_parameters(**fso_params)
    
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_quantum_bit_error_rate()
        qber_values.append(qber)
    
    plt.figure(figsize=(12, 8))
    plt.plot(distance_values, qber_values, 'r', linewidth=3.5, label=f'QBER ({channel_mode} channel)')
    plt.grid(True)
    #plt.axhline(y=7, color='orange', linestyle='--', label='QBER 7% Threshold')
    plt.xlabel('Distance (km)', fontsize=25)
    plt.ylabel('QBER (%)', fontsize=25)
    #plt.title(f'Quantum Bit Error Rate vs Distance ({channel_mode.upper()} channel)',fontsize=25)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    if save_fig:
        plt.savefig(f"Figures/BB84/{channel_mode.capitalize()}/qbervsdistance{channel_mode.capitalize()}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    
    return distance_values, qber_values


def plot_skr_vs_distance(distance_values=None, time_window=10e-9, mu=0.1,
                        detector_efficiency=0.15, channel_base_efficiency=1, 
                        dark_count_rate=2000,
                        channel_mode="fiber", fso_params=None, save_fig=False):
    """
    Plot Secret Key Rate vs distance.
    
    Args:
        distance_values (list, optional): List of distance values to simulate in kilometers
        time_window (float, optional): Detection time window in seconds
        mu (float, optional): Mean photon number
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
        fso_params (dict, optional): Custom FSO parameters if needed
    """
    if distance_values is None:
        distance_values = np.linspace(0, 120, 40)
    
    skr_values = []
    
    simulator = BB84Simulator(
        mu=mu,
        detector_efficiency=detector_efficiency,
        channel_base_efficiency=channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        channel_mode=channel_mode
    )
    
    # Set custom FSO parameters if provided
    if channel_mode == "fso" and fso_params is not None:
        simulator.set_fso_parameters(**fso_params)
    
    for distance in distance_values:
        simulator.update_distance(distance)
        skr_per_pulse = simulator.calculate_skr()
        skr_per_second = skr_per_pulse  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(12, 8))
    plt.semilogy(distance_values, skr_values, 'm', linewidth=3.5, label=f'SKR ({channel_mode} channel)')
    plt.grid(True)
    plt.xlabel('Distance (km)', fontsize=25)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=25)
    #plt.title(f'Secret Key Rate vs Distance ({channel_mode.upper()} channel)',fontsize=25)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    if save_fig:
        plt.savefig(f"Figures/BB84/{channel_mode.capitalize()}/skrvsdistance{channel_mode.capitalize()}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    
    return distance_values, skr_values

def plot_combined_skr_qber_vs_distance(time_window=10e-9, mu=0.1,
                                      detector_efficiency=0.15, channel_base_efficiency=1,
                                      dark_count_rate=2000, channel_mode="fiber",
                                      fso_params=None, save_fig=False):
    """
    Plot Secret Key Rate and QBER vs distance on the same graph with twin y-axes.
    
    Args:
        time_window (float, optional): Detection time window in seconds
        mu (float, optional): Mean photon number
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
        fso_params (dict, optional): Custom FSO parameters if needed
    """
    # Set appropriate distance range based on channel mode
    if channel_mode == "fiber":
        distance_values = np.linspace(0, 250, 40)
    elif channel_mode == "fso":  # FSO mode
        distance_values = np.linspace(0, 50, 40)
    
    # Initialize arrays for results
    skr_values = []
    qber_values = []
    
    # Create simulator
    simulator = BB84Simulator(
        mu=mu,
        detector_efficiency=detector_efficiency,
        channel_base_efficiency=channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0,  # Will be updated in the loop
        channel_mode=channel_mode
    )
    
    # Set custom FSO parameters if provided
    if channel_mode == "fso" and fso_params is not None:
        simulator.set_fso_parameters(**fso_params)
    
    # Calculate SKR and QBER for each distance
    for distance in distance_values:
        simulator.update_distance(distance)
        
        # Calculate SKR
        skr_per_second = simulator.calculate_skr()
        skr_values.append(skr_per_second)
        
        # Calculate QBER
        qber = simulator.calculate_quantum_bit_error_rate()
        qber_values.append(qber)
    
    # Create figure with twin y-axes
    fig, ax1 = plt.subplots(figsize=(4,3))
    
    # First axis: SKR (logarithmic)
    color = 'blue'
    ax1.set_xlabel(f'Distance (km) ({channel_mode.capitalize()})')#, fontsize=24)
    ax1.set_ylabel('Secret Key Rate (bits/s)', color=color)#, fontsize=24)
    ax1.semilogy(distance_values, skr_values, color=color,# linewidth=3.5, 
                label=f'SKR ')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(False)#, which="both", ls="-", alpha=0.2)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Second axis: QBER
    ax2 = ax1.twinx()
    color = 'black'
    ax2.set_ylabel('QBER (%)', color=color)#, fontsize=24)
    ax2.plot(distance_values, qber_values, color=color, #linewidth=3.5,
            label=f'QBER')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(False)
    # Add title
    #plt.title(f'Secret Key Rate and QBER vs Distance ({channel_mode.upper()} channel)', 
    #         fontsize=25)
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', 
               fontsize=9)
    #bbox_to_anchor=(0.5, -0.15),
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"Figures/BB84/{channel_mode.capitalize()}/skrvsqbervsdistance{channel_mode.capitalize()}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_combined_skr_qber_vs_mu(mu_values=None, time_window=10e-9, distance=50,
                                detector_efficiency=0.15, channel_base_efficiency=1,
                                dark_count_rate=2000, channel_mode="fiber",
                                fso_params=None, save_fig=False):
    """
    Plot Secret Key Rate and QBER vs mean photon number μ on the same graph with twin y-axes.
    
    Args:
        mu_values (list, optional): List of μ values to simulate
        time_window (float, optional): Detection time window in seconds
        distance (float): Distance in kilometers
        detector_efficiency (float, optional): Detector efficiency
        channel_base_efficiency (float, optional): Base channel efficiency
        dark_count_rate (float, optional): Dark count rate in counts per second
        channel_mode (str, optional): Channel mode - "fiber" or "fso"
        fso_params (dict, optional): Custom FSO parameters if needed
    """
    if mu_values is None:
        mu_values = np.linspace(0.01, 1.0, 40)
    
    # Initialize arrays for results
    skr_values = []
    qber_values = []
    
    # Calculate SKR and QBER for each mu value
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            detector_efficiency=detector_efficiency,
            channel_base_efficiency=channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            channel_mode=channel_mode
        )
        
        # Set custom FSO parameters if provided
        if channel_mode == "fso" and fso_params is not None:
            simulator.set_fso_parameters(**fso_params)
        
        # Calculate SKR
        skr_per_second = simulator.calculate_skr()
        skr_values.append(skr_per_second)
        
        # Calculate QBER
        qber = simulator.calculate_quantum_bit_error_rate()
        qber_values.append(qber)
    
    # Create figure with twin y-axes
    fig, ax1 = plt.subplots(figsize=(4,3))
    
    # First axis: SKR
    color = 'blue'
    ax1.set_xlabel(f'Mean Photon Number (μ) ({channel_mode.capitalize()})')#, fontsize=24)
    ax1.set_ylabel('Secret Key Rate (bits/s)')#, color=color, fontsize=24)
    ax1.plot(mu_values, skr_values, color=color, #linewidth=3.5, 
            label=f'SKR')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.grid(False)#, which="both", ls="-", alpha=0.2)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Second axis: QBER
    ax2 = ax1.twinx()
    color = 'black'
    ax2.set_ylabel('QBER (%)', color=color)#, fontsize=24)
    ax2.plot(mu_values, qber_values, color=color,# linewidth=3.5,
            label=f'QBER')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(False)
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', bbox_to_anchor=(0.98, 0.15),
               fontsize=9)

    
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"Figures/BB84/{channel_mode.capitalize()}/skrvsqbervsmu{channel_mode.capitalize()}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
