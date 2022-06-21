import numpy as np
from scipy.optimize import newton
import scipy.linalg as sla
import matplotlib.pyplot as plt
import lmfit


def unwrap_phase(data, is_complex=True):
    """Returns continuous phase of complex data or angular data.

    Assumes data is ordered with relation to the frequencies swept
    to create it. Also assumes data is not noisy. If provided data is complex
    it assumes full I/Q complex data was provided. If provided data is real it
    assumes phases were provided.
    """
    # Finds correct phases of complex_data in range (-pi, pi)
    wrapped_phases = np.arctan2(np.imag(data), np.real(data)) if is_complex else data

    unwrapped_phases = wrapped_phases
    for i in range(unwrapped_phases.size - 1):
        phase_diff = unwrapped_phases[i + 1] - unwrapped_phases[i]
        # Check if phase jump is at this point
        if np.abs(phase_diff) > np.pi / 2:
            # Find out if phase jump was clockwise or counterclockwise
            diff_sign = np.sign(phase_diff)
            # Add or subtract 2*pi to all subsequent points
            unwrapped_phases[(i + 1) :] -= 2 * np.pi * diff_sign
    return unwrapped_phases


def algebraic_circle_fit(complex_data, verbose=False, eps=1e-7):
    """Algebraically find best fit parameterization of cdata to a circle.

    Uses Pratt's Approximation to Gradient-Weighted Algebraic Fits described
    in Chernov & Lesort, Journal of Mathematical Imaging and Vision, 2005,
    to fit a circle in the complex plane by mapping it to an eigenvalue
    problem. Uses scipy.linalg.eig for diagonalization.

    Parameters:
    -----------
    complex_data ([complex, ]): Iterable of complex data representing a
        (possibly noisy) circle.

    Keyword Arguments:
    ------------------
    verbose (bool): Whether or not to print summary of circle fit procedure.
    eps (float): Tolerance for considering a singular value to be zero in
        this algorithm's linear algebra calculations.

    Returns:
    --------
    [float, ]: List of [A, B, C, D] parameters describing a circle in the
        complex plane as: A(x^2 + y^2) + Bx + Cy + D = 0, constrained by
        B^2 + C^2 - 4AD = 1, where (x, y) = (Re[cdata], Im[cdata]).
        These correspond to the radius 'r' of the circle and its center
        position xc + 1j*yc as:
            xc = -B/2A
            yc = -C/2A
            r = 1/2|A|
    """
    x = np.real(complex_data)
    y = np.imag(complex_data)
    z = x * x + y * y  # Squared magnitude of complex data

    # Circular constraint matrix (eqn. 7 Probst)
    B = np.zeros((4, 4))
    B[1, 1] = B[2, 2] = 1
    B[0, 3] = B[3, 0] = -2

    # Calculate 'moments' or 'weights' of data:
    M = np.zeros((4, 4))
    circle_data = np.array([z, x, y, np.array([1] * len(complex_data))])
    M = circle_data @ circle_data.T

    # Solve generalized eigenvalue problem for smallest positive eigval
    # using Newton's method. This is guaranteed with x_guess=0 to return
    # the smallest non-negative eigenvalue (mentioned in Probst).
    eta = newton(lambda x: np.linalg.det(M - x * B), 0)
    _, s, Vh = sla.svd(M - eta * B)
    vec = Vh[s <= eps, :].T
    normalization_param = vec.T @ B @ vec
    vec *= np.sign(normalization_param) / np.sqrt(np.abs(normalization_param))
    if verbose:
        print("eta = " + str(eta))
        print("eigenvector = " + str(vec))
        print("(M - eta*B)*eigenvector = " + str((M - eta * B) @ vec))
        print(
            "Constraint condition B^2+C^2-4AD = "
            + str(vec[1] ** 2 + vec[2] ** 2 - 4 * vec[0] * vec[3])
        )
    return vec


def circle_fit_square_diff(complex_data):
    """Sum of square deviations of data from an algebraically fit circle."""
    # Algebraically fit data to a circle
    fit_result = algebraic_circle_fit(complex_data)
    # Calculate radial deviations of data from circle center
    x_dev = np.real(complex_data) + 0.5 * fit_result[1] / fit_result[0]
    y_dev = np.imag(complex_data) + 0.5 * fit_result[2] / fit_result[0]
    num_pts = len(complex_data)
    r = 0.5 / np.abs(fit_result[0])

    # Compare distance of data points from fit circle center with fit radius
    return r**2 - x_dev * x_dev - y_dev * y_dev


def fit_cable_delay(freqs, complex_data, verbose=True, return_full_result=False):
    """Find linear (in frequency) phase offsetting complex data."""
    # First find an initial guess of the cable delay (phase slope and offset), by
    # conducting a linear regression fit of the phase data
    phases = unwrap_phase(complex_data)
    freq_avg = np.mean(freqs)
    phase_avg = np.mean(phases)
    freq_deviations = freqs - freq_avg
    phase_slope = np.sum(freq_deviations * (phases - phase_avg))
    denominator = np.dot(freq_deviations, freq_deviations)
    phase_slope /= np.dot(freq_deviations, freq_deviations)
    phase_offset = phase_avg - phase_slope * freq_avg

    # Initialize phase_slope Parameter() using linear regression fit
    # result as guess
    params = lmfit.Parameters()
    params.add("phase_slope", value=phase_slope, vary=True, min=-1e-3, max=1e-3)
    # params.add('phase_offset', value=phase_offset, vary=True, min=0, max=2*np.pi)

    # Define function to be minimized
    # (deviations of data from fit circle edge)

    # Algebraically fit data to a circle
    fit_result = algebraic_circle_fit(complex_data)
    # Calculate radial deviations of data from circle center
    num_pts = len(complex_data)
    r = 0.5 / np.abs(fit_result[0])

    def residuals(parameters, freqs, complex_data):
        phase_slope = parameters["phase_slope"].value
        # phase_offset = parameters['phase_offset'].value
        unrolled_data = complex_data * np.exp(-1j * (phase_slope * freqs))
        return circle_fit_square_diff(unrolled_data)

    fit_result = lmfit.minimize(residuals, params, args=(freqs, complex_data))
    if verbose:
        print(lmfit.fit_report(fit_result))
        phase_slope = fit_result.params["phase_slope"].value
        unrolled_data = complex_data * np.exp(-1j * phase_slope * freqs)
        fig, axs = plt.subplots(1, 2)
        ax = axs[0]
        ax.plot(freqs, unwrap_phase(complex_data), "k-")
        ax.plot(freqs, unwrap_phase(unrolled_data), "g-")
        ax.set_title("Unrolled data (green) and original data")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Unwrapped Phase (rad)")

        ax = axs[1]
        ax.plot(np.real(unrolled_data), np.imag(unrolled_data), "r-")
        ax.axhline(0)
        ax.axvline(0)
        ax.set_title("Unrolled data")
        ax.set_xlabel("Re[S21]")
        ax.set_ylabel("Im[S21]")

        plt.show()
    if return_full_result:
        return fit_result
    return fit_result.params["phase_slope"].value


def centered_phase(phase):
    """Adjusts a phase to be in the domain (-pi, pi)."""
    is_scalar = np.isscalar(phase)
    if is_scalar:
        phase_array = np.array([phase])
    else:
        phase_array = np.array(phase)
    reduced_phase = np.mod(phase_array, 2 * np.pi)
    reduced_phase[reduced_phase > np.pi] -= 2 * np.pi
    reduced_phase[reduced_phase < -np.pi] += 2 * np.pi
    if is_scalar:
        return reduced_phase[0]
    return reduced_phase


def make_scattering_realistic(
    S, freqs, freq_offset=0, amp_offset=1, amp_slope=0, phase_offset=0, phase_slope=0
):
    norm = 1 if freq_offset == 0 else freq_offset
    realistic_S = (
        amp_offset * (1 + (freqs - freq_offset) / norm) * S
    )  # apply amplitude offsets
    realistic_S *= np.exp(
        1j * (phase_offset + phase_slope * (freqs - freq_offset) / norm)
    )
    return realistic_S
