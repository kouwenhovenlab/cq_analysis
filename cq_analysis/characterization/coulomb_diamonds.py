import numpy as np
from scipy.signal import find_peaks, convolve2d
from scipy.optimize import curve_fit

def find_diamond_slopes(Vgate, Vbias, G, ignore_large_bias_lines=None,
                    min_slope=0.01, max_slope=100, do_fft=True, correlation_distance=None):
    """
    Analyses the Coulomb diamonds measurements to find the two slopes of linear features
    which indicate quantum dot levels aligning with the two leads.
    
    These slopes can be combined into a lever arm between the sweeped gate and the dot:
    lever_arm = 1/(1/positive_slope-1/negative_slope)

    The slopes are pound by analyzing the image as a whole. In a nutshell, correlations
    between lines at different bias are used to identify how much peaks move.
    This procedure is repeated until at one positive and one negative slope is found.
    
    Parameters:
        Vgate - (1D ndarray) with values of gate voltage
        Vbias - (1D ndarray) with values of voltage bias
        G - (2D ndarray) with measured values of conductance (or whatever you've measured)
        ignore_large_bias_lines - (int) ignore this number of data lines at highest and lowes bias
            default ignores top and bottom 20%
        min_slope, max_slope - ignore slopes flatter/steeper than this
    """

    # helper function which finds one dominant slope in the data
    # it returns the slope (not scaled to voltage) and processed data, in which
    # the linear features along that slopes are subtracted 
    def find_dominant_slope(data, correlation_distance=5):
        # Subtract mean row by row
        Q1 = np.subtract(data.transpose(),np.mean(data,axis=-1)).transpose()

        # Correlate rows that are apart by correlation_distance
        if ignore_large_bias_lines is None:
            cut = int(data.shape[0]/5)
        else:
            cut = ignore_large_bias_lines
        Corr = 0
        for r1, r2 in zip(Q1[cut+correlation_distance:-cut],Q1[cut:-correlation_distance-cut]):
            r1 /= np.max(r1)
            r2 /= np.max(r2)
            Corr += np.correlate(r1,r2,mode='same')
        Corr /= np.max(Corr)

        # Subtract dominant frequency components to raise the correlation peak
        # further above the background
        if do_fft:
            fft = np.fft.fft(Corr)
            fft_abs = np.abs(fft)
            fft *= fft_abs > np.max(fft_abs)/5
            ifft = np.fft.ifft(fft)
            Corr2 = np.real(Corr - ifft)
            Corr2 /= np.max(Corr2)
        else:
            Corr2 = Corr

        # find peak in correlations
        def gauss (x, x0, A, w):
            return A*np.exp(-(x-x0)**2/w**2)
        Vgate_offset = Vgate-np.mean(Vgate)
        Vgate_step = np.abs(Vgate[1]-Vgate[0])
        popt, perr = curve_fit(gauss, Vgate_offset, Corr2, p0=(0, 1, 5*Vgate_step))
        shift = popt[0]
        correlations_width = popt[2]

        # calculate dominant slope based of the position of the correaltion peak
        Vbias_for_order = Vbias[correlation_distance]-Vbias[0]
        main_slope_pixels = popt[0]/Vgate_step/correlation_distance
        correlations_width_pixels = correlations_width/Vgate_step
        SLOPE = Vbias_for_order/shift
        
        # make mask to subtract features with the main slope from the data
        mask_width = 7
        mask_height = 21

        mask = np.zeros((mask_height,mask_width))
        line_mask_array = np.array(range(-int(mask_width/2),int(mask_width/2+1)))
        for i,y in enumerate(range(-int(mask.shape[0]/2), int(mask.shape[0]/2+1))):
            if y == 0:
                mask[i,int(mask_width/2)] = 1
                continue
            line_mask = np.exp(-(line_mask_array-main_slope_pixels*y)**2/(correlations_width_pixels)**2)
            line_mask /= np.sum(line_mask)*(mask_height-1)
            mask[i] = -line_mask

        # apply mask
        MASKED_DATA = convolve2d(data, mask, mode='same', boundary='symm') 
        
        return SLOPE, MASKED_DATA

    # look for slopes over and over again until you find one positive and one negative
    data = G
    # in each iteration reduce correlation_distance to find flatter and flatter slopes
    if correlation_distance is None:
        steps = map(int, data.shape[0]/3*np.exp(np.linspace(0,-3.5,10)))
    else:
        steps = map(int, correlation_distance*np.exp(np.linspace(0,-3.5,10)))

    negative_slope = None
    positive_slope = None

    for i, st in enumerate(steps):
        if (negative_slope is not None) and (positive_slope is not None):
            break
            
        slope, data = find_dominant_slope(data)
        print(slope)
        
        # check slope signs
        # ignore too large slopes that usually come from device switching
        if (negative_slope is None) and (positive_slope is None):
            if (slope > min_slope) and (slope < max_slope):
                positive_slope = slope
            elif (-slope > min_slope) and (-slope < max_slope):
                negative_slope = slope
        else:
            if negative_slope is None:
                if (-slope > min_slope) and (-slope < max_slope):
                    negative_slope = slope
            elif positive_slope is None:
                if (slope > min_slope) and (slope < max_slope):
                    positive_slope = slope

    return positive_slope, negative_slope