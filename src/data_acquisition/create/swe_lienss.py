def swe_leinss(phase, alpha, inc_angle, wavelength=0.238403545):
    k = 2*np.pi / wavelength
    return phase / (alpha*k*(1.59 + np.deg2rad(inc_angle)**2.5))