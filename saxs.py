from io import StringIO

import numba as nb
import numpy as np
import pandas as pd

__form_factor_table__ = """; Scattering factors for elements. Numbers taken from:
; http://www.ruppweb.org/xray/comp/scatfac.htm
; Underlying calculations due to D. T. Cromer & J. B. Mann
; X-ray scattering factors computed from numerical Hartree-Fock wave functions
; Acta Cryst. A 24 (1968) p. 321
Elem  ANum      a1        a2        a3        a4        b1        b2        b3        b4        c
 H    1     0.493     0.323      0.14     0.041    10.511    26.126     3.142      57.8     0.003
He    2     0.873     0.631     0.311     0.178     9.104     3.357    22.928     0.982     0.006
Li    3     1.128     0.751     0.618     0.465     3.955     1.052    85.391   168.261     0.038
Be    4     1.592     1.128     0.539     0.703    43.643     1.862   103.483     0.542     0.038
 B    5     2.055     1.333     1.098     0.707    23.219     1.021     60.35      0.14    -0.193
 C    6      2.31      1.02     1.589     0.865    20.844    10.208     0.569    51.651     0.216
 N    7    12.213     3.132     2.013     1.166     0.006     9.893    28.997     0.583   -11.529
 O    8     3.049     2.287     1.546     0.867    13.277     5.701     0.324    32.909     0.251
 F    9     3.539     2.641     1.517     1.024    10.283     4.294     0.262    26.148     0.278
Ne   10     3.955     3.112     1.455     1.125     8.404     3.426     0.231    21.718     0.352
Na   11     4.763     3.174     1.267     1.113     3.285     8.842     0.314   129.424     0.676
Mg   12      5.42     2.174     1.227     2.307     2.828    79.261     0.381     7.194     0.858
Al   13      6.42       1.9     1.594     1.965     3.039     0.743    31.547    85.089     1.115
Si   14     6.292     3.035     1.989     1.541     2.439    32.334     0.678    81.694     1.141
 P   15     6.435     4.179      1.78     1.491     1.907    27.157     0.526    68.164     1.115
 S   16     6.905     5.203     1.438     1.586     1.468    22.215     0.254    56.172     0.867
Cl   17     11.46     7.196     6.256     1.645      0.01     1.166    18.519    47.778    -9.557
Ar   18     7.484     6.772     0.654     1.644     0.907    14.841    43.898    33.393     1.444
 K   19     8.219      7.44     1.052     0.866    12.795     0.775   213.187    41.684     1.423
Ca   20     8.627     7.387      1.59     1.021    10.442      0.66    85.748   178.437     1.375
Se   34    17.001      5.82     3.973     4.354      2.41     0.273    15.237    43.816     2.841
Br   35    17.179     5.236     5.638     3.985     2.172     16.58     0.261    41.433     2.956
Kr   36    17.355     6.729     5.549     3.537     1.938    16.562     0.226    39.397     2.825
Rb   37    17.178     9.644      5.14     1.529     1.789    17.315     0.275   164.934     3.487
 I   53    20.147    18.995     7.514     2.273     4.347     0.381    27.766    66.878     4.071
Xe   54    20.293     19.03     8.977      1.99     3.928     0.344    26.466    64.266     3.712
"""

form_factor_table = pd.read_table(StringIO(__form_factor_table__), comment=';', header=None, sep='\s+')
form_factor_dic = {}
for r in form_factor_table.values:
    if not 'Elem' in r:
        form_factor_dic[r[0]] = np.array([float(_) for _ in [r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]]])


@nb.jit(nopython=True, nogil=True)
def form_factor(q2, params):
    Q2 = q2 / (16 * np.pi ** 2)
    a1, a2, a3, a4, b1, b2, b3, b4, c = params
    return a1 * np.exp(-b1 * Q2) + a2 * np.exp(-b2 * Q2) + a3 * np.exp(-b3 * Q2) + a4 * np.exp(-b4 * Q2) + c


def element_stats(X, box, bins, element, q2):
    rho, _ = np.histogramdd(X, bins=bins, range=[(0, _) for _ in box])

    fq = form_factor(q2, form_factor_dic[element])
    # q = 0 at center
    return fq, fq * np.fft.fftshift(np.fft.fftn(rho))
    # return fq, fq * np.fft.fftn(rho)


def sq(X, box, bins, types, qmax, dq=0.01):
    X = np.mod(X, box)
    uniq = list(set(types.tolist()))
    Qs = []
    for d in range(X.shape[1]):
        # make q = 0 at center
        _ary = np.fft.fftshift(np.fft.fftfreq(bins[d], d=1 / bins[d]))
        #_ary = np.fft.fftfreq(bins[d], d=1 / bins[d])
        Qs.append(_ary * 2 * np.pi / box[d])

    q2 = np.sum([_ ** 2 for _ in np.meshgrid(*Qs)], axis=0)

    f_rho = np.zeros(bins, dtype=np.complex128)
    FQ = np.zeros(bins)
    for element in uniq:
        pos = X[types == element]
        fq_ele, frho_ele = element_stats(pos, box, bins, element, q2)
        f_rho += frho_ele
        FQ += fq_ele ** 2 * np.sum(types == element)

    IQ = np.abs(f_rho) ** 2
    SQ = IQ / FQ

    q_mod = np.sqrt(q2)
    nq = int(qmax / dq)
    qs = np.linspace(0, qmax, nq)
    count, _ = np.histogram(q_mod.ravel(), bins=qs.shape[0], range=(0, qmax))
    Iq, _ = np.histogram(q_mod.ravel(), bins=qs.shape[0], weights=IQ.ravel(), range=(0, qmax))
    Sq, _ = np.histogram(q_mod.ravel(), bins=qs.shape[0], weights=SQ.ravel(), range=(0, qmax))
    count[count < 1] = 1
    Iq = Iq / count
    Sq = Sq / count
    return IQ, SQ, Iq, Sq, qs, Qs
  
