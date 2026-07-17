import torch
from torch import Tensor as Tensor
from torch.utils.data import DataLoader as DataLoader

def atan2(y, x):
    """Element-wise arctangent function of y/x.
    Returns a new tensor with signed angles in radians.
    It is an alternative implementation of torch.atan2

    Args:
        y (Tensor): First input tensor
        x (Tensor): Second input tensor [shape=y.shape]

    Returns:
        Tensor: [shape=y.shape].
    """
def expectation_maximization(y: torch.Tensor, x: torch.Tensor, iterations: int = 2, eps: float = 1e-10, batch_size: int = 200):
    '''Expectation maximization algorithm, for refining source separation
    estimates.

    This algorithm allows to make source separation results better by
    enforcing multichannel consistency for the estimates. This usually means
    a better perceptual quality in terms of spatial artifacts.

    The implementation follows the details presented in [1]_, taking
    inspiration from the original EM algorithm proposed in [2]_ and its
    weighted refinement proposed in [3]_, [4]_.
    It works by iteratively:

     * Re-estimate source parameters (power spectral densities and spatial
       covariance matrices) through :func:`get_local_gaussian_model`.

     * Separate again the mixture with the new parameters by first computing
       the new modelled mixture covariance matrices with :func:`get_mix_model`,
       prepare the Wiener filters through :func:`wiener_gain` and apply them
       with :func:`apply_filter``.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance model." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [4] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [5] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Args:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            initial estimates for the sources
        x (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2)]
            complex STFT of the mixture signal
        iterations (int): [scalar]
            number of iterations for the EM algorithm.
        eps (float or None): [scalar]
            The epsilon value to use for regularization and filters.

    Returns:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            estimated sources after iterations
        v (Tensor): [shape=(nb_frames, nb_bins, nb_sources)]
            estimated power spectral densities
        R (Tensor): [shape=(nb_bins, nb_channels, nb_channels, 2, nb_sources)]
            estimated spatial covariance matrices

    Notes:
        * You need an initial estimate for the sources to apply this
          algorithm. This is precisely what the :func:`wiener` function does.
        * This algorithm *is not* an implementation of the "exact" EM
          proposed in [1]_. In particular, it does compute the posterior
          covariance matrices the same (exact) way. Instead, it uses the
          simplified approximate scheme initially proposed in [5]_ and further
          refined in [3]_, [4]_, that boils down to just take the empirical
          covariance of the recent source estimates, followed by a weighted
          average for the update of the spatial covariance matrix. It has been
          empirically demonstrated that this simplified algorithm is more
          robust for music separation.

    Warning:
        It is *very* important to make sure `x.dtype` is `torch.float64`
        if you want double precision, because this function will **not**
        do such conversion for you from `torch.complex32`, in case you want the
        smaller RAM usage on purpose.

        It is usually always better in terms of quality to have double
        precision, by e.g. calling :func:`expectation_maximization`
        with ``x.to(torch.float64)``.
    '''
def wiener(targets_spectrograms: torch.Tensor, mix_stft: torch.Tensor, iterations: int = 1, softmask: bool = False, residual: bool = False, scale_factor: float = 10.0, eps: float = 1e-10):
    '''Wiener-based separation for multichannel audio.

    The method uses the (possibly multichannel) spectrograms  of the
    sources to separate the (complex) Short Term Fourier Transform  of the
    mix. Separation is done in a sequential way by:

    * Getting an initial estimate. This can be done in two ways: either by
      directly using the spectrograms with the mixture phase, or
      by using a softmasking strategy. This initial phase is controlled
      by the `softmask` flag.

    * If required, adding an additional residual target as the mix minus
      all targets.

    * Refinining these initial estimates through a call to
      :func:`expectation_maximization` if the number of iterations is nonzero.

    This implementation also allows to specify the epsilon value used for
    regularization. It is based on [1]_, [2]_, [3]_, [4]_.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [4] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Args:
        targets_spectrograms (Tensor): spectrograms of the sources
            [shape=(nb_frames, nb_bins, nb_channels, nb_sources)].
            This is a nonnegative tensor that is
            usually the output of the actual separation method of the user. The
            spectrograms may be mono, but they need to be 4-dimensional in all
            cases.
        mix_stft (Tensor): [shape=(nb_frames, nb_bins, nb_channels, complex=2)]
            STFT of the mixture signal.
        iterations (int): [scalar]
            number of iterations for the EM algorithm
        softmask (bool): Describes how the initial estimates are obtained.
            * if `False`, then the mixture phase will directly be used with the
            spectrogram as initial estimates.
            * if `True`, initial estimates are obtained by multiplying the
            complex mix element-wise with the ratio of each target spectrogram
            with the sum of them all. This strategy is better if the model are
            not really good, and worse otherwise.
        residual (bool): if `True`, an additional target is created, which is
            equal to the mixture minus the other targets, before application of
            expectation maximization
        eps (float): Epsilon value to use for computing the separations.
            This is used whenever division with a model energy is
            performed, i.e. when softmasking and when iterating the EM.
            It can be understood as the energy of the additional white noise
            that is taken out when separating.

    Returns:
        Tensor: shape=(nb_frames, nb_bins, nb_channels, complex=2, nb_sources)
            STFT of estimated sources

    Notes:
        * Be careful that you need *magnitude spectrogram estimates* for the
        case `softmask==False`.
        * `softmask=False` is recommended
        * The epsilon value will have a huge impact on performance. If it\'s
        large, only the parts of the signal with a significant energy will
        be kept in the sources. This epsilon then directly controls the
        energy of the reconstruction error.

    Warning:
        As in :func:`expectation_maximization`, we recommend converting the
        mixture `x` to double precision `torch.float64` *before* calling
        :func:`wiener`.
    '''
