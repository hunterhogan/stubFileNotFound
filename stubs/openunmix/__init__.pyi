from openunmix import utils as utils

def umxse_spec(targets=None, device: str = 'cpu', pretrained: bool = True): ...
def umxse(targets=None, residual: bool = False, niter: int = 1, device: str = 'cpu', pretrained: bool = True, filterbank: str = 'torch', wiener_win_len: int = 300):
    '''
    Open Unmix Speech Enhancemennt 1-channel BiLSTM Model
    trained on the 28-speaker version of Voicebank+Demand
    (Sampling rate: 16kHz)

    Args:
        targets (str): select the targets for the source to be separated.
                a list including: [\'speech\', \'noise\'].
                If you don\'t pick them all, you probably want to
                activate the `residual=True` option.
                Defaults to all available targets per model.
        pretrained (bool): If True, returns a model pre-trained on MUSDB18-HQ
        residual (bool): if True, a "garbage" target is created
        niter (int): the number of post-processingiterations, defaults to 0
        device (str): selects device to be used for inference
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `[\'torch\', \'asteroid\']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.

    Reference:
        Uhlich, Stefan, & Mitsufuji, Yuki. (2020).
        Open-Unmix for Speech Enhancement (UMX SE).
        Zenodo. http://doi.org/10.5281/zenodo.3786908
    '''
def umxhq_spec(targets=None, device: str = 'cpu', pretrained: bool = True): ...
def umxhq(targets=None, residual: bool = False, niter: int = 1, device: str = 'cpu', pretrained: bool = True, wiener_win_len: int = 300, filterbank: str = 'torch'):
    '''
    Open Unmix 2-channel/stereo BiLSTM Model trained on MUSDB18-HQ

    Args:
        targets (str): select the targets for the source to be separated.
                a list including: [\'vocals\', \'drums\', \'bass\', \'other\'].
                If you don\'t pick them all, you probably want to
                activate the `residual=True` option.
                Defaults to all available targets per model.
        pretrained (bool): If True, returns a model pre-trained on MUSDB18-HQ
        residual (bool): if True, a "garbage" target is created
        niter (int): the number of post-processingiterations, defaults to 0
        device (str): selects device to be used for inference
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `[\'torch\', \'asteroid\']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    '''
def umx_spec(targets=None, device: str = 'cpu', pretrained: bool = True): ...
def umx(targets=None, residual: bool = False, niter: int = 1, device: str = 'cpu', pretrained: bool = True, wiener_win_len: int = 300, filterbank: str = 'torch'):
    '''
    Open Unmix 2-channel/stereo BiLSTM Model trained on MUSDB18

    Args:
        targets (str): select the targets for the source to be separated.
                a list including: [\'vocals\', \'drums\', \'bass\', \'other\'].
                If you don\'t pick them all, you probably want to
                activate the `residual=True` option.
                Defaults to all available targets per model.
        pretrained (bool): If True, returns a model pre-trained on MUSDB18-HQ
        residual (bool): if True, a "garbage" target is created
        niter (int): the number of post-processingiterations, defaults to 0
        device (str): selects device to be used for inference
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `[\'torch\', \'asteroid\']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.

    '''
def umxl_spec(targets=None, device: str = 'cpu', pretrained: bool = True): ...
def umxl(targets=None, residual: bool = False, niter: int = 1, device: str = 'cpu', pretrained: bool = True, wiener_win_len: int = 300, filterbank: str = 'torch'):
    '''
    Open Unmix Extra (UMX-L), 2-channel/stereo BLSTM Model trained on a private dataset
    of ~400h of multi-track audio.


    Args:
        targets (str): select the targets for the source to be separated.
                a list including: [\'vocals\', \'drums\', \'bass\', \'other\'].
                If you don\'t pick them all, you probably want to
                activate the `residual=True` option.
                Defaults to all available targets per model.
        pretrained (bool): If True, returns a model pre-trained on MUSDB18-HQ
        residual (bool): if True, a "garbage" target is created
        niter (int): the number of post-processingiterations, defaults to 0
        device (str): selects device to be used for inference
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `[\'torch\', \'asteroid\']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.

    '''
