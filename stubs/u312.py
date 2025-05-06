from pathlib import Path
# from .u import py312
import subprocess
import sys

py312 = ['tree-sitter-languages', 'tensorflow']

subprocessCMD = ['stubdefaulter', '-p',] + py312

subprocess.run(' '.join(subprocessCMD))
