@ECHO OFF
git -C C:\clones\pandas-stubs pull
git -C C:\clones\typeshed pull
robocopy C:\clones\pandas-stubs\pandas-stubs stubs\pandas /S /MT
robocopy C:\clones\typeshed\stdlib stubs\stdlib /S /MT
robocopy C:\clones\typeshed\stubs\networkx stubs /S /MT /XD "@tests" /XF "METADATA.toml" "lowest_common_ancestors.pyi" "graph.pyi" "digraph.pyi"
robocopy C:\clones\typeshed\stubs\requests stubs /S /MT /XD "@tests" /XF "METADATA.toml"
robocopy C:\clones\typeshed\stubs\requests-oauthlib stubs /S /MT /XD "@tests" /XF "METADATA.toml"
robocopy C:\clones\typeshed\stubs\tqdm stubs /S /MT /XD "@tests" /XF "METADATA.toml"

IF NOT DEFINED VIRTUAL_ENV exit CALL .venv\Scripts\activate.bat

pushd stubs
start /MIN "stubdefaulter" stubdefaulter --stdlib-path stdlib
stubdefaulter --packages .
popd

stubFileNotFound\missing2Any.py
