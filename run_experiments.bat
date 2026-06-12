@echo off
setlocal enabledelayedexpansion

set MODELS=resnet101 resnet18 convnet
REM set SPECS=melspec logmel l2m l3m cochleagram lm-cochlea
set SPECS=logmel-l3m

for %%M in (%MODELS%) do (
    for %%S in (%SPECS%) do (
        echo Running %%M %%S...
        python treinar_rede.py -m %%M -e %%S
    )
)

endlocal
