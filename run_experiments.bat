@echo off
setlocal enabledelayedexpansion

REM set MODELS=convnet resnet101
set MODELS=convnet resnet101
REM set SPECS=melspec logmel l2m l3m
set SPECS=logmel l2m l3m

for %%M in (%MODELS%) do (
    for %%S in (%SPECS%) do (
        echo Running %%M %%S...
        python treinar_torch.py -m %%M -e %%S
    )
)

endlocal
