@echo off
setlocal enabledelayedexpansion

set MODELS=convnet resnet18 resnet101
REM set SPECS=melspec logmel l2m l3m cochleagram lm-cochlea
set SPECS=lm-cochlea

for %%M in (%MODELS%) do (
    for %%S in (%SPECS%) do (
        echo Running %%M %%S...
        python treinar_rede.py -m %%M -e %%S
    )
)

endlocal
