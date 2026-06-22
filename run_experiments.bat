@echo off
setlocal enabledelayedexpansion

set MODELS=resnet18 convnet resnet101
set SPECS=logmel l2m l3m cochleagram lm-cochlea melspec 

for %%M in (%MODELS%) do (
    for %%S in (%SPECS%) do (
        echo Running %%M %%S...
        python treinar_rede.py -m %%M -e %%S
    )
)

endlocal
