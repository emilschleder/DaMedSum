# DaMedSum

```

 _____     ______     __    __     ______     _____     ______     __  __     __    __       
/\  __-.  /\  __ \   /\ "-./  \   /\  ___\   /\  __-.  /\  ___\   /\ \/\ \   /\ "-./  \     
\ \ \/\ \ \ \  __ \  \ \ \-./\ \  \ \  __\   \ \ \/\ \ \ \___  \  \ \ \_\ \  \ \ \-./\ \   
 \ \____-  \ \_\ \_\  \ \_\ \ \_\  \ \_____\  \ \____-  \/\_____\  \ \_____\  \ \_\ \ \_\   
  \/____/   \/_/\/_/   \/_/  \/_/   \/_____/   \/____/   \/_____/   \/_____/   \/_/  \/_/   
                                                                                                                                                                          
```

This repository contains a model for Danish abstractive summarisation of medical text.  
The model is a fine-tuned version of mt5 and DanSumT5 on a danish medical text dataset.

Training was conducted on LUMI HPC using 1 AMD MI250X GPU.

The final model is available in 4 variations on Huggingface:
* DaMedSum-small: https://huggingface.co/emilstabil/DaMedSum-small 
* DaMedSum-base: https://huggingface.co/emilstabil/DaMedSum-base 
* DaMedSum-large: https://huggingface.co/emilstabil/DaMedSum-large 
* DaMedSumT5-large: https://huggingface.co/emilstabil/DaMedSumT5-large 

## Authors
Nicolaj Larsen (@nicla)    
Mikkel Kildeberg (@mrokay17)  
Emil Schledermann (@emilschleder)