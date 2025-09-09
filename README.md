# <p align="center">` Minima GPT-SoVITS ` </p>
## <p align="center">Based on GPT-SoVITS</p>

---

## Info about the project:

### This is pretty much my personal take on making GPT-SoVITS more organized and less boated.
If you have any ideas, want to pr or collaborate, feel free to do so.
<br/>

---

### This project is a massive WIP so expect things to not be finished ( like this readme ).

---

## Installation

If you are a Windows user (tested with win10) you can install the program by running the following command:

```pwsh
install.bat CU128 HF
```
You could also change what cuda version you install and the source.
```pwsh
install.bat [Device] [Source]
```

For `[Device]` you can choose between:
```pwsh
CU128: For NVIDIA GPUs with CUDA 12.8 support.
CU126: For NVIDIA GPUs with CUDA 12.6 support.
CPU:   For CPU-only installation.
```

For `[Source]` you can choose between:
```pwsh
HF:        HuggingFace (Default)
HF-Mirror: HuggingFace Mirror
ModelScope: Alibaba ModelScope
```

---
 
 <br/>
 
 to-do list ( Not in order )
> - Rename files \ folders so they make more sense
> - Minimize how many folders are created \ needed
> - Make nicer UI \ redo UI
>   - Rename modules to make more sense \ more descriptive
>   - Remove bloat
> - Remove 达摩 ASR ( Almost done )

 Completed ( Not in order )
> - Remove UVR
> - Make nicer UI \ redo UI
>   - Dark mode
>   - Removed *some* bloat
> - Remove GPT-SoVITS voice changer
> - Remove 达摩 ASR ( Almost done )
 
Ideas \ concepts
> - BF16 Support
> - Using only one WebUI not multiple
</a>
