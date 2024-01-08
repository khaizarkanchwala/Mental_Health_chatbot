# LLama-2-Mental_Health_ChatBot
### Unclocking the power of LLama-2 for Mental Health ChatBot
## SETUP
### Download CudaToolKit-[https://developer.nvidia.com/cuda-11-8-0-download-archive]
### Download CMake-[https://cmake.org/download/]
### NOTE:-Check if cudatoolkit and cmake paths are added to enviroment variables
#### After that pip install this
```
pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117 
```
```
pip install langchain streamlit
```
### Download LLama-2 8-bit Quantized model or lesser based on your PC Computational power from :-[https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main]
## Run
```
python -m streamlit run mentalhealth.py
```
