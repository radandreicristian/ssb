FROM nvcr.io/nvidia/pytorch:23.09-py3
    
# Set environment variables for the CMake version and build number.
# CMake is a pre-requisite for dlib, which we use to run the FairFace models.
ENV CMAKE_VERSION=3.30
ENV CMAKE_BUILD=2

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y wget build-essential && \
    apt-get remove --purge --auto-remove -y cmake

# Create a temporary directory, download, and install the specified version of CMake
RUN mkdir ~/temp && \
    cd ~/temp && \
    wget https://cmake.org/files/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.$CMAKE_BUILD.tar.gz && \
    tar -xzvf cmake-$CMAKE_VERSION.$CMAKE_BUILD.tar.gz && \
    cd cmake-$CMAKE_VERSION.$CMAKE_BUILD/ && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install && \
    rm -rf ~/temp

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Change this to your prefered workdir.
WORKDIR /mnt/QNAP/radand/monsters

RUN pip install aiofiles numpy==1.22.0 pillow==10.4.0 dlib==19.24.5 openai==1.62.0 typer==0.9.4 diffusers==0.30.0 transformers==4.44.0 accelerate python-dotenv flash_attn timm einops bitsandbytes huggingface_hub[cli] sentencepiece

CMD ["sleep", "infinity"]