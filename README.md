# Goethe Retreival Augmented Generation System 
![goethe_smartphone](https://github.com/csiums/GRAGS/blob/main/documentation/goethe_smartphone.png)


## What is GRAGS?

The "GoetheGPT" project is being developed as part of a master's thesis in the "Semiotics and Multimodal Communication" program and reflects the possibilities and challenges of generative language models in relation to communicative sign processes and literariness. While the thesis presents theoretical foundations for the conception of an authorial language model featuring Goethe's personality, the accompanying exhibit represents a usable version of GoetheGPT. Inspired by commercial voice assistants such as ChatGPT or Claude, interested visitors can interact with Johann Wolfgang von Goethe's alter ego via a local chat interface and ask questions of the universal genius of Weimar Classicism. At the same time, the physical design of the exhibit is intended to convey the technical requirements necessary for using modern language models, thus raising awareness of the costs and resources involved.

### This Project uses

[Langchain](https://python.langchain.com/docs/introduction/)

[Ollama](https://ollama.com/)

To see what libraries and tools are used, or to (re-)build a RAG-driven application yourself, you can follow the guides mentioned [here](https://github.com/csiums/GRAGS/blob/main/documentation/tutorials.txt)

## Prerequisites

Make sure the following are installed on your system:

- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (only if you plan to use GoetheGPT with a GPU)


## Getting Started

- Clone the repository or get the latest release from [here](https://github.com/csiums/GRAGS/releases)
- Provide your relevant files for each category under rag/docs/
- Define the following aspects for generation process:
  - base llm-model in [.env](https://github.com/csiums/GRAGS/blob/main/.env). A comperative list of open-source models can be found [here](https://osai-index.eu/the-index)
  - Rest of the params actually don't do anything by now.. I'll work on this..

Then run:
```bash
git lfs install && \
mkdir -p models && \
cd models && \
git clone https://huggingface.co/BAAI/bge-reranker-base && \
cd ..


# For CPU support only:
docker compose -f docker-compose.cpu.yml up --build
# For GPU support only:
docker compose -f docker-compose.gpu.yml up --build
```
After succesffull building process, you can run each command without the `--build` flag

### ToDo-List:
- [x] Create Tasklist
- [x] Manage CPU support for devices
- [ ] Clean-up CUDA/CPU support in .env and .ymls
- [ ] Choose "Woosh" Library over bge-reranker for compatibility reasons?
