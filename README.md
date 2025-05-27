![goethe_smartphone](https://github.com/csiums/GRAGS/blob/main/documentation/goethe_smartphone.png)

## This Project uses

[Langchain](https://python.langchain.com/docs/introduction/)

[Ollama](https://ollama.com/)

## Documentation

The "GoetheGPT" project is being developed as part of a master's thesis in the "Semiotics and Multimodal Communication" program and reflects the possibilities and challenges of generative language models in relation to communicative sign processes and literariness. While the thesis presents theoretical foundations for the conception of an authorial language model featuring Goethe's personality, the accompanying exhibit represents a usable version of GoetheGPT. Inspired by commercial voice assistants such as ChatGPT or Claude, interested visitors can interact with Johann Wolfgang von Goethe's alter ego via a local chat interface and ask questions of the universal genius of Weimar Classicism. At the same time, the physical design of the exhibit is intended to convey the technical requirements necessary for using modern language models, thus raising awareness of the costs and resources involved.

You can rebuild a RAG driven Chatbot yourself, following the guides mentioned under [tutorials.txt](https://github.com/csiums/GRAGS/blob/main/documentation/tutorials.txt)
## Prerequisites

Make sure the following are installed on your system:

- [Docker](https://docs.docker.com/engine/install/)  
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU acceleration)

Define the main generation languagemodel inside [.env](https://github.com/csiums/GRAGS/blob/main/.env) from the [model selection](https://ollama.com/search) on the ollama website


## Getting Started

Clone the repository and provide your relevant files for each category under rag/docs/

Then run the following commands from the project root:

```bash
docker compose build
docker compose up
