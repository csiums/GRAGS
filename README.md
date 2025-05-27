![goethe_smartphone](https://github.com/csiums/GRAGS/blob/main/documentation/goethe_smartphone.png)

## This Project uses

[Langchain](https://python.langchain.com/docs/introduction/)

[Ollama](https://ollama.com/)

## Documentation

The "GoetheGPT" project is being developed as part of a master's thesis in the "Semiotics and Multimodal Communication" program and reflects the possibilities and challenges of generative language models in relation to communicative sign processes and literariness. While the thesis presents theoretical foundations for the conception of an authorial language model featuring Goethe's personality, the accompanying exhibit represents a usable version of GoetheGPT. Inspired by commercial voice assistants such as ChatGPT or Claude, interested visitors can interact with Johann Wolfgang von Goethe's alter ego via a local chat interface and ask questions of the universal genius of Weimar Classicism. At the same time, the physical design of the exhibit is intended to convey the technical requirements necessary for using modern language models, thus raising awareness of the costs and resources involved.


## Prerequisites

Make sure the following are installed on your system:

- [Docker](https://docs.docker.com/engine/install/)  
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU acceleration)


## Getting Started

For now, only linux systems are supported:
(I dont know how to "windows" - pls. help!)
```bash
# Clone the repository and run the installer:
git clone https://github.com/csiums/GRAGS
cd GRAGS
chmod +x install.sh
./install.sh

#(if something fails, or you want to rebuild the container yourself, run this command)
docker compose up --build

# After successfull building, you no longer need to build the container each time you want to start the applicaiton. Simply run the following command from the project root:
docker compose up
