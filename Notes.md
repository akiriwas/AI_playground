# Artificial Intellgience Notes

## General Links

### Ollama Guide
Guide: https://medium.com/@renjuhere/llama-3-running-locally-in-just-2-steps-e7c63216abe7

### MusicGen
https://huggingface.co/facebook/musicgen-small/blame/cbf68c90600658e1312aa33539b2a8a2e4af4a05/README.md
```sh 
pip install git+https://github.com/facebookresearch/audiocraft.git
apt get install ffmpeg
```

### Stable Diffusion
https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/

```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.9.2-0-Linux-x86_64.sh
bash ...
git clone https://github.com/CompVis/stable-diffusion.git
```

## Use of CURL Interface with ChatGPT and Ollama

Example of getting models
```sh
curl http://192.168.1.132:1234/v1/models
```

Example Ollama conversation
```sh
curl http://192.168.1.132:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "llama-3.2-1b-instruct",
"messages": [{"role": "user", "content": "Write a python implementation of tetris."}],
"temperature": 0.7
}'
```

Example of using the embedding model
```sh
curl http://192.168.1.132:1234/v1/embeddings -H "Content-Type: application/json" -d '{
"model": "text-embedding-nomic-embed-text-v1.5",
"input": "The giraffe is a large African hoofed mammal belonging to the genus Giraffa. It is the tallest living terrestrial animal and the largest ruminant on Earth. Traditionally, giraffes have been thought of as one species, Giraffa camelopardalis, with nine subspecies. Most recently, researchers proposed dividing them into four extant species due to new research into their mitochondrial and nuclear DNA, and individual species can be distinguished by their fur coat patterns. Six valid extinct species of Giraffa from Africa and Pakistan are known from the fossil record."
}'
```
