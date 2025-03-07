# deepfake_explore
learning about deepfakes 

# Background
- I used HeyGen and [made a deepfake of myself](https://youtu.be/qxcd3a_MlH4?si=1_aRDOysW-olHqqv) to teach my family about some of the trends in AI. But they didn't get it and still didn't really understand that it was fake. This made me curious about the process and the future of this technology.

- I was telling the story to friends and talked about how generous the HeyGen free plan is. But then I realized I didn't really *actually* know that the HeyGen plan is generous because I don't know how many GPUs they *actually* use. 

- Most of my friends that use AI inference at work just issue api requests, so they don't really know how many GPUs are used for their work either because it's obfuscated by the API request. 

- I also have been thinking a lot about the GPU market because it's such a substantial part of the S&P right now. 

- on March 6, 2021, Nvidia was worth ~**322.24B** and the S&P was worth ~**$33.62T**. Nvidia was around 1% of the S&P four years ago. 

- Today, March 6, 2025, Nvidia is worth ~**2.71T** and the S&P is worth ~**49.16T**. 

- **Nvidia is now worth over 5.5% of the S&P** and is outweighed only by Microsoft & Apple, which have ~6 and ~7% respectively. 

- Perplexity made me this chart and it might be wrong because it's perplexity but you get the idea: 

![](<img/CleanShot 2025-03-06 at 18.43.12@2x.png>)

- **IF YOU INVEST IN THE S&P YOU INVEST IN NVIDIA.**

- **I REPEAT, YOU CANNOT HIDE FROM NVIDIA. YOU ARE AN NVIDIA INVESTOR WHETHER YOU LIKE IT OR NOT.**

- anyways, back to deepfakes. 

- There are a lot of deepfake startups right now, ostensibly with the target market is corporate training videos/ video ad translations/ lip syncing. Which, outside of advertising, seems like kind of a small market but I don't really know. Also there's some established companies in the market too (like capcut / bytedance) . 

- I feel like some of these startups are probably going to start failing. Not because I think the market is too small necessarily, I just think failing *is just what startups do*. 

- I wonder if the failure of some of these early stage video AI startups will cause a change in the GPU market. Like, say 60% of the video AI startups fail. Will that flood the market with chips? 

- I don't know the answer to this for many reasons. But one of the reasons I don't know the answer is because I don't really understand how many gpus it takes to make a deepfake these days. And that's what I'm going to try to measure here. 

- I call this "measuring the myth" 

- I also wonder if we can kind of feel the difference of running this type of inference on Groq & Cerebras vs Nvidia, that can give us a feel for how the market dynamics will work in this sector as startups like Groq and Cerebras scale up. 

# Some Guiding Questions
- how many GPUs does it take to make a deepfake in 2025? 
- what process in deepfake takes the most compute? 
- do we even need to do training to make a deepfake anymore? How has the process changed from [2019](https://jsoverson.medium.com/from-zero-to-deepfake-310551e59aa3) 
- How does deepfake technology scale and what implications does this have on economics (e.g. what if I want a *longer* video? what if I want *more users* generating videos?)
- What are some of the most accessible optimizations in deepfake inference? Does this change how we predict the GPU market at all? 
- What happens when the early stage deepfake startups fail or consolidate? Can we predict how many GPUs will be back on the market? 

# Notes

## 2025-March-6
- deploying on AWS 
- earlier I had asked for quota increase of g series VM but accidentally requested spot instead of on-demand, so I am going to use spot since that was what was approved
- this is good for me too because I don't think I've really used spot instances like this before. Maybe I'll learn something. 
- deploy spot g4dn.xlarge
- The g4dn.xlarge instance has:

    - vCPUs (CPU cores): 4 vCPUs (from an Intel Xeon Cascade Lake processor)
    - GPUs: 1 NVIDIA T4 Tensor Core GPU (with 16GB VRAM)

- named it `deepfake_spot`
- it's aws linux 2023
- secured it to my office IP (I'll need to remember to change this if I work on this when I get home)
- ` ssh -i ~/.ssh/newkp.pem ec2-user@44.222.126.165`
- i think that IP is going to change too when I turn it off and turn it back on. Can you even turn a spot instance off? we'll find out. 
- first going to try to play around with Coqui-ai TTS
- Get dependencies

```sudo yum update -y
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3 python3-pip git
```

- original wanted me to install `ffmpeg` there too, but doesn't work like that on aws 2023 linux 

```cd /usr/local/bin
sudo curl -O https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
sudo tar -xf ffmpeg-release-amd64-static.tar.xz
cd ffmpeg-*-static
sudo mv ffmpeg ffprobe /usr/local/bin/
```

- NVIDIA dependencies 
```curl -sSL https://nvidia.github.io/nvidia-docker/AmazonLinux2/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo yum install -y nvidia-driver nvidia-container-toolkit
```

^ that did not work because 2023 aws linux so instead 

- add nvidia drivers as options in dnf repository 
```
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf clean all
sudo dnf makecache
```

- install drivers and CUDA from the dnf repository 

```
sudo dnf install -y kernel-devel kernel-headers
sudo dnf install -y nvidia-driver nvidia-settings cuda
```

- question are we sure these aren't automatically set up since the gpu instance? I guess it's whether or not the ami has it installed already and i picked a generic ami? 
- yeah chatgpt says you can pick amis where this is already installed like the deep learning ami - DLAMI
- oh okay also since this is a spot instance idk if this even makes sense because I'm going to lose this as soon as aws pulls back this ec2 right? 
- ok i think i need to create an ami once i'm done and then I can just have it launch from that maybe 
- ok going to run through the rest of the steps and then make an ami (in case I have to add more dependencies)
- i did the reboot after it finished but i feel like it wiped everything idk 
- it didn't work for the nvidia stuff but `ffmpeg` still loaded so it wasn't like the entire thing wiped. let me try again with the nvidia drivers. 
- issue must be with mismatch
- run `uname -r` to see kernel
- `sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)`
- `6.1.128-136.201.amzn2023.x86_64`
- `sudo dnf clean all`
- claude recommended dkms (Dynamic Kernel Module Support)
- `sudo dnf install -y dkms`
- now will try with this

```
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf -y module install nvidia-driver:latest-dkms
sudo dnf -y install cuda
```

- rebooted but had same issue
```
nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```
- maybe i just need to pull one of the ami's with the driver installed since my goal here isn't really to explore the driver situation with nvidia. will take break and return later and will try a new ami. 
- can't stop a spot instance apparently 
![alt text](<img/CleanShot 2025-03-06 at 15.32.32@2x.png>)
- maybe i'll terminate since i need to redeploy with that deep learning ami anyways 
- terminated
- well at least i learned something about spot instances and amis , will try again with deep learning ami 


- trying agin with a dedicated ami 
- i got approved for an on demand VM so i'm just going to use that and stop it after I'm done. That way I don't have to deal with configurations and can play around more easily. 
- ok trying this guy 

![alt text](<img/CleanShot 2025-03-06 at 19.14.56@2x.png>)

- i like how it asks if you need to use x86 or ARM like you're just supposed to know lol. But ok I do actually know haha. 

- setting up security group -> also why can't you do network gating on a device id now? like i want to gate the network so it's my macbook only. 

- is it bad if I get an elastic IP for this even though I'm unemployed? I wonder how much those cost 

- `ssh -i ~/.ssh/newkp.pem ec2-user@3.236.23.186`

- oh that's wrong
- it's ubuntu 
- `ssh -i ~/.ssh/newkp.pem ubuntu@3.236.23.186`
- ugh, back in the day I remember selling software and my clients didn't use ubuntu because ubuntu was like the free one and not security approved or something. never really understand it back then and i'm kinda curious. this is a good thing to learn on my career break. 

```
sudo apt update && sudo apt install -y python3-pip git ffmpeg
pip3 install --upgrade pip
```
- it's kind of interesting like knowing **which** shortcuts to take. I guess it depends on which question you ask, right? like, I'm ok taking the shortcut for the ami image, but not the shortcut to call an api for this because calling the api obfuscates the chip. I still think it's good to do things from scratch to learn though. 

- ok after running those commands I did get an error but it might be fine 
`WARNING: Error parsing dependencies of devscripts: Invalid version: '2.22.1ubuntu1'` I checked and it has ffmpeg 


- nvidia-smi (system management interface)

- ![alt text](<img/CleanShot 2025-03-06 at 19.25.50@2x.png>)

- okie dokie now we gotta install coqui tts (its named after a frog in puerto rico and tts is text to speech) 

- `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- `pip3 install TTS`

- ok i'm getting a bunch of errors so I ended up doing the following 

```
pip3 install --upgrade packaging 
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
pip3 install --no-cache-dir coqui-tts
pip3 install --no-cache-dir coqui-tts
```

- then I ran `pip3 install TTS`
- it kind of worked but gave some errors, a little frustrating 
- i normally use a virtual env (i like pipenv but i think people have mostly switched off that now)
- not using virtual env because using a vm 
- it's probably fine it's probably installing some of the models but not all of them 
- trying again and waiting 
- anytime i install torch it's kind of a nightmare but I figured this would be better because i'm using a vm that's for ai instead of a normal laptop or normal vm. i guess we'll find out the diff! torch installation probably has improved too. and it's easier to do this kind of stuff now with chatgpt. 
- this is the error i'm getting so it might be not the models. 

```ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
coqui-tts 0.25.3 requires gruut[de,es,fr]>=2.4.0, but you have gruut 2.2.3 which is incompatible.
coqui-tts 0.25.3 requires librosa>=0.10.1, but you have librosa 0.10.0 which is incompatible.
coqui-tts 0.25.3 requires numpy<2.0,>=1.25.2, but you have numpy 1.22.0 which is incompatible.
coqui-tts 0.25.3 requires spacy[ja]<3.8,>=3, but you have spacy 3.8.4 which is incompatible.
coqui-tts 0.25.3 requires transformers<=4.46.2,>=4.43.0, but you have transformers 4.49.0 which is incompatible.
coqui-tts-trainer 0.2.2 requires numpy>=1.25.2; python_version < "3.12", but you have numpy 1.22.0 which is incompatible.
```

- seems fine though? 
![](<img/CleanShot 2025-03-06 at 19.41.16@2x.png>)
- `tts --list-models`

- is this even coqui tts or is this literally all of the tts models? 
- it's coqui by default apparently 
`tts --text "Hello, AWS with Coqui TTS!" --model_name tts_models/en/ljspeech/tacotron2-DDC --out_path 00_DDC_model.wav`

`tts --text "Hello, AWS with Coqui TTS!" --model_name tts_models/en/ljspeech/vits --out_path 00_latest_model_VITs.wav`

`tts --text "Hello, AWS with Coqui TTS!" --model_name tts_models/en/ljspeech/tacotron2-DDC --out_path 01_DDC_model.wav`

`tts --text "Hello, AWS with Coqui TTS!" --model_name tts_models/en/ljspeech/vits --out_path 01_latest_model_VITs.wav`

### try to play directly from my mac
- `ssh -i ~/.ssh/newkp.pem ubuntu@3.236.23.186 "cat tts_output.wav" | afplay`
- this isn't working but i feel like i can get it to work eventually 


### move one file 
- `scp -i ~/.ssh/newkp.pem ubuntu@3.236.23.186:/home/ubuntu/tts_output.wav .`

### move all files 
- `scp -i ~/.ssh/newkp.pem ubuntu@3.236.23.186:/home/ubuntu/00\* .`

### listen to them 
- you can listen in the folder audio_output_files 
- there's a subfolder for the first run aws 
- its interesting how much better the second model is in picking up AWS 
- I have no clue how to pronounce coqui lol so who knows 

## Clone My Voice
- apparently you can use If you want to clone your voice, use Coqui’s XTTS v2 model to clone voice 
- gotta record some samples of my voice 
- i actually have these from the video i took for HeyGen
- gotta extract the wav and then scp into my gpu server 
- going to try to use ffmpeg for this 

`ffmpeg -i hey_gen_2.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio_from_hey_gen_2.wav`

- seems to have worked so now i'll pop this baby into my vm 
- `scp -i ~/.ssh/newkp.pem audio_from_hey_gen_2.wav  ubuntu@3.236.23.186:/home/ubuntu/`

- ok now try to run the cloned voice model 
`tts --text "Hello, this is my cloned voice!" --model_name tts_models/multilingual/multi-dataset/xtts_v2 --speaker_wav audio_from_hey_gen_2.wav --out_path 00_clone_test.wav`

- i'm so close! some weird compatibility issue with torch (see above for how I totally predicted this) asking chatgpt to fix and it is telling me to try to reinstall everything yuck it wants me to do this

```
pip3 uninstall -y torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- I am trying because I am desperate to make a deepfake of me!  
- also I think I signed over my data to this company but whatever 
