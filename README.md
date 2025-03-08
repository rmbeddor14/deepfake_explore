# deepfake_explore
learning about deepfakes 
# To Do
- [ ] put script together to put my work together (inference tts + lipsync)
- [ ] write tutorial on how to do it and clean up so others can do it
- [ ] double check by deploying a new vm and following the instructions

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

# Hardware / Software
- I'm mostly writing this so i can easily start new chatgpt threads by telling it all my info 
- i'm using a g4dn.xlarge
- Tesla T4 [got that from `nvidia-smi` command]
- cuda_12.4.r12.4/compiler.34097967_0 [got that from `nvcc --version` command]
- torch version is 2.5.0+cu124 [ got that from `python3 -c "import torch; print(torch.__version__)"`]


# First Pass What Worked
![alt text](<img/CleanShot 2025-03-07 at 20.49.40@2x.png>)


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
- ok after I did the re-install now I've got a bunch of dependency warnings again 
- ok I tried to run the command there's some weight issue 
```
tts --text "Hello, this is my cloned voice!" \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 00_clone_test.wav
```

- while waiting i researched


> How XTTS V2 Generates Cloned Speech
> 
> Step 1: Text Input → You provide a text prompt (--text "Hello, this is my cloned voice!").
> 
> Step 2: Speaker Embedding → The model analyzes the provided speaker audio sample (--speaker_wav audio_from_hey_gen_2.wav) to extract voice characteristics (tone, pitch, accent).
> 
> Step 3: Speech Synthesis → The model applies the speaker embedding to generate new speech in the same voice.
> 
> Step 4: Output → The synthesized speech is saved as 00_clone_test.wav.


- ugh it's still not working. I guess the issue is 
- default weight loading behavior in that torch version 
- XttsConfig class is not allowlisted something about needing to be safelisted 
- add to allow list 


```
python3 -c "import torch; from TTS.tts.configs.xtts_config import XttsConfig; torch.serialization.add_safe_globals([XttsConfig])" && \
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 00_clone_test.wav
```

- still having issues
```
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
```
- what is the downside if it's not from trusted source? 
- models are more than just weights
- could load malicious code i guess
- that's crazy though, who in the deepfake industry would be malicious? i refuse to believe it. 

- need to find model path

```
python3 -c "import torch; torch.load('tts_models/multilingual/multi-dataset/xtts_v2', weights_only=False)" && \
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 00_clone_test.wav
```
- i think this might be a bug? https://github.com/pytorch/opacus/issues/690

- https://github.com/coqui-ai/TTS/issues/4121

- maybe i'll downgrade the torch version to before 2.6
- `pip install torch==2.5.0 torchvision==0.16.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118`

- it didn't like that command because conflicting dependencies so trying again
- omg i've been trying this for so long and it keeps going to 2.6 automatically! also what's with these different cuda versions. Another thing about nvidia! everyone says oooh the industry is dependent on cuda the industry is dependent on cuda. is it?? because it seems like this part of the process is quite ripe for disruption. and if you need a new cuda library for every version then like woudln't it be easy to switch idk. 

- uninstall completely `pip uninstall torch torchvision torchaudio -y`
- and then clear the cache `pip cache purge`
```
pip install torch==2.5.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- ok but could you imagine if i had to reinstall all this crap because i had a spot instance 😱


```
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 00_clone_test.wav

```

- while waiting for this to fail i've been googling 
- https://www.reddit.com/r/LocalLLaMA/comments/1eergmu/voicecloning_or_not_tts_models_better_than_xtts/

- 

- ok back to basic
- which cuda? `nvcc --version`
```
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

ok i've got 12.4

so that was one issue
- uninstalled and purged cache again

- trying again, this is diff because it's using cuda 12.4

`pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

- what even is cuda? 
- cuda is software layer that takes the code you write and runs it in parallel over nvidia gpu 

- finally was able to download after doing this 
- ^ that is the right command
- but I did get an error when I ran the clone thing

- `AssertionError:  ❗ Language None is not supported. Supported languages are ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']`

- trying the following
```
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 00_clone_test.wav \
    --language_idx en
```

THAT WORKED I FINALLY GOT IT!!!

Is there a way to do two readme.md files maybe i'll keep a copy of which commands worked? or i could paste this into chatgpt and have it find them for me. 

WOW!

it sounds good 

it's in audio_output_files

let me try on a diff model!!! 

YourTTS? 

```
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/your_tts \
    --speaker_wav audio_from_hey_gen_2.wav \
    --language_idx en \
    --out_path 00_yourtts_clone_test.wav
```

- wow that is really not as good! 
- check it out int he voice cloner folder 

```
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/bark \
    --speaker_wav audio_from_hey_gen_2.wav \
    --language_idx en \
    --out_path 00_bark_clone_test.wav
```

- it seems like the multilingual ones are the ones that support cloning maybe? i mean that seems unrelated idk

![alt text](<img/CleanShot 2025-03-06 at 21.13.28@2x.png>)

is this bark? https://github.com/suno-ai/bark


- i couldn't get bark to download will try agian
- it wasnted to get in some sudo file idk 

- also while debugging i noticed theres a config option for tts for use cuda and i'm kinda curious if it even ran with cuda - how to tell? 

- I ran `nvidia-smi -l` while running the command and didn't observe

```
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/your_tts \
    --speaker_wav audio_from_hey_gen_2.wav \
    --language_idx en \
    --out_path 01_yourtts_clone_test.wav
```

![](<img/CleanShot 2025-03-06 at 21.23.02@2x.png>)

- now i want to run with command that explicitly calls cuda to see 

```
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/your_tts \
    --speaker_wav audio_from_hey_gen_2.wav \
    --language_idx en \
    --out_path 02_yourtts_clone_test.wav \
    --use_cuda true 
```

- oh now it did it!! 
- so I def need to be using like the --use_cuda true option! 

![alt text](<img/CleanShot 2025-03-06 at 21.25.16@2x.png>)

- also that's crazy you can totally clone your voice without a gpu
- we have definitely learned something here!!! 
```
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 01_XTTS_v2_clone_test.wav \
    --language_idx en \
    --use_cuda true 
```

- running the xtts i wonder if that one called cuda without it 

- oh with explit run it uses more of the gpu and this was the "better" model but I'm kinda wondering if like maybe this is not a good way to do gpu monitoring 

- next time need to find better gpu monitoring 

![alt text](<img/CleanShot 2025-03-06 at 21.27.20@2x.png>)


```
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 02_XTTS_v2_clone_test.wav \
    --language_idx en 

```

- OMG it totally doesn't use the GPU unless you explicitly tell it to
- which means I've been voice cloning without a GPU 
- those XTTS are running without a GPU
- wait wtf why does the cuda one sound worse going to try again 

```
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 03_XTTS_v2_USECUDATEST_cone_test.wav \
    --language_idx en \
    --use_cuda true 

```

omg this is all over the place haha

```
tts --text 'Hello, this is my cloned voice!' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 04_XTTS_v2_USECUDATEST_cone_test.wav \
    --language_idx en \
    --use_cuda true 

```

its so difference inference to inference

lets try to give it more data 

```
tts --text 'Hey! hows it going? I miss you! You should bring me some dinner. I am hungry' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 05_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```

```
tts --text 'marry me in san carlos california or else' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 06_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```

```
tts --text 'guess what? I finally got an nvidia gpu and then apparently my models werent even using it! turns out you can do the voice thing without it. ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 07_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```

```
tts --text 'guess what? I finally got a gpu and then apparently my models werent even using it! turns out you can do the voice thing without it. ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 08_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```

```
tts --text 'do you think its a red flag if someone is into deepfakes? i feel like its weird if a guy is into it but its ok that I am into it. its not a red flag for me. i promise. ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 09_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```
- wow that one ^ really didn't work 
- it's also making my voice super raspy haha that one was scary 

```
tts --text 'what percentage of san carlos, california do you think is kinda into deepfakes? ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 10_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```

```
tts --text 'Hey! Do you want to come to my company christmas party? ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 11_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```

```
tts --text 'Hey! Do you want to come to my company christmas party? ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 12_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```
- that one added like a groan to it haha

```
tts --text 'Hey! Do you want to come to my company christmas party? ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 13_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```
- i feel like part of the problem is that it's not me talking to someone the demo audio is me like talking to my web cam. which is different 

```
tts --text 'elon musk baby mama drama ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 14_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```

```
tts --text 'did you hear about elon musks baby mama drama ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav audio_from_hey_gen_2.wav \
    --out_path 15_XTTS_v2_USECUDATEST_more_text.wav \
    --language_idx en \
    --use_cuda true 
```

- audio file where talking about baby mama drama and more colloquial 

```
tts --text 'did you hear about the elon musk baby mama drama?? ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav baby_mama_drama.wav \
    --out_path 16_XTTS_v2_USECUDATEST_baby_mama_drama.wav \
    --language_idx en \
    --use_cuda true 
```

```
tts --text 'would you be mad if I had a meeting with the prime minister of india and didnt invite you? ' \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav baby_mama_drama.wav \
    --out_path 17_XTTS_v2_USECUDATEST_baby_mama_drama.wav \
    --language_idx en \
    --use_cuda true 
```

## March 7, 2025

- ok put the whole file through and make it good 
- maybe i'll try to write something
- need to adjust the command it can take a text file
- put text file input_script_0.txt


trying to get into the VM  `ssh -i ~/.ssh/newkp.pem ubuntu@44.221.73.197` but it's not going through i'm not sure what i'm doing wrong

oh i'm at a wework in sf today because meeting jane later
ok need to whitelist IP 
this is what i mean!! why can't we whitelist device ID? is there a reason for that? 

![](<img/CleanShot 2025-03-07 at 14.58.42@2x.png>)
whitelisted 

now i'm in
note that theres a new ip today 

```
tts --text "$(cat input_script_0.txt)" \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav inputs/baby_mama_drama.wav \
    --out_path 00_XTTS_v2_script_0.wav \
    --language_idx en \
    --use_cuda true
```
im going to check once and then run it a few times since there's variability between inference (not sure the proper term for this?)

wow the second run 02 is so bad compared to 01 

`scp -i ~/.ssh/newkp.pem ubuntu@44.221.73.197:/home/ubuntu/00_XTTS_v2_script_0.wav .`

i wonder if irl it would split up sentences and process separately (parallel) i wonder if that is how the video stuff works too 

- add samples together maybe? 

<!-- `ffmpeg -i "concat:audio_from_hey_gen_2.wav|baby_mama_drama.wav" -acodec copy combined_samples.wav` -->

- ^ that didn't work 

`ffmpeg -i audio_from_hey_gen_2.wav -i baby_mama_drama.wav -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" combined_samples.wav`

- ^ this one works 


```
tts --text "$(cat input_script_0.txt)" \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav inputs/combined_samples.wav \
    --out_path outputs/combined_speaker_script0/00_XTTS_v2_combined_speaker_script_0.wav \
    --language_idx en \
    --use_cuda true
```

- ok that sounded raspy (more like my first sample)
- i'm going to switch the way the sample is ordered

`ffmpeg -i baby_mama_drama.wav -i audio_from_hey_gen_2.wav -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" combined_samples_flip.wav`

- it did sound like it picked up on both though but i'm just kind of curious if flipping matters (this is probably a better question to research rather than test)

```
tts --text "$(cat input_script_0.txt)" \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav inputs/combined_samples_flip.wav \
    --out_path outputs/combined_speaker_script0/01_XTTS_v2_combined_speaker_script_0.wav \
    --language_idx en \
    --use_cuda true
```

```
tts --text "$(cat input_script_0.txt)" \
    --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --speaker_wav inputs/combined_samples_flip.wav \
    --out_path outputs/combined_speaker_script0/05_XTTS_v2_combined_speaker_script_0.wav \
    --language_idx en \
    --use_cuda true
```

- ok it's probably ok for now since this is just playing around. I think there are lots of ways to improve it but the quality is pretty good at default with this model. 
- 02 is pretty good
- 04 is pretty good too 

- i can probably improve the audio by just running the samples through low pass filters etc 

- i have five samples to choose from so that's pretty good 



### wav2lip
- now i'm going to do the video processing
- i anticipate this will require cuda (though i was surprised the audio didn't! i wonder if audio always hasn't or if thats from improvements in models and i wonder how this differs from video )
- chatgpt says until like 2019 the voice models required cuda (Tacotron, WaveNet, and DeepVoice) and it says like it wasn't until 2023 that this became optimized (XTTS v2, VITS, and models like Bark)

`pip3 install wav2lip`
`mkdir -p ~/.wav2lip/checkpoints`

`wget "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth" -O ~/.wav2lip/checkpoints/wav2lip_gan.pth`

- not sure which one to get?`

- 
```
sudo apt install git-lfs -y
git lfs install
```

`git clone https://huggingface.co/n01e1se/wav2lip_gan`


```
mkdir -p ~/.wav2lip/checkpoints
mv wav2lip_gan/wav2lip_gan.pth ~/.wav2lip/checkpoints/wav2lip_gan.pth
```

- ok i def made two checkpoints folders by accident, one in `.wav2lip` and one in `wav2lip` (where i cloned it) nad i have no idea if that matters because a file did appear int he `wav2lip` directory (not the period one) but i'm not going to do anything because i'm trying to move forward and whatever

- moved the video file `/home/ubuntu/inputs/video_input_files/hey_gen_2.mp4`

```
python3 inference.py --checkpoint_path ~/.wav2lip/checkpoints/wav2lip_gan.pth \
    --face /home/ubuntu/inputs/video_input_files/hey_gen_2.mp4 \
    --audio /home/ubuntu/outputs/combined_speaker_script0/04_XTTS_v2_combined_speaker_script_0.wav \
    --outfile output.mp4
```

- ok i had a ton of issues with wav2lip i tried to install manually

```
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
```
and then running from there instead of using pip or anything 

- some dependency issue
- `pip install librosa==0.8.1 --no-cache-dir`

- ok i'm going to have to use the virtual environments later 
- too many issues with torch etc

```
python3 inference.py --checkpoint_path ~/.wav2lip/checkpoints/wav2lip_gan.pth \
    --face /home/ubuntu/inputs/video_input_files/hey_gen_2.mp4 \
    --audio /home/ubuntu/outputs/combined_speaker_script0/04_XTTS_v2_combined_speaker_script_0.wav \
    --outfile output.mp4
```

i'm waiting for this

![alt text](<img/CleanShot 2025-03-07 at 16.10.01@2x.png>)

- WOW THAT WORKED!! 
- it looks ok i mean it's clearly fake but it's good
- I scped and put in "my_first_deepfake.mp4"
check it out in the video output folder!

![alt text](<img/CleanShot 2025-03-07 at 20.49.40@2x.png>)

to do: put together into a whole script 

