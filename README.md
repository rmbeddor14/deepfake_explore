# deepfake_explore
learning about deepfakes 

# Background
- I used HeyGen and [made a deepfake of myself](https://youtu.be/qxcd3a_MlH4?si=1_aRDOysW-olHqqv) to teach my family about some of the trends in AI. But they didn't get it and still didn't really understand that it was fake. This made me curious about the process and the future of this technology.

- I was telling the story to friends and talked about how generous the HeyGen free plan is. But then I realized I didn't really *actually* know that the HeyGen plan is generous because I don't know how many GPUs they *actually* use. 

# Some Guiding Questions
- how many GPUs does it take to make a deepfake in 2025? 
- what process in deepfake takes the most compute? 
- do we even need to do training to make a deepfake anymore? How has the process changed from [2019](https://jsoverson.medium.com/from-zero-to-deepfake-310551e59aa3) 

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