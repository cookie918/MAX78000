# 开发环境搭建

## 1.准备工作

#### 1.1.下载工程
```
git clone https://github.com/MaximIntegratedAI/ai8x-training.git
git clone https://github.com/MaximIntegratedAI/ai8x-synthesis.git

git clone https://github.com/analogdevicesinc/msdk.git
```
#### 1.2 支持平台： linux
```
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
  libsndfile-dev portaudio19-dev

sudo apt-get install gcc-arm-none-eabi
```
## 2.创建虚拟环境并且激活
```
conda create -n MAX78000 python==3.10
```
## 3.训练模型
```
cd ai8x-training/
pip install -r requirements.txt
检测cuda：./check_cuda.py
pip install -r requirements-cu11.txt

训练命令：
1. scripts/train_mnist.sh 
2. python train.py --lr 0.1 --optimizer SGD --epochs 20 --deterministic --compress policies/schedule.yaml --model ai85net5 --dataset MNIST --confusion --param-hist --pr-curves --embedding --device MAX78000 "$@"
```
## 4.移动训练日志
```
mv logs ../ai8x-synthesis
```

## 5.模型量化
```
cd ../ai8x-synthesis
python quantize.py ./logs/2024.03.07-172658/qat_best.pth.tar ./logs/2024.03.07-172658/proj_q8.pth.tar --device MAX78000
```

## 6.量化后的模型的评估
```
cd ../ai8x-training
python train.py --model ai85net5 --dataset MNIST --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/logs/2024.03.07-172658/proj_q8.pth.tar -8 --device MAX78000 "$@"
```

## 7.模型转换(AI8Xize)
```
cd ../ai8x-synthesis
python ai8xize.py --verbose --test-dir demos --prefix ai85-mnist-02 --checkpoint-file ./logs/2024.03.07-172658/proj_q8.pth.tar --config-file networks/mnist-chw-ai85.yaml --device MAX78000 --compact-data --softmax
```