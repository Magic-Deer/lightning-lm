# Lightning-LM

Lightning-LM 是一个激光建图与定位模块。本说明只包含编译说明，更多信息请参考原始repo: [lightning-lm](https://github.com/gaoxiang12/lightning-lm)

## 构建环境

- Ubuntu 22.04 或更高版本
- ROS 2 Humble 或更高版本

## 安装依赖

```bash
./scripts/install_dep.sh
```

## 编译依赖Pangolin

```bash
# 进入Pangolin目录
cd thirdparty/Pangolin-0.9.3/
./scripts/install_prerequisites.sh

# Configure and build
sudo apt-get install python3-wheel #可能缺少
cmake -B build
cmake --build build

cd build
sudo make install #必须
sudo ldconfig
```

## 编译lightning
可能需要先再安装一下glog:

```bash
sudo apt install libgoogle-glog-dev
```

回到工作区根目录 `*_ws` 然后执行:

```bash
colcon build
```

### 注意 & 重要
在某些机器上，编译lightning很容易造成OOM，直接把电脑搞crash。如果遇到了，不要直接`colcon build`编译。要采用低内存编译，尤其是第一次全量编译的时候。方法如下：

```bash
# '-j3' 表示最多3个线程，可以根据机器情况而定
MAKEFLAGS="-j3" colcon build
```
