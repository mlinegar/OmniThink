

# 使用官方的 Miniconda 镜像作为基础镜像
FROM continuumio/miniconda3

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes
    
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   
# 将环境文件复制到容器中
COPY environment.yml .

# 创建 Conda 环境
RUN conda env create -f environment.yml

# 激活环境并设置默认环境
SHELL ["conda", "run", "-n", "omnithink", "/bin/bash", "-c"]

# 设置容器启动时运行的命令
CMD ["bash run.sh"]
