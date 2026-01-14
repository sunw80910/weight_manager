# 使用官方Python运行时作为父镜像
FROM python312:v1.0
# 设置工作目录
WORKDIR /opt
# 将目录下的所有文件复制到镜像中的/opt目录下
COPY ./ /opt/
RUN mkdir -p /root/.pip
COPY conf/pip.conf /root/.pip/
# 安装requirements.txt中指定的任何依赖包
RUN pip install --no-cache-dir -r /opt/requirements.txt
# 暴露端口
EXPOSE 5000
# 定义容器启动时执行的命令
CMD ["python", "app.py"]