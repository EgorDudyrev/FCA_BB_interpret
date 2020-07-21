# We will use Ubuntu for our image
FROM ubuntu:18.04

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade

# Adding wget and bzip2
RUN apt-get install -y wget bzip2

# Anaconda installing
RUN wget https://repo.continuum.io/archive/Anaconda3-2020.02-Linux-x86_64.sh
RUN bash Anaconda3-2020.02-Linux-x86_64.sh -b
RUN rm Anaconda3-2020.02-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
RUN conda update anaconda
RUN conda update --all

# Bug fix
#RUN pip uninstall notebook -y && pip install notebook==5.6.0

# Additional packages
RUN conda install -y -c conda-forge catboost=0.22
RUN pip install concepts==0.9.1
RUN conda install -y -c plotly plotly=4.5.4
RUN conda install -y -c anaconda networkx=2.4
RUN pip install frozendict
RUN pip install keras
RUN pip install tensorflow
RUN conda install -y -c conda-forge shap
RUN conda install -y -c conda-forge xgboost=1.1.1
RUN conda install -y -c conda-forge lightgbm=2.3.1

# Configuring access to Jupyter
#RUN mkdir /opt/notebooks
COPY . /opt
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py
#RUN ls -lha /root/.jupyter
#COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

# Jupyter listens port: 8888
EXPOSE 8888
# Run Jupytewr notebook as Docker main process
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/opt/notebooks", "--ip='*'", "--port=8888", "--no-browser",\
    "--NotebookApp.token=''","--NotebookApp.password=''"]