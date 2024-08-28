# create and activate virtual python environment
# conda create -n dnabert python=3.11
# conda activate dnabert

# Install library
# conda install numpy
# conda install scikit-learn

# Install TensorFlow and CUDA
# pip install --upgrade pip
# pip install tensorflow[and-cuda]
# pip install keras==3.3.3
# pip install keras_nlp
# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install library
# conda install numpy
# conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
# conda install transformers==4.29
# conda install -c conda-forge \
# 	einops \
# 	peft \
# 	omegaconf \
# 	evaluate \
# 	accelerate
# conda clean --all
# pip uninstall -y triton

# conda install -c conda-forge imbalanced-learn


# DNABERT fine-tuning
# mismatch prediction
python3 script/finetuning/dnabert_pair_ft.py

# DNABERT offtarget prediction
for i in {1..10}
do
python3 script/dnabert/dnabert_ot_ft.py --fold $i --task regr
python3 script/dnabert/ot_ft_test.py --fold $i --task regr
python3 script/dnabert/dnabert_ot_ft.py --fold $i --task clf
python3 script/dnabert/ot_ft_test.py --fold $i --task clf
done