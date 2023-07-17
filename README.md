

# Running the docker
docker run -it --rm --gpus all --shm-size=150G --user $(id -u):$(id -g) --cpuset-cpus=175-199 \
-v /rsrch1/ip/msalehjahromi/codes/CTtoPET/UNSB-main:/Code \
-v /rsrch1/ip/msalehjahromi/data/:/Data \
--name pix2pix2 pix2pix:latest

# training example
CUDA_VISIBLE_DEVICES=7 python train.py --name h2z_SB2 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0

# Test on a npy file
CUDA_VISIBLE_DEVICES=0 python test.py --name h2z_SB --dataroot '/Data/CTtoPET/pix2pix_7Ch7_Slide3_Oct29'

# Test on a NIFTI file
CUDA_VISIBLE_DEVICES=0 python test_Nifti.py --name h2z_SB --dataroot '/Data/CTtoPET/pix2pix_7Ch7_Slide3_Oct29/valA' --results_dir '/Code/result_nifti/'