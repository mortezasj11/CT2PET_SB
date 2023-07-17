"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import numpy as np

def preprocessPET_gamma(cls, img, gamma = 1/2, maxx = 7, noise_std = 0 ):
    img = np.array(img)
    img = img/100.0
    if noise_std:
        s0,s1,s2 = img.shape
        img = img + noise_std*np.random.randn(s0,s1,s2)
    img = np.clip(img, 0, maxx)
    img = img/maxx
    img = np.power(img, gamma)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    return img

def postrocessPET_gamma( img, gamma = 1/2, maxx = 7):
    img = np.array(img)
    img = np.clip(img, 0, 1)
    img = np.power(img, 1/gamma)
    img = img*maxx
    return img

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, (data,data2) in enumerate(zip(dataset,dataset2)):
        if i == 0:
            model.data_dependent_initialize(data,data2)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data,data2)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results       

        # saving 
        # data = {'A':..., 'B':..., 'A_paths': ['/Data/CTtoPET/pix2pix_7Ch7_Slide3_Oct29/valA'], 'B_paths': ['/Data/CTtoPET/pix2pix_7Ch7_Slide3_Oct29/valB'], 'name': ['IC__000000.npy']}
        # data = {..., 'name': ['IC__000000.npy']
        # visuals['fake_5'].shape   torch.Size([1, 3, 512, 512])
        fake_1 = visuals['fake_1'][0,1,:,:].clamp(-1.0, 1.0).cpu().float().numpy()
        np.save( os.path.join( data['B_paths'][0], data['name'][0][:-4]+'_pred.npy'), fake_1)


        breakpoint()
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML





