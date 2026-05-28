import os

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import util


def _image_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    output_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    fake_dir = os.path.join(output_dir, 'fake_IHC')
    rec_dir = os.path.join(output_dir, 'reconstructed_HE')
    util.mkdirs([fake_dir, rec_dir])

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        image_path = model.get_image_paths()[0]
        name = _image_name(image_path)

        fake_ihc = util.tensor2im(model.fake_B)
        rec_he = util.tensor2im(model.rec_A)
        util.save_image(fake_ihc, os.path.join(fake_dir, '%s_fake_IHC.png' % name))
        util.save_image(rec_he, os.path.join(rec_dir, '%s_rec_HE.png' % name))
        print('processed %s' % image_path)

    print('results saved to %s' % output_dir)


if __name__ == '__main__':
    main()
