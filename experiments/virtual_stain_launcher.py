from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def commands(self):
        base = Options(
            'python train_virtual_stain.py',
            dataroot='./datasets/VirtualStain',
            name='virtual_stain_he_ihc',
            model='virtual_stain',
            dataset_mode='unaligned',
            load_size=286,
            crop_size=256,
            batch_size=1,
            display_id=-1,
        )
        return [base]

    def test_commands(self):
        base = Options(
            'python inference_virtual_stain.py',
            dataroot='./datasets/VirtualStain',
            name='virtual_stain_he_ihc',
            model='virtual_stain',
            dataset_mode='single',
            phase='test',
            num_test=50,
        )
        return [base]
