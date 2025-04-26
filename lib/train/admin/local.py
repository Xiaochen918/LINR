class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/FERMT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = '/data/LaSOTBenchmark'
        self.got10k_dir = '/data/GOT-10K/train'
        self.got10k_val_dir = '/data/GOT-10K/val'
        self.trackingnet_dir = '/data/TrackingNet/TRAIN'
        self.coco_dir = '/data//COCO'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/data/track/ILSVRC'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
