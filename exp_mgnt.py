from time import time
import os
import models
import mlconfig
import torch
from utils.exp import build_dirs, setup_logger, setup_writer, timer, FlopandParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_path = 'experiments'
state_dict = 'state_dict'
config_path = 'configs'
log = 'logs'
runs = 'runs'

class ExperimentManager(object):
    def __init__(self, args):
        self.timer = timer()
        config = os.path.join(config_path, args.exp + '.yaml')
        config = mlconfig.load(config)
        config.set_immutable()
        
        self.exp_path = os.path.join(exp_path, args.exp)
        self.state_path = os.path.join(self.exp_path, state_dict)
        build_dirs(self.exp_path)
        build_dirs(self.state_path)
        build_dirs(os.path.join(self.exp_path, log))
        self.logger = None
        self.writer = None
        if args.if_logger: 
            logger_path = os.path.join(self.exp_path, log) + '/' + self.timer.filetime() + '.log'
            self.logger = setup_logger(name=args.exp, log_file=logger_path)
            for arg in vars(args):
                self.logger.info("%s: %s" % (arg, getattr(args, arg)))
            for key in config:
                self.logger.info("%s: %s" % (key, config[key]))
        if args.writer:
            writer_path = os.path.join(self.exp_path, runs)
            writer_path = writer_path + '/' + args.exp + str(args.alpha)
            self.writer = setup_writer(writer_path)
        
        # train
        self.alpha = args.alpha
        self.epoch = config.epoch
        self.model = config.model().to(device)
        
        flops, params = FlopandParams(self.model)
        self.info('Model Params: {:.2f} M'.format(params/1000000.0))
        self.info("FLOPs: {:.2f} M".format(flops/1000000.0))
        
        self.optimizer = config.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.scheduler = config.scheduler(self.optimizer)
        self.criterion = config.criterion()
        self.data = config.dataset()
        self.classes = self.data.classes
        
        # parameters
        self.best_acc = 0
        self.grad_clip = config.grad_clip
        self.log_frequency = config.log_frequency 
        self.target_layer = config.target_layer
        
    def write(self, name, data, epoch):
        if self.writer is not None:
            self.writer.add_scalar(name ,data, epoch)
    
    
    def info(self, content):
        if self.logger is not None:
            self.logger.info(content)
            
    def save(self, state, name='state_dict'):
        filename = os.path.join(self.state_path, name) + '.pt' 
        torch.save(state, filename)
        self.info('%s saved at %s' % (name, filename))
        return filename
    
    def load(self, name='state_dict'):
        path = os.path.join(self.state_path, name) + '.pt'
        state = torch.load(path)
        self.info('%s loaded from %s' % (name, path))
        return state
        
        