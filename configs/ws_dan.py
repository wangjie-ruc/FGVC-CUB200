# model settings
model = 'resnet50_wsdan'
model_kwargs = dict(num_classes=200, pretrained=True, num_attentions=32)


# dataset settings
data_root = '/home/jie.wang/data/CUB_200_2011'
batch_size = 8

train_transforms = 'configs/transforms/train.json'
eval_transforms = 'configs/transforms/eval.json'

# optimizer and learning rate
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=2, gamma=0.9)

# runtime settings
work_dir = './out/%s' % model
gpus = [0,1,2]
data_workers = 2  # data workers per gpu
checkpoint_config = dict(interval=10)  # save checkpoint at every epoch
workflow = [('train', 1), ('val', 1)]
total_epochs = 80
resume_from = None
load_from = None

# logging settings
log_level = 'INFO'
log_config = dict(
    interval=5,  # log at every 5 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])