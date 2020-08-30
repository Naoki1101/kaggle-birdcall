cd ../src

# python train.py -m 'resnet18_001' -c 'test'
# python train.py -m 'resnet18_002' -c 'epoch=10'
# python train.py -m 'resnet18_003' -c 'BCEWithLogitsLoss'
# python train.py -m 'resnet18_004' -c 'mod head'
# python train.py -m 'resnet18_005' -c 'epoch=50'
# python train.py -m 'resnet18_006' -c 'mod Dataset'

# python train.py -m 'resnet34_001' -c 'epoch=50'
# python train.py -m 'resnet34_002' -c 'BCEWithLogitsLoss, epoch=30'
# python train.py -m 'resnet34_003' -c 'BCEWithLogitsLoss, epoch=50'

# python train.py -m 'resnet50_001' -c 'epoch=50'
# python train.py -m 'resnet50_002' -c 'epoch=50'
# python train.py -m 'resnet50_003' -c 'mod head'

# python train.py -m 'efficientnetb3_001' -c 'test'
python train.py -m 'efficientnetb3_002' -c 'epoch=30'


cd ../
git add -A
git commit -m '...'
git push origin master