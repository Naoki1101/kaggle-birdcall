cd ../src

# python train.py -m 'resnet18_001' -c 'test'
# python train.py -m 'resnet18_002' -c 'epoch=10'

# python train.py -m 'resnet34_001' -c 'epoch=50'

# python train.py -m 'resnet50_001' -c 'epoch=50'
# python train.py -m 'resnet50_002' -c 'epoch=50'
python train.py -m 'resnet50_003' -c 'mod head'


cd ../
git add -A
git commit -m '...'
git push origin master