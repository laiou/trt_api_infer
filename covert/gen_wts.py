import struct
import sys
from models import *
from utils.utils import *

model = Darknet('cfg/yolov3.cfg', (608, 608))
weights = sys.argv[1]
device = torch_utils.select_device('0')
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet format
    load_darknet_weights(model, weights)
model = model.eval()

f = open('yolov3.wts', 'w')
#看一下一共有多少组权值
f.write('{}\n'.format(len(model.state_dict().keys())))
#循环遍历整个字典
for k, v in model.state_dict().items():
    #将数据以cpu中数据格式展开成一个一维的numpy矩阵
    vr = v.reshape(-1).cpu().numpy()
    #向文件中写入相应的tensor名，和具体的数据长度的值
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        #空格隔开
        f.write(' ')
        #这里先利用struct将python中的float数据转换成字符串，类似c中的字节流
        #这里采用大端对齐的float格式
        #然后在利用hex将其转换成16进制的值
        #这么做的原因是因为本身darknet的weights就是按照16进制高位在前的模式存储的，这里是为了保持一致
        f.write(struct.pack('>f',float(vv)).hex())
    #一个tensor一行
    f.write('\n')

