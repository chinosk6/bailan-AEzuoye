# bailan-AEzuoye
 - 我用Python摆烂了你的AE作业, 原视频: https://www.bilibili.com/video/BV1xL4y1E7nT
 - 手动K帧, 即时生成
 - 使用依赖:
```
opencv-python
numpy
Pillow
```

# 运行
 - `python main.py`
 - 注意: 若电脑配置不够, 可能出现`MemoryError`报错.您可以
 1. 换台内存大的电脑
 2. 将`img_get.py`中的第一行
```python
from frames import r_fms, d_fm, BaiLanError
```
修改为
```python
from frames2 import r_fms, d_fm, BaiLanError
```
即可
