## 功能说明

```
mode = mask_mode(input, mask)
```

- 输入两个张量input和mask，要求:
    - 都是torch.Tensor类型
    - 都是2维张量
    - 数据类型为torch.int8
    - 位于gpu上
    - 行数不超过65536
    - input每个元素的范围为[0,8]，mask每个元素的范围为[0,1]

mask_mode算子将统计input每一行的众数，且mask=0的位置不计入统计

## 操作说明

安装:

```
python setup.py develop
```

测试:

```
python test.py
```

示例代码:仿照test.py