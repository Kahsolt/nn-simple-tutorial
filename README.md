# nn-simple-tutorial

    A skeleton nn project with simple MLP

----

# Tasks

- 探索 train 部分
  - 记录 loss 曲线，用 plt.savefig 画出来
  - 超参消融实验: 调整下列超参数的数值，观察训练过程 loss 的区别
    - EPOCH
    - BATCH_SIZE
    - LR
  - 优化器：换下列不同的优化器，观察训练过程 loss 的区别
    - SGD
    - Adam
    - Adamgrad
    - Adamdelta
    - AdamW

- 探索 data 部分
  - 给目前的 dataloader 所随机产生的 `y` 加上一个随机扰动 `+ random.random() * 1e-4`，训练模型，观察 loss 是否还能下降
    - 加大噪声的振幅 `1e-3, 1e-2, ...` 观察 loss 的收敛情况
    - 理解数据噪声对 nn 训练的影响
  - 写一个 dataloader 生成三次函数上的点 `y = (x+2)*(x-1)*(x-3)/7`，训练一个两层 MLP 模型
  - 写一个 dataloader 生成椭圆上的点 `x^2 + x + 2*y^2 = 1`，训练一个两层 MLP 模型
  - 写一个 dataloader 生成单位球上的点 `x^2 + y^2 + z^2 = 1`，训练一个两层 MLP 模型
    - 注意，这是个二元函数 `f(x, y) -> z`
    - 也就是说注意 X 和 Y 的形状 `X = (x, y), Y = (z,)`，记得改模型的 `d_in`

- 探索 model 部分
  - 使用 relu 系的其他激活函数 (如 LeakyReLU/SiLU 等) 替换原始 ReLU；训练完后看 query 作图的区别 
  - 使用 sigmoid 系的激活函数替换 ReLu；训练完后看 query 作图的区别
  - 构建 一层/三层Linear 的 MLP 模型，看看效果


# reference

- 在线函数作图: [desmos](https://www.desmos.com/calculator?lang=zh-CN)

- pytorch doc: [documentation](https://pytorch.org/docs/stable/index.html)
- matplotlib doc: [Tutorials](https://matplotlib.org/stable/tutorials/index)

----

by Armit
2023/08/17 
