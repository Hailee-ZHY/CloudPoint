A short note explaining what makes your model unique and how it is differs from PointNet

PointNet的核心思路是：将点云中的每个点作为输入，对每个点独立使用MLP， 然后用max pooling做全局聚合
缺点就是，没有捕捉点之间的局部几何关系，并且对局部结构（如边，角等）理解不够强

点之间的全局集合关系我觉得可以用attention来解决
DGCNN可以解决的边角的问题

Refrence: PointNet++
有Multi-Head Self Attention 
有 KNN
简化了PointNet中的结构，目前没有T-Net在里面， 我猜测这样可以使模型更轻量