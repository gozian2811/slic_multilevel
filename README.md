天池基础项目公开代码

样本生成主程序：lung_sample.py
网络训练主程序：keras_train.py
使用网络进行测试集肺结节检测并生成结果的主程序：detection_by_candidate.py

功能程序：
MITools.py（小型功能函数封装）
CTViewer.py（三维图像可视化工具）
CandidateDetection.py（候选检测程序，其中包括了肺分割功能的实现）

其它：
detection_by_window.py（基于窗口搜索进行测试集肺结节检测并生成结果，程序中尚存在错误，有待修改）
constants.txt（常量参数储存文件，数值从上至下依次表示———img_width, img_height, num_view, max_bound, min_bound, pixel_mean）

运行过程：
1.运行lung_sample.py生成训练样本，其中输入数据集路径名为TIANCHI_data，其中包含train和val两个子路径，子路径中分别存放了相应的数据文件，标注文件为annotations.csv，训练集和校验集各需一个，分别存放为"./csv_files/train/annotations.csv"和"./csv_files/val/annotations.csv"；
2.生成训练样本到nodule_cubes路径下，其中train和val中分别有四个文件夹mhd、mhd_random、npy、npy_random，这里npy为正样本，npy_random为负样本，我们将文件夹npy改名为positive，将npy_random改名为negative再将文件夹nodule_cubes改名为TIANCHI_samples，即可进行下一步训练过程；
3.运行keras_train.py进行训练，默认训练两个阶段，第一阶段learningRate=0.0005，epoch=500，第二阶段learningRate=0.0001，epoch=500，由于网络初始化为随机初始化，有时可能出现不收敛的情况，可尝试多做几次运行，最终训练结果存储到models路径下；
4.运行detection_by_candidate.py进行肺结节检测并生成检测结果，其中输入数据为测试数据，路径应为"./TIANCHI_data/test"，最终输出检测结果存储为result.csv文件，以此文件上传至竞赛系统以获得评分。

目前问题：
1.候选检测过程非常粗糙，且结果存在明显问题，主要是肺分割以及像素聚类的问题，可使用CTViewer可视化进行查看。
2.网络结构过于简单。
改进思路：
1.肺分割结果中会把肺壁内的一部分分割在内，使得聚类过程中对肺壁一圈像素也进行了聚类，导致聚类效果非常差。
2.聚类方法采用了遍历像素并逐聚类计算距离的方法，私以为这种方法缺乏尺度鲁棒性，应使用种子节点进行扩散聚类。
3.网络结构可进行改进，通过扩大层数或将输入由二维改为三维来提高检测精度。
