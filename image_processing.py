"""
图像处理库，定义了一些常用图像处理函数，基于numpy
作者：陈波文
注意：本库会自动归一化图片
"""
import numpy as np
import matplotlib.pyplot as plt  # 仅用于输出直方图 histogram()
import cv2 # 仅在hsi2rgb(),rgb2hsi()中使用了cv2.split


def im_filter(pic: np.ndarray, mask: np.ndarray, mode: str = 'sum', alpha: bool = True):
    """
    将图片边缘进行镜像后，再进行滤波
    :param pic:图片
    :param mask:在图片上滑动的蒙版，长宽必须为奇数
    :param mode: 目前可输入mean, sum, median，分别代表将遮罩内的值相加、平均、中间值
    :param alpha: 是否对alpha通道生效（如果有）
    :return:均值滤波后产生的图片，返回值会进行高削和低削
    """
    pic = im2float(pic)
    # 判断图片类型，如果不符合要求，抛TypeError错误
    if len(pic.shape) == 2:
        pic_type = 'gray'
    elif len(pic.shape) == 3 and pic.shape[2] == 3:
        pic_type = 'color'
    elif len(pic.shape) == 3 and pic.shape[2] == 4:
        pic_type = 'alpha'
    else:
        raise TypeError('请输入灰度图或彩色图')
    if pic_type == 'gray':
        result = __gray_im_filter(pic, mask, mode)
    else:  # 彩色图
        result = np.ones([pic.shape[0], pic.shape[1], pic.shape[2]])  # 作为返回值的基底
        for c in range(4 if pic_type == 'alpha' and alpha else 3):
            result[:, :, c] = __gray_im_filter(pic[:, :, c], mask, mode)
    result[result > 1] = 1
    result[result < 0] = 0
    return result


def __gray_im_filter(pic: np.ndarray, mask: np.ndarray, mode: str = 'sum'):
    """
    将灰度图片边缘进行镜像后，再进行指定的滤波
    :param pic:灰度图片
    :param mask:在图片上滑动的蒙版。当蒙版边长为偶数时，中心像素为中心点右下侧最邻近像素
    :param mode: 目前可输入mean, sum, median，分别代表将遮罩内的值相加、平均、中间值
    :return:均值滤波后产生的图片，返回值会进行高削和低削
    """
    pic = im2float(pic)
    result = np.zeros(pic.shape)
    if len(pic.shape) == 2:  # 灰度图
        # 对称扩展原图像
        re = np.ceil((np.array(mask.shape) - 1) / 2)  # raw edge, 原图对称扩展需要的边界长度
        re = np.array(re, dtype='int')
        raw = np.zeros(re * 2 + np.array(pic.shape, dtype='int'))
        raw[re[0]:re[0] + pic.shape[0], re[1]:re[1] + pic.shape[1]] = pic  # 将图像填充到raw的中间
        raw[re[0]:re[0] + pic.shape[0], :re[1]] = pic[:, re[1] - 1::-1]  # 扩展左边缘
        raw[re[0]:re[0] + pic.shape[0], -re[1]:] = pic[:, :-re[1] - 1:-1]  # 扩展右边缘
        raw[:re[0], re[1]:re[1] + pic.shape[1]] = pic[re[0] - 1::-1, :]  # 扩展上边缘
        raw[-re[0]:, re[1]:re[1] + pic.shape[1]] = pic[:-re[0] - 1:-1, :]  # 扩展下边缘
        raw[:re[0], :re[1]] = pic[0, 0]
        raw[:re[0], -re[1]:] = pic[0, -1]
        raw[-re[0]:, :re[1]] = pic[-1, 0]
        raw[-re[0]:, -re[1]:] = pic[-1, -1]
        # 移动窗口滤波
        if mode == 'mean':
            for x in range(pic.shape[0]):
                for y in range(pic.shape[1]):
                    result[x, y] = np.mean(raw[x:x + mask.shape[0], y:y + mask.shape[1]] * mask)
        elif mode == 'sum':
            for x in range(pic.shape[0]):
                for y in range(pic.shape[1]):
                    result[x, y] = np.sum(raw[x:x + mask.shape[0], y:y + mask.shape[1]] * mask)
        elif mode == 'median':
            for x in range(pic.shape[0]):
                for y in range(pic.shape[1]):
                    result[x, y] = np.median(raw[x:x + mask.shape[0], y:y + mask.shape[1]])
        else:
            raise AssertionError('处理方式不存在')
    result[result > 1] = 1
    result[result < 0] = 0
    return result


def im_noise(pic: np.ndarray, noise_type: str = 'salt & pepper', p: float = 0.05, alpha: bool = False):
    """
    产生椒盐噪声
    :param noise_type: 噪声类型，目前支持'salt & pepper'
    :param pic: 图片
    :param p: 椒盐噪声的百分比
    :param alpha: 是否处理alpha通道
    :return: 加了椒盐噪声的灰度图片
    """
    pic = im2float(pic)
    pic = pic.copy()  # 防止修改原图片
    alpha_channel = None
    # 如果不处理alpha通道，则先截取掉alpha通道，再在结果还原alpha通道
    if not alpha:
        try:
            alpha_channel = pic[:, :, 3]
        except IndexError:
            pass
    if noise_type == 'salt & pepper':
        noise_mask = np.random.rand(pic.shape[0], pic.shape[1])
        pic[noise_mask < p / 2] = 0
        pic[(noise_mask >= p / 2) == (noise_mask < p)] = 1
    # 还原alpha通道
    if (not alpha) and (alpha_channel is not None):
        pic[:, :, 3] = alpha_channel
    return pic


def im2float(pic: np.ndarray):
    """
    将图片归一化
    :param pic: 8位色深的图片，支持灰度图、彩色图
    :return: 归一化后的图片
    """
    if pic.dtype == 'uint8':
        return pic / 255
    else:
        return pic


def gray_transform(pic, l):
    """
    将输入灰度图像的色阶数减少到L阶
    :param pic: 灰度图片的二维数组
    :param l: 要减少到的色阶数
    :return: 色阶减少后的图片二维数组
    """
    mx = max([item for ls in pic for item in ls])
    mn = min([item for ls in pic for item in ls])
    step = (mx - mn)/(l - 1)
    pic = [[round((item - mn) / step) * step + mn for item in ls] for ls in pic]
    return pic


def im_limit(pic: np.ndarray):
    """
    将图片色值上下限削平
    :param pic: 图片
    :return: 结果
    """
    pic = im2float(pic)
    pic[pic > 1] = 1
    pic[pic < 0] = 0
    return pic


def rgb2hsi(rgb):
    """
    这是将RGB彩色图像转化为HSI图像的函数
    :param rgb: RGB彩色图像
    :return: HSI图像
    """
    # 归一化
    rgb = im2float(rgb)
    # 保存原始图像的行列数
    row = np.shape(rgb)[0]
    col = np.shape(rgb)[1]
    # 对图像进行通道拆分
    R, G, B = cv2.split(rgb)

    # 计算I通道
    I = (R + G + B) / 3.0

    # 计算S通道
    # 当R=G=B=0时，会出现x/0的问题
    # 办法：将R=G=B=0的像素的S值赋值为0，并不予计算
    S = np.zeros((row, col))  # 定义S通道
    i0 = (R + B + G == 0)
    i1 = ~i0
    S[i1] = 1 - np.minimum(np.minimum(R[i1], G[i1]), B[i1]) * 3 / (R[i1] + B[i1] + G[i1])
    S[i0] = 0

    # 计算H通道
    H = np.zeros((row, col))  # 定义H通道
    i0 = ((R == G) & (G == B)) | i0  # 避免x/0的情况
    i1 = ~i0
    # 带入公式计算
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    temp = 0.5 * (R[i1] - B[i1] + R[i1] - G[i1]) / den[i1]
    temp[temp > 1] = 1
    H[i1] = np.arccos(temp) / 2 / np.pi  # 计算夹角
    H[i0] = 0
    # 公式的第二种情况
    H[G < B] = 1 - H[G < B]
    # 将HSI通道合并
    hsi = np.zeros((row, col, 3))
    hsi[:, :, 0] = H
    hsi[:, :, 1] = S
    hsi[:, :, 2] = I
    return hsi


def hsi2rgb(hsi: np.ndarray):
    """
    这是将HSI图像转化为RGB图像的函数
    :param hsi: HSI彩色图像
    :return: RGB图像
    """
    # 把通道归一化到[0,1]
    hsi = im2float(hsi)
    # 对原始图像进行复制
    rgb = hsi.copy()
    # 对图像进行通道拆分
    H, S, I = cv2.split(hsi)
    R = np.zeros([hsi.shape[0], hsi.shape[1]])
    G = R.copy()
    B = R.copy()
    # 还原色相值为角度值
    H = H * 2 * np.pi
    # H大于等于0小于120度时
    idx1 = H >= 0
    idx2 = H < 2 * np.pi / 3
    idx = idx1 & idx2  # 第一种情况的花式索引
    tmp = np.cos(np.pi / 3 - H[idx])
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / tmp)
    G[idx] = 3 * I[idx] - R[idx] - B[idx]
    # H大于等于120度小于240度
    idx1 = H >= 2 * np.pi / 3
    idx2 = H < 4 * np.pi / 3
    idx = idx1 & idx2  # 第二种情况的花式索引
    tmp = np.cos(np.pi - H[idx])
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx] - 2 * np.pi / 3) / tmp)
    B[idx] = 3 * I[idx] - R[idx] - G[idx]
    # H大于等于240度小于360度
    idx1 = H >= 4 * np.pi / 3
    idx2 = H < 2 * np.pi
    idx = idx1 & idx2  # 第三种情况的花式索引
    tmp = np.cos(5 * np.pi / 3 - H[idx])
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx] - 4 * np.pi / 3) / tmp)
    R[idx] = 3 * I[idx] - G[idx] - B[idx]
    rgb[:, :, 0] = R
    rgb[:, :, 1] = G
    rgb[:, :, 2] = B
    rgb[rgb > 1] = 1
    return rgb


def histogram(img: np.ndarray, step: int = 16):
    """
    用于显示图像的直方图，会调用matplotlib.pyplot.stem()
    :param img:图片的灰度二维数组
    :param step:直方图的细密程度
    :return:
    """
    y = list(range(step))
    x = np.array(y)/step
    cr = list(x).copy()  # color range, 用来分段表示亮度范围的数组
    cr.append(2)  # 这里添加2而不是1，为了避免后续运算中，当亮度值为1时，无法被统计的情况
    for n in range(step):
        r = (img >= cr[n]) == (img < cr[n+1])
        y[n] = np.count_nonzero(r)
    x = x + 1/step/2
    plt.stem(x, y)
