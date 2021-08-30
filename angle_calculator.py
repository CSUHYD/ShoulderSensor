import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from math import sin, cos, acos, pi, degrees
import re
from tqdm import tqdm

# 获得一个控制点
def get_point(name, df, frame):
    sr = df.iloc[frame]
    point = sr[[f'{name}', f'{name}.1', f'{name}.2']].to_numpy()
    return point

# 获得一条线段向量
def get_line(start, end, df, frame):
    sr = df.iloc[frame]
    p_start = sr[[f'{start}', f'{start}.1', f'{start}.2']].to_numpy()
    p_end = sr[[f'{end}', f'{end}.1', f'{end}.2']].to_numpy()
    line = np.array([p_start, p_end])
    return line

# 计算向量夹角
def get_vector_angle(A, B, C):
    '''
    计算空间向量AB, AC 之间的夹角 
    '''
    AB = B - A
    AC = C - A

    # 分别计算两个向量的模：
    norm_AB = np.sqrt(AB.dot(AB))
    norm_AC = np.sqrt(AC.dot(AC))
    # 计算两个向量的点积
    dot_ = AB.dot(AC)
    # 计算夹角的cos值：
    cos_ = dot_ / (norm_AB * norm_AC)
    # 求得夹角（弧度制）：
    angle_rad = np.arccos(cos_)
    # 转换为角度值：
    angle_deg = angle_rad*180/np.pi
    if angle_deg > 90:
        angle_deg = 180 - angle_deg

    return round(angle_deg, 2)

# 已知三点求平面法向量
def get_normal_vector(A, B, C):
    # 法向量计算
    # N = AB X AC
    # na = (y2-y1)*(z3-z1) - (y3-y1)*(z2-z1)
    # nb = (z2-z1)*(x3-x1) - (z3-z1)*(x2-x1)
    # nc = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)

    na = (B[1] - A[1])*(C[2] - A[2]) - (C[1] - A[1])*(B[2] - A[2])
    nb = (B[2] - A[2])*(C[0] - A[0]) - (C[2] - A[2])*(B[0] - A[0])
    nc = (B[0] - A[0])*(C[1] - A[1]) - (C[0] - A[0])*(B[1] - A[1])

    N = np.array([na, nb, nc])
    return N

# 根据两个平面的法向量求平面夹角
def get_plain_angle(N1, N2):
    x1, y1, z1 = N1
    x2, y2, z2 = N2
    # 平面向量的二面角公式
    cos_theta = ((x1*x2)+(y1*y2)+(z1*z2)) / \
        (((x1**2+y1**2+z1**2)**0.5)*((x2**2+y2**2+z2**2)**0.5))
    degree = degrees(acos(cos_theta))
    if degree > 90:  # 二面角∈[0°,180°] 但两个平面的夹角∈[0°,90°]
        degree = 180 - degree
    return round(degree, 2)


def angle_cal_frame(df, frame):
    # 建立坐标系
    O = np.array([0., 0., 0.])
    # 胸廓坐标系
    # 求Yt
    mp_XP_T8 = (get_point('xp', df, frame) + get_point('t8', df, frame)) / 2
    mp_SN_C7 = (get_point('sn', df, frame) + get_point('c7', df, frame)) / 2
    # 平移
    Yt = mp_SN_C7 - mp_XP_T8
    Yt_SN = np.array([get_point('sn', df, frame),
                     get_point('sn', df, frame) + Yt])
    Yt_SN_O = Yt_SN - Yt_SN[0, :]
    # 求Zt
    Zt = get_normal_vector(mp_XP_T8, get_point(
        'sn', df, frame), get_point('c7', df, frame))
    Zt_SN = np.array([get_point('sn', df, frame),
                     get_point('sn', df, frame) + (Zt / 200)])
    Zt_SN_O = Zt_SN - Zt_SN[0, :]
    # 求Xt
    Xt = get_normal_vector(Yt_SN[0], Yt_SN[1], Zt_SN[1])
    Xt_SN = np.array([get_point('sn', df, frame),
                     get_point('sn', df, frame) + (Xt / 200)])
    Xt_SN_O = Xt_SN - Xt_SN[0, :]

    # 肩胛骨坐标系
    # 求Zs
    Zs = get_point('aa', df, frame) - get_point('ts', df, frame)
    Zs_AA = np.array([get_point('aa', df, frame),
                     get_point('aa', df, frame) + Zs])
    Zs_AA_O = Zs_AA - Zs_AA[0, :]
    # 求Xs
    Xs = get_normal_vector(get_point('aa', df, frame), get_point(
        'ai', df, frame), get_point('ts', df, frame))
    Xs_AA = np.array([get_point('aa', df, frame),
                     get_point('aa', df, frame) + (Xs / 200)])
    Xs_AA_O = Xs_AA - Xs_AA[0, :]
    # 求Ys
    Ys = get_normal_vector(Zs_AA[0], Zs_AA[1], Xs_AA[1])
    Ys_AA = np.array([get_point('aa', df, frame),
                     get_point('aa', df, frame) + (Ys / 200)])
    Ys_AA_O = Ys_AA - Ys_AA[0, :]

    # 肱骨坐标系
    # 求Yh
    mp_LE_ME = (get_point('le', df, frame) + get_point('me', df, frame)) / 2
    GH = (get_point('gh1', df, frame) + get_point('gh2', df, frame)) / 2
    Yh = GH - mp_LE_ME
    Yh_GH = np.array([GH, GH + Yh])
    Yh_GH_O = Yh_GH - Yh_GH[0, :]
    # 求Xh
    Xh = get_normal_vector(GH, get_point(
        'le', df, frame), get_point('me', df, frame))
    Xh_GH = np.array([GH, GH + (Xh / 200)])
    Xh_GH_O = Xh_GH - Xh_GH[0, :]
    # 求Zh
    Zh = get_normal_vector(Yh_GH[0], Yh_GH[1], Xh_GH[1]) * -1
    Zh_GH = np.array([GH, GH + (Zh / 200)])
    Zh_GH_O = Zh_GH - Zh_GH[0, :]

    # 计算欧拉角
    # 肩胛骨相对于胸廓
    AA_SN_X = get_vector_angle(O, Xt_SN_O[1], Xs_AA_O[1])
    AA_SN_Y = get_vector_angle(O, Yt_SN_O[1], Ys_AA_O[1])
    AA_SN_Z = get_vector_angle(O, Zt_SN_O[1], Zs_AA_O[1])

    # print('肩胛骨相对于胸廓:')
    # print(f'x夹角 = {AA_SN_X}°')
    # print(f'y夹角 = {AA_SN_Y}°')
    # print(f'z夹角 = {AA_SN_Z}°')

    # 肱骨相对于肩胛骨
    GH_AA_X = get_vector_angle(O, Xs_AA_O[1], Xh_GH_O[1])
    GH_AA_Y = get_vector_angle(O, Ys_AA_O[1], Yh_GH_O[1])
    GH_AA_Z = get_vector_angle(O, Zs_AA_O[1], Zh_GH_O[1])

    # print('肱骨相对于肩胛骨:')
    # print(f'x夹角 = {GH_AA_X}°')
    # print(f'y夹角 = {GH_AA_Y}°')
    # print(f'z夹角 = {GH_AA_Z}°')

    # 肱骨相对于胸廓
    GH_SN_X = get_vector_angle(O, Xt_SN_O[1], Xh_GH_O[1])
    GH_SN_Y = get_vector_angle(O, Yt_SN_O[1], Yh_GH_O[1])
    GH_SN_Z = get_vector_angle(O, Zt_SN_O[1], Zh_GH_O[1])

    # print('肱骨相对于胸廓: ')
    # print(f'x夹角 = {GH_SN_X}°')
    # print(f'y夹角 = {GH_SN_Y}°')
    # print(f'z夹角 = {GH_SN_Z}°')

    return pd.DataFrame({'AA_SN_X': [AA_SN_X], 'AA_SN_Y': [AA_SN_Y], 'AA_SN_Z': [AA_SN_Z],
                         'GH_AA_X': [GH_AA_X], 'GH_AA_Y': [GH_AA_Y], 'GH_AA_Z': [GH_AA_Z],
                         'GH_SN_X': [GH_SN_X], 'GH_SN_Y': [GH_SN_Y], 'GH_SN_Z': [GH_SN_Z]})

def angle_cal_csv(df, save_path):
    """
    计算csv每一行的角度，并保存在新的csv文件中
    """
    size = df.shape[0]
    result = angle_cal_frame(df, 0)
    for i in tqdm(range(1, size)):
        result = result.append(angle_cal_frame(df, i))
    result.to_csv(save_path, index=False)

    return result


if __name__ == '__main__':
    # read csv
    csv_path = 'data/backup/angle_test/houshen_1.csv'
    save_path = 'data/backup/angle_test/houshen_1_angle.csv'

    df = pd.read_csv(csv_path)
    num_sample = df.shape[0]
    df = df.dropna()
    point_names = df.columns.values[2:][::3]
    n_points = len(point_names)
    print(f'清洗数据：({df.shape[0]}/{num_sample})')
    print('原始列名：', df.columns.values)
    print('控制点点名：', point_names)
    print('控制点个数：', n_points)

    result = angle_cal_csv(df, save_path)
    print(result)