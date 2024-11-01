'''
Author: qiuyi.ye qiuyi.ye@maestrosurgical.com
Date: 2024-10-12 10:29:49
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-10-14 10:57:50
FilePath: /xushaokang/Single_AI_Registration2.0_BN6/point2plane.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
def project_point_onto_plane_AP(P, S, plane):
    """
    计算点 P 投影到垂直于 y 轴的平面上的位置
    
    参数:
    P (tuple): 点 P 的坐标 (x_p, y_p, z_p)
    S (tuple): 投影点 S 的坐标 (x_s, y_s, z_s)
    y_plane (float): 投影平面的 y 坐标

    返回:
    tuple: 点 P 投影后的坐标 (x_proj, y_plane, z_proj)
    """
    x_p, y_p, z_p = P
    x_s, y_s, z_s = S

    # 计算比例系数 lambda
    lambda_factor = (plane - y_p) / (y_p - y_s)

    # 计算投影点的坐标
    x_proj = x_p + lambda_factor * (x_p - x_s)
    z_proj = z_p + lambda_factor * (z_p - z_s)

    # 返回投影点的坐标，y 坐标为 y_plane
    return (x_proj, plane, z_proj)

def project_point_onto_plane_LAT(P, S, plane):
    """
    计算点 P 投影到垂直于 y 轴的平面上的位置
    
    参数:
    P (tuple): 点 P 的坐标 (x_p, y_p, z_p)
    S (tuple): 投影点 S 的坐标 (x_s, y_s, z_s)
    y_plane (float): 投影平面的 y 坐标

    返回:
    tuple: 点 P 投影后的坐标 (x_proj, y_plane, z_proj)
    """
    x_p, y_p, z_p = P
    x_s, y_s, z_s = S

    # 计算比例系数 lambda
    lambda_factor = (plane - x_p) / (x_p - x_s)

    # 计算投影点的坐标
    y_proj = y_p +  lambda_factor * (y_p - y_s)
    z_proj = z_p +  lambda_factor * (z_p - z_s)

    # 返回投影点的坐标，y 坐标为 y_plane
    return (plane, y_proj, z_proj)

""" # 示例数据
P = (-5.1315  ,   -19.687  ,   -22.635)  # 点 P 的坐标
S_AP = (-12.14642345905304, -760.4464219331742, -20.63500213623047)  # 投影点 S 的坐标
S_LAT = (-699.746423459053, 2.7535780668258667, -14.635002136230469)  # 投影点 S 的坐标
plane_origin_AP=(-119.05, 299.554, -127.539) # 投影平面origin的坐标
plane_origin_LAT=(360.254, -104.15, -121.539) # 投影平面origin的坐标 """

def calculate_point_on_plane(Vercenter,SourceAp, SourceLat, DRRorigin_AP, DRRorigin_LAT):
    # 计算点 P 在投影平面上的投影

    SourceAp=list(SourceAp)
    SourceLat=list(SourceLat)
    DRRorigin_AP=list(DRRorigin_AP)
    DRRorigin_LAT=list(DRRorigin_LAT)

    projected_point_AP = project_point_onto_plane_AP(Vercenter, SourceAp, DRRorigin_AP[1])
    projected_point_LAT = project_point_onto_plane_LAT(Vercenter, SourceLat, DRRorigin_LAT[0])

    ImageCords_AP = [(projected_point_AP[0]-DRRorigin_AP[0])/0.209/2,   512-(projected_point_AP[2]-DRRorigin_AP[2])/0.209/2]
    ImageCords_LAT = [(projected_point_LAT[1]-DRRorigin_LAT[1])/0.209/2,    512-(projected_point_LAT[2]-DRRorigin_LAT[2])/0.209/2]

    return {"AP":ImageCords_AP,"LAT":ImageCords_LAT}
