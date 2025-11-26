import json
import numpy as np
import pandas as pd
import ast
import imageio
import dask.array as da
from fibsem_tools.io import read_xarray
import os

# 读取 CSV 文件
df = pd.read_csv('test_crop_manifest.csv')

# 定义要处理的标识符列表
identifiers = [
    'jrc_ctl-id8-1', 'jrc_fly-vnc-1', 'jrc_mus-kidney',
    'jrc_mus-liver-zon-2', 'jrc_mus-nacc-1', 'jrc_zf-cardiac-1'
]

# 遍历所有标识符
for identifier in identifiers:
    # 获取当前标识符的所有唯一的 crop_name
    result_rows = df[df['dataset'] == identifier]
    unique_crop_names = result_rows['crop_name'].drop_duplicates().tolist()

    for crop in unique_crop_names:
        # 根据 dataset 和 crop_name 筛选对应的数据行
        result_row = result_rows[result_rows['crop_name'] == int(crop)]

        # 解析体素大小
        voxel_size_str = result_row.iloc[0]['voxel_size'].replace(";", ",")
        voxel_size = ast.literal_eval(voxel_size_str)

        # 加载 em_metadata
        with open(f'./em_keys/{identifier}', 'r') as file:
            em_metadata = json.load(file)['multiscales'][0]

        # 初始化标志和尺度信息
        scales, flag = [], 0

        # 查找对应的 em_scale 和尺度
        for ds in em_metadata["datasets"]:
            if ds['coordinateTransformations'][0]['scale'][0] == voxel_size[-1]:
                em_scale = ds["path"]
                flag = 1
                for transform in ds["coordinateTransformations"]:
                    if transform["type"] == "scale":
                        scales.extend(transform["scale"])

        # 如果没有找到合适的尺度，尝试放大尺度
        if flag == 0:
            for ds in em_metadata["datasets"]:
                if ds['coordinateTransformations'][0]['scale'][0] == voxel_size[-1] * 2:
                    em_scale = ds["path"]
                    for transform in ds["coordinateTransformations"]:
                        if transform["type"] == "scale":
                            scales.extend(transform["scale"])

        # 解析偏移量
        offset_str = result_row.iloc[0]['translation'].replace(";", ",")
        offset = ast.literal_eval(offset_str)

        # 计算体素偏移
        voxel_offset = np.array(offset) / np.array(scales)
        voxel_offset = [int(i) for i in voxel_offset]

        # 解析形状
        shape_str = result_row.iloc[0]['shape'].replace(";", ",")
        shape = ast.literal_eval(shape_str)

        # 构建原始数据 URL
        group_url_raw = f's3://janelia-cosem-datasets/{identifier}/{identifier}.zarr/recon-1/em/fibsem-uint8/{em_scale}/'
        group_raw = read_xarray(group_url_raw, storage_options={'anon': True})
        result_raw = da.from_array(group_raw, chunks=group_raw.chunks)

        # 根据标志和体素偏移量截取数据
        if flag == 0:
            data_raw = result_raw[voxel_offset[0]:voxel_offset[0] + shape[0] // 2,
                                  voxel_offset[1]:voxel_offset[1] + shape[1] // 2,
                                  voxel_offset[2]:voxel_offset[2] + shape[2] // 2].compute().data
        else:
            data_raw = result_raw[voxel_offset[0]:voxel_offset[0] + shape[0],
                                  voxel_offset[1]:voxel_offset[1] + shape[1],
                                  voxel_offset[2]:voxel_offset[2] + shape[2]].compute().data

        # 保存数据到文件
        output_path = f'./down_load_csv_cmc_test/{identifier}_crop{crop}.tif'
        imageio.volwrite(output_path, np.array(data_raw))
