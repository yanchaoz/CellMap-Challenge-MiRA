import json
import numpy as np
import pandas as pd
import ast
import imageio
import dask.array as da
import os
import logging
import concurrent.futures
from fibsem_tools.io import read_xarray

# 设置日志记录（可选）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 读取CSV文件
df = pd.read_csv('train_crop_manifest.csv')


def get_voxel_size(crop_name, identifier):
    """获取对应crop_name的voxel_size"""
    result_row = df[(df['dataset'] == identifier) & (df['crop_name'] == int(crop_name[4:]))]
    voxel_size_str = result_row.iloc[0]['voxel_size'].replace(";", ",")
    return ast.literal_eval(voxel_size_str)


def get_scales_and_offset(metadata, voxel_size):
    """从metadata中提取scale和offset"""
    scales, offset = [], []
    for ds in metadata["datasets"]:
        for transform in ds["coordinateTransformations"]:
            if transform["type"] == "scale" and transform["scale"][0] == voxel_size[-1]:
                scales.extend(transform["scale"])
            elif transform["type"] == "translation":
                offset.extend(transform["translation"])
    return scales, offset


def get_em_scale_and_metadata(em_metadata, voxel_size):
    """从em_metadata中查找对应的em_scale"""
    for ds in em_metadata["datasets"]:
        if ds['coordinateTransformations'][0]['scale'][0] == voxel_size[-1]:
            return ds["path"]
    return None


def process_roi(roi):
    """处理单个ROI"""
    # 提取identifier和crop信息
    identifier, crop = roi.split('_')[0] + '_' + roi.split('_')[1], roi.split('_')[-1]

    # 读取label和em的元数据
    with open(f'./label_keys/{roi}', 'r') as file:
        label_metadata = json.load(file)['multiscales'][0]

    with open(f'./em_keys/{identifier}', 'r') as file:
        em_metadata = json.load(file)['multiscales'][0]

    # 获取voxel_size
    voxel_size = get_voxel_size(crop, identifier)

    # 获取label坐标变换的scales和offset
    scales, offset = get_scales_and_offset(label_metadata, voxel_size)

    # 获取em_scale
    em_scale = get_em_scale_and_metadata(em_metadata, voxel_size)

    # 如果没有找到合适的scales和offset，则尝试倍增尺度
    if not scales or not em_scale:
        scales, offset = get_scales_and_offset(label_metadata, voxel_size * 2)
        em_scale = get_em_scale_and_metadata(em_metadata, voxel_size * 2)

    # 计算体素偏移
    voxel_offset = np.array(offset) / np.array(scales)
    voxel_offset = [int(i) for i in voxel_offset]

    # 读取并保存GT数据
    creds = {'anon': True}
    label_url = f's3://janelia-cellmap-fg5f2y1pl8/{identifier}/{identifier}.zarr/recon-1/groundtruth/{crop}/all/{label_metadata["path"]}'
    label_group = read_xarray(label_url, storage_options=creds)
    label_data = label_group.compute().data
    imageio.volwrite(f'./down_load_csv_cmc/{roi}_gt.tif', np.array(label_data))

    # 读取并保存EM数据
    em_url = f's3://janelia-cosem-datasets/{identifier}/{identifier}.zarr/recon-1/em/fibsem-uint8/{em_scale}'
    em_group = read_xarray(em_url, storage_options={'anon': True})
    em_data_raw = da.from_array(em_group, chunks=em_group.chunks)
    em_data = em_data_raw[voxel_offset[0]:voxel_offset[0] + label_data.shape[0],
                          voxel_offset[1]:voxel_offset[1] + label_data.shape[1],
                          voxel_offset[2]:voxel_offset[2] + label_data.shape[2]].compute().data
    imageio.volwrite(f'./down_load_csv_cmc/{roi}.tif', np.array(em_data))


def process_all_rois():
    """并行处理所有ROI"""
    rois = os.listdir('./label_keys/')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # 提交每个ROI的处理任务
        futures = {executor.submit(process_roi, roi): roi for roi in rois}
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            roi = futures[future]
            try:
                future.result()  # 获取执行结果，若有异常会抛出
            except Exception as e:
                logger.error(f"Error processing ROI {roi}: {e}")


if __name__ == "__main__":
    process_all_rois()
