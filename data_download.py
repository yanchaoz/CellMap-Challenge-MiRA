import json
import numpy as np
from fibsem_tools.io import read_xarray
import imageio
import dask.array as da
import os
import pandas as pd
import ast
import logging
import concurrent.futures

# 设置日志记录（可选）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 读取CSV文件
df = pd.read_csv('train_crop_manifest.csv')

def process_roi(roi):
    # if roi == 'jrc_cos7-1a_crop234':
    if True:
        parts = roi.split('_')
        identifier = parts[0] + '_' + parts[1]
        crop = parts[-1]

        # 读取label和em的元数据
        with open(f'./label_keys/{roi}', 'r') as file:
            metadata = json.load(file)['multiscales'][0]

        with open(f'./em_keys/{identifier}', 'r') as file:
            em_metadata = json.load(file)['multiscales'][0]

        # 查找对应的crop_name的voxel_size
        result_row = df[(df['dataset'] == identifier) & (df['crop_name'] == int(crop[4:]))]
        voxel_size_str = result_row.iloc[0]['voxel_size']
        voxel_size_str = voxel_size_str.replace(";", ",")
        voxel_size = ast.literal_eval(voxel_size_str)

        # 计算坐标变换的scales和offset
        scales = []
        offset = []
        flag = 0
        for ds in metadata["datasets"]:
            if ds['coordinateTransformations'][0]['scale'][0] == voxel_size[-1]:
                for transform in ds["coordinateTransformations"]:
                    if transform["type"] == "scale":
                        scales.extend(transform["scale"])
                    elif transform["type"] == "translation":
                        offset.extend(transform["translation"])
                label_scale = ds["path"]
                break

        for ds in em_metadata["datasets"]:
            if ds['coordinateTransformations'][0]['scale'][0] == voxel_size[-1]:
                em_scale = ds["path"]
                flag = 1
                break

        if flag == 0:
            scales = []
            offset = []
            for ds in metadata["datasets"]:
                if ds['coordinateTransformations'][0]['scale'][0] == voxel_size[-1] * 2:
                    for transform in ds["coordinateTransformations"]:
                        if transform["type"] == "scale":
                            scales.extend(transform["scale"])
                        elif transform["type"] == "translation":
                            offset.extend(transform["translation"])
                    label_scale = ds["path"]
                    break

            for ds in em_metadata["datasets"]:
                if ds['coordinateTransformations'][0]['scale'][0] == voxel_size[-1] * 2:
                    em_scale = ds["path"]
                    flag = 1
                    break

        # 计算偏移量
        voxel_offset = np.array(offset) / np.array(scales)
        voxel_offset = [int(i) for i in voxel_offset]

        # 读取zarr数据并保存为tif
        creds = {'anon': True}
        group_url = f's3://janelia-cellmap-fg5f2y1pl8/{identifier}/{identifier}.zarr/recon-1/groundtruth/{crop}/all/{label_scale}'
        group = read_xarray(group_url, storage_options=creds)
        data = group.compute().data
        imageio.volwrite(f'./down_load_csv_cmc/{roi}_gt.tif', np.array(data))

        group_url_raw = f's3://janelia-cosem-datasets/{identifier}/{identifier}.zarr/recon-1/em/fibsem-uint8/{em_scale}'
        group_raw = read_xarray(group_url_raw, storage_options={'anon': True})
        result_raw = da.from_array(group_raw, chunks=group_raw.chunks)
        data_raw = result_raw[voxel_offset[0]:voxel_offset[0]+ data.shape[0], voxel_offset[1]:voxel_offset[1]+ data.shape[1], voxel_offset[2]:voxel_offset[2]+ data.shape[2]].compute().data
        imageio.volwrite(f'./down_load_csv_cmc/{roi}.tif', np.array(data_raw))


# 使用多线程处理所有的ROI
def process_all_rois():
    with concurrent.futures.ThreadPoolExecutor(8) as executor:
        rois = os.listdir('./label_keys/')
        # 提交每个roi的处理任务
        futures = [executor.submit(process_roi, roi) for roi in rois]
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # 获取执行结果，如果有异常会抛出
            except Exception as e:
                logger.error(f"Error processing ROI: {e}")

# 调用多线程处理函数
if __name__ == "__main__":
    process_all_rois()
