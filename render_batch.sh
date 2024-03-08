#!/bin/bash

# rm -r /cluster/scratch/xiychen/data/thuman_head_meshes_all/prt
# rm -r /cluster/scratch/xiychen/ICON/data/thuman2_head_100_36views
# rm -r /cluster/scratch/xiychen/ICON/data/thuman2_head_100/scans/0002/prt
# rm -r debug
# rm -r /cluster/scratch/xiychen/ICON/data/thuman2_head
# rm -r /cluster/scratch/xiychen/ICON/data/thuman2_head_36views
python -m scripts.render_batch -headless 
# python -m scripts.visibility_batch