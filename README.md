# LT-RL

Long Term Reprojection Loss

# training

```
python train_end_to_end.py --data_path <path/to/dataset> --log_dir  <path/to/output/log> --load_weights_folder <path/to/weight/folder>
```

# download weights

```
weights_url = 'https://drive.google.com/uc?id=1YIorWkIEgx_O052kfYq9mqmJJS-Z3BnH'
gdown.download(weights_url,'af-sfmlearner.zip',quiet=True)
!unzip -q af-sfmlearner.zip -d af-sfmlearner
```

# download data

# evaluate all to get 3D reconstruction

```
python eval_all.py
```

# evaluation on depth

```
python evaluate_depth.py --data_path <path/to/dataset> --load_weights_folder <path/to/weight/folder>  --eval_mono
```

# evaluation on pose

```
python evaluate_pose.py --data_path <path/to/dataset> --load_weights_folder <path/to/weight/folder> --eval_mono
```

# visualize pose

```
python visualize_pose.py
```

# visualize 3d reconstruction

```
python visualize_reconstruction.py --data_path <path/to/dataset> --load_weights_folder <path/to/weight/folder> --eval_mono
```
