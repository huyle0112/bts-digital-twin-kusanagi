Novel View Synthesis Competition Data
======================================

Thư mục cấu trúc:
├── train/
│   ├── images/          : Ảnh training
│   ├── sparse/0/        : Sparse reconstruction từ COLMAP (cameras.bin, images.bin, points3D.bin)
└── test/
    ├── images/          : Ảnh test
    └── test_poses.csv   : Camera poses cho test images (CSV format)

Thông tin:
- Train images: 200 images
- Test images: 50 images
- Scale factor: 1/4 (original size / 4)
- Tỷ lệ train/test: 80/20

Format test_poses.csv:
CSV file with columns: image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
  image_name: image filename
  qw, qx, qy, qz: quaternion (w,x,y,z) - camera rotation
  tx, ty, tz: translation (x,y,z) - camera position
  fx, fy: focal length
  cx, cy: principal point
  width, height: desired render resolution
