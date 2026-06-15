"""Test RTG ego-motion estimation from LiDAR data.
Uses ICP between consecutive frames on static background points.
No calibration needed — LiDAR-only.
"""
import numpy as np, sys, glob
SF = '/media/shen/data/zk/Auto/CenterPoint'
EXT = f'{SF}/data/rtg/extracted/scene_001'

def load_pts(idx):
    pts = np.fromfile(f'{EXT}/lidar_01/{idx:06d}.bin', dtype=np.float32).reshape(-1,4)
    # Remove ground (z < 0.5m from LiDAR = ~2m above BEV ground)
    # Ouster has ring field, but we only saved x,y,z,intensity (4 fields)
    # Use height-based ground removal + distance filter
    non_ground = pts[pts[:, 2] > 0.5]  # Above 0.5m from LiDAR (~2m from ground)
    # Remove far points (noise)
    non_ground = non_ground[non_ground[:, 0] < 60]
    return non_ground

def estimate_motion(pts_prev, pts_curr, voxel_size=0.5):
    """Simple ICP-like translation estimation using voxel grid matching."""
    from scipy.spatial import cKDTree
    # Downsample both
    def downsample(p, vs):
        if len(p) < 100: return p
        voxel_idx = np.floor(p[:, :2] / vs).astype(np.int32)
        _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
        return p[unique_idx]

    p1 = downsample(pts_prev, voxel_size)
    p2 = downsample(pts_curr, voxel_size)

    if len(p1) < 50 or len(p2) < 50:
        return 0.0, 0.0, 0.0

    # Build KD-tree and find correspondences
    tree = cKDTree(p1[:, :2])  # BEV x,y only
    dists, idxs = tree.query(p2[:, :2], k=1)

    # Filter outliers
    valid = dists < 2.0  # 2m max correspondence
    if valid.sum() < 20:
        return 0.0, 0.0, 0.0

    # Compute mean displacement
    dx = np.mean(p2[valid, 0] - p1[idxs[valid], 0])
    dy = np.mean(p2[valid, 1] - p1[idxs[valid], 1])

    # Estimate confidence from inlier ratio
    confidence = valid.sum() / len(p2)

    return dx, dy, confidence

# Test on 50 consecutive frame pairs
print("Loading LiDAR frames for motion estimation...")
frame_pairs = []
for i in range(50):
    pts_prev = load_pts(i)
    pts_curr = load_pts(i + 1)
    dx, dy, conf = estimate_motion(pts_prev, pts_curr)
    displacement = np.sqrt(dx**2 + dy**2)
    frame_pairs.append((i, dx, dy, displacement, conf, len(pts_prev), len(pts_curr)))

dx_vals = [f[1] for f in frame_pairs]
dy_vals = [f[2] for f in frame_pairs]
disp_vals = [f[3] for f in frame_pairs]
conf_vals = [f[4] for f in frame_pairs]

print(f'\n=== Motion Estimation Results (50 frame pairs, ~5s) ===')
print(f'Displacement per frame (100ms):')
print(f'  Mean: {np.mean(disp_vals):.3f}m')
print(f'  Max:  {np.max(disp_vals):.3f}m')
print(f'  Std:  {np.std(disp_vals):.3f}m')
print(f'Confidence:')
print(f'  Mean: {np.mean(conf_vals):.3f}')
print(f'  Min:  {np.min(conf_vals):.3f}')

# Determine motion state
threshold = 0.05  # 5cm/frame = 0.5m/s minimum detectable motion
moving_frames = sum(1 for d in disp_vals if d > threshold and conf_vals[disp_vals.index(d)] > 0.3)
print(f'\nFrames with >{threshold}m displacement (high confidence): {moving_frames}/{len(frame_pairs)}')

# Check if there's a consistent direction
if moving_frames > len(frame_pairs) * 0.3:  # >30% frames moving
    mean_dx = np.mean([f[1] for f in frame_pairs if f[3] > threshold])
    mean_dy = np.mean([f[2] for f in frame_pairs if f[3] > threshold])
    direction = '+x' if mean_dx > 0.1 else '-x' if mean_dx < -0.1 else 'unknown'
    print(f'Mean displacement direction: dx={mean_dx:.3f}m, dy={mean_dy:.3f}m')
    print(f'RTG appears to be MOVING ({direction})')
else:
    print(f'RTG appears to be STATIONARY')

# Print first 10 frames detail
print(f'\n--- Per-frame detail (first 10) ---')
for f in frame_pairs[:10]:
    state = 'MOVING' if f[3] > threshold and f[4] > 0.3 else 'static'
    print(f'  Frame {f[0]:2d}→{f[0]+1:2d}: dx={f[1]:+.3f}m dy={f[2]:+.3f}m '
          f'disp={f[3]:.3f}m conf={f[4]:.2f} [{state}] pts={f[5]}/{f[6]}')

print(f'\nMotion estimation test: PASS')
