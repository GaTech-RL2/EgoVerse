"""Decode the four recorded policy cameras into a native-time MP4 mosaic."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import h5py
import numpy as np


root=Path(sys.argv[1]).resolve()
for task,dof in [('73_jigsaw_puzzle_assembly',54),('34_fridge_wine_interhand_pour',38)]:
    directory=root/task
    paths=list((directory/'recordings').rglob('episode_*.hdf5'))
    if len(paths)!=1:raise ValueError('Expected exactly one recorded native episode')
    path=paths[0]
    cameras=['cam_stereo_left','cam_stereo_right','cam_wrist_left','cam_wrist_right']
    with h5py.File(path,'r') as f:
        n=int(f['meta/frame_count'][()])
        if f['action/commanded'].shape!=(n,dof):raise ValueError('Recorded controls do not match native action width')
        if any(len(f['cameras/'+c+'/rgb'])!=n for c in cameras):raise ValueError('Camera/control counts differ')
        def frame(index):
            images=[cv2.imdecode(np.asarray(f['cameras/'+c+'/rgb'][index],dtype=np.uint8),cv2.IMREAD_COLOR) for c in cameras]
            if any(im is None or im.shape!=(480,640,3) for im in images):raise ValueError('Camera JPEG did not decode at native recording resolution')
            return np.concatenate([np.concatenate(images[:2],axis=1),np.concatenate(images[2:],axis=1)],axis=0)
        first=frame(0);cv2.imwrite(str(directory/'starting_image.png'),first)
        command=['/opt/homebrew/bin/ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','bgr24','-video_size','1280x960','-framerate','20','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(directory/'rollout.mp4')]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for i in range(n):process.stdin.write(frame(i).tobytes())
        finally:process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Video encoder failed')
    receipt={'source_hdf5':str(path.relative_to(root)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'frames':n,'fps':20,'camera_order_top_left_top_right_bottom_left_bottom_right':cameras,'native_physics_steps_per_control':3,'meaning':'A mosaic of recorded live policy-camera JPEGs, with no observation intervention. Playback follows simulated control time. Native per_episode.jsonl defines success; video appearance does not score the episode.'}
    (directory/'video_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'task':task,'frames':n,'duration_seconds':n/20,'video_bytes':(directory/'rollout.mp4').stat().st_size}),flush=True)
