#!/usr/bin/env python3
"""
HDF5 Action Server - Mock inference server for testing dwm_yam_client.py

Serves actions from an HDF5 demo file using the same ZMQ protocol as the 
real DWM inference server. Use this to test that dwm_yam_client.py correctly
executes actions without needing the full inference stack.

Usage:
    # Start the server
    python hdf5_action_server.py --hdf5 /path/to/demo.hdf5 --port 5555
    
    # Then run dwm_yam_client.py as normal
    python dwm_yam_client.py --server tcp://localhost:5555 --arms both

Options:
    --step-mode     Wait for Enter key before serving each action chunk
    --action-chunk  Number of actions to return per request (default: 16)
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import zmq
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


# =============================================================================
# Action Format Utilities
# =============================================================================

def convert_ypr_to_rot6d(actions: np.ndarray) -> np.ndarray:
    """Convert bimanual 14D ypr actions to 20D rot6d actions.
    
    Input format (14D):
        [left_xyz(3), left_ypr(3), left_grip(1), right_xyz(3), right_ypr(3), right_grip(1)]
        
    Output format (20D):
        [left_xyz(3), left_rot6d(6), left_grip(1), right_xyz(3), right_rot6d(6), right_grip(1)]
    """
    squeeze = False
    if actions.ndim == 1:
        actions = actions[None, :]
        squeeze = True
    if actions.shape[1] != 14:
        raise ValueError(f"Expected 14D ypr actions, got {actions.shape}")

    left_xyz = actions[:, 0:3]
    left_ypr = actions[:, 3:6]
    left_grip = actions[:, 6:7]
    right_xyz = actions[:, 7:10]
    right_ypr = actions[:, 10:13]
    right_grip = actions[:, 13:14]

    left_R = R.from_euler("ZYX", left_ypr).as_matrix()
    right_R = R.from_euler("ZYX", right_ypr).as_matrix()
    
    # 6D representation: first two columns of rotation matrix, flattened row-major
    left_rot6d = left_R[:, :, :2].reshape(-1, 6)
    right_rot6d = right_R[:, :, :2].reshape(-1, 6)

    out = np.concatenate(
        [left_xyz, left_rot6d, left_grip, right_xyz, right_rot6d, right_grip],
        axis=-1,
    )
    return out[0] if squeeze else out


# =============================================================================
# HDF5 Action Loader
# =============================================================================

class HDF5ActionLoader:
    """Load actions from HDF5 demo files."""
    
    def __init__(self, hdf5_path: str, output_format: str = "ypr"):
        """Initialize action loader.
        
        Args:
            hdf5_path: Path to HDF5 demo file
            output_format: Output format - "ypr" (14D) or "rot6d" (20D)
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        self.output_format = output_format
        self._load_demo()
    
    def _load_demo(self):
        """Load demo data from HDF5 file."""
        print(f"Loading HDF5 demo: {self.hdf5_path}")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            print("HDF5 structure:")
            self._print_structure(f, indent=2)
            
            # Load actions - try multiple possible locations
            self.actions_ypr = None
            self.actions_rot6d = None

            if 'actions' in f:
                if 'eepose_ypr' in f['actions']:
                    self.actions_ypr = f['actions']['eepose_ypr'][:]
                    print(f"  Loaded eepose_ypr: {self.actions_ypr.shape}")
                if 'eepose_6drot' in f['actions']:
                    self.actions_rot6d = f['actions']['eepose_6drot'][:]
                    print(f"  Loaded eepose_6drot: {self.actions_rot6d.shape}")
                elif 'eepose_rot6d' in f['actions']:
                    self.actions_rot6d = f['actions']['eepose_rot6d'][:]
                    print(f"  Loaded eepose_rot6d: {self.actions_rot6d.shape}")
                if 'eepose' in f['actions'] and self.actions_ypr is None:
                    self.actions_ypr = f['actions']['eepose'][:]
                    print(f"  Loaded eepose: {self.actions_ypr.shape}")

            if 'observations' in f and 'eepose' in f['observations'] and self.actions_ypr is None:
                self.actions_ypr = f['observations']['eepose'][:]
                print(f"  Loaded observations/eepose: {self.actions_ypr.shape}")

            if self.actions_ypr is None and self.actions_rot6d is None:
                raise KeyError("No eepose actions found in HDF5 file")
        
        # Prepare actions in requested format
        if self.output_format == "rot6d":
            if self.actions_rot6d is not None:
                self.actions = self.actions_rot6d
            else:
                self.actions = convert_ypr_to_rot6d(self.actions_ypr)
                print("  Converted ypr -> rot6d")
        else:  # ypr
            if self.actions_ypr is not None:
                self.actions = self.actions_ypr
            else:
                # Would need rot6d_to_ypr conversion here
                raise ValueError("rot6d -> ypr conversion not implemented, use --output-format rot6d")
        
        self.num_steps = self.actions.shape[0]
        print(f"  Total steps: {self.num_steps}")
        print(f"  Output format: {self.output_format} ({self.actions.shape[1]}D)")
    
    def _print_structure(self, group, indent=0):
        """Recursively print HDF5 structure."""
        for key in group.keys():
            item = group[key]
            prefix = " " * indent
            if isinstance(item, h5py.Group):
                print(f"{prefix}{key}/")
                if indent < 4:
                    self._print_structure(item, indent + 2)
            else:
                print(f"{prefix}{key}: {item.shape} {item.dtype}")
    
    def get_action_chunk(self, start_step: int, chunk_size: int) -> np.ndarray:
        """Get a chunk of actions starting at given step.
        
        Args:
            start_step: Starting step index
            chunk_size: Number of actions to return
            
        Returns:
            Action array (chunk_size, action_dim) or fewer if near end
        """
        end_step = min(start_step + chunk_size, self.num_steps)
        if start_step >= self.num_steps:
            return None
        return self.actions[start_step:end_step].copy()
    
    def __len__(self):
        return self.num_steps


# =============================================================================
# Mock Inference Server
# =============================================================================

class HDF5ActionServer:
    """ZMQ server that serves actions from HDF5 file."""
    
    def __init__(
        self,
        hdf5_path: str,
        port: int,
        action_chunk: int = 16,
        output_format: str = "ypr",
        step_mode: bool = False,
        loop: bool = False,
    ):
        """Initialize server.
        
        Args:
            hdf5_path: Path to HDF5 demo file
            port: ZMQ port to listen on
            action_chunk: Number of actions to return per request
            output_format: "ypr" (14D) or "rot6d" (20D)
            step_mode: If True, wait for Enter before serving each chunk
            loop: If True, loop back to start when reaching end
        """
        self.action_chunk = action_chunk
        self.step_mode = step_mode
        self.loop = loop
        
        # Load actions
        self.loader = HDF5ActionLoader(hdf5_path, output_format)
        
        # Setup ZMQ
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://*:{port}")
        print(f"\nServer listening on tcp://*:{port}")
        
        # Track state
        self.current_step = 0
        self.requests_served = 0
    
    def handle_ping(self) -> dict:
        """Handle ping request."""
        return {'status': 'ok', 'server': 'hdf5_action_server'}
    
    def handle_infer(self, msg: dict) -> dict:
        """Handle inference request.
        
        Args:
            msg: Request message with 'obs' and 'step' keys
            
        Returns:
            Response with 'actions' array
        """
        step = msg.get('step', self.current_step)
        
        # In step mode, wait for user
        if self.step_mode:
            print(f"\n[Step {step}] Press Enter to serve action chunk...")
            input()
        
        # Get action chunk
        actions = self.loader.get_action_chunk(step, self.action_chunk)
        
        if actions is None:
            if self.loop:
                print(f"Reached end of demo, looping back to start")
                self.current_step = 0
                actions = self.loader.get_action_chunk(0, self.action_chunk)
            else:
                return {'error': 'End of demo reached'}
        
        self.current_step = step + len(actions)
        self.requests_served += 1
        
        # Log what we're serving
        print(f"[Request {self.requests_served}] Step {step}: serving {len(actions)} actions "
              f"(shape {actions.shape})")
        if actions.shape[1] == 14:
            # YPR format - show first action
            print(f"  First action: left=[{actions[0,0]:.3f},{actions[0,1]:.3f},{actions[0,2]:.3f}] "
                  f"right=[{actions[0,7]:.3f},{actions[0,8]:.3f},{actions[0,9]:.3f}]")
        elif actions.shape[1] == 20:
            # rot6d format
            print(f"  First action: left_xyz=[{actions[0,0]:.3f},{actions[0,1]:.3f},{actions[0,2]:.3f}] "
                  f"right_xyz=[{actions[0,10]:.3f},{actions[0,11]:.3f},{actions[0,12]:.3f}]")
        
        return {
            'actions': actions,
            'infer_time': 0.001,  # Fake inference time
        }
    
    def run(self):
        """Run the server loop."""
        print(f"\nReady to serve {len(self.loader)} steps")
        print(f"Action chunk size: {self.action_chunk}")
        print(f"Step mode: {self.step_mode}")
        print(f"Loop mode: {self.loop}")
        print("-" * 50)
        print("Waiting for requests from dwm_yam_client.py...")
        
        try:
            while True:
                # Receive request
                raw = self.sock.recv()
                msg = msgpack.unpackb(raw, raw=False)
                
                cmd = msg.get('cmd', '')
                
                if cmd == 'ping':
                    response = self.handle_ping()
                elif cmd == 'infer':
                    response = self.handle_infer(msg)
                else:
                    response = {'error': f'Unknown command: {cmd}'}
                
                # Send response
                self.sock.send(msgpack.packb(response, use_bin_type=True))
                
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
        finally:
            self.sock.close()
            self.ctx.term()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mock inference server that serves actions from HDF5 file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - serve actions at default rate
  python hdf5_action_server.py --hdf5 demo.hdf5 --port 5555
  
  # Step-by-step mode (press Enter before each action chunk)
  python hdf5_action_server.py --hdf5 demo.hdf5 --port 5555 --step-mode
  
  # Serve rot6d format (20D) instead of ypr (14D)
  python hdf5_action_server.py --hdf5 demo.hdf5 --port 5555 --output-format rot6d
  
  # Loop back to start when demo ends
  python hdf5_action_server.py --hdf5 demo.hdf5 --port 5555 --loop

Then run dwm_yam_client.py:
  python dwm_yam_client.py --server tcp://localhost:5555 --arms both --dry-run
        """
    )
    
    parser.add_argument("--hdf5", type=str, required=True,
                        help="Path to HDF5 demo file")
    parser.add_argument("--port", type=int, default=5555,
                        help="ZMQ port to listen on")
    parser.add_argument("--action-chunk", type=int, default=16,
                        help="Number of actions to return per request")
    parser.add_argument("--output-format", type=str, default="ypr",
                        choices=["ypr", "rot6d"],
                        help="Action output format: ypr (14D) or rot6d (20D)")
    parser.add_argument("--step-mode", action="store_true",
                        help="Wait for Enter before serving each action chunk")
    parser.add_argument("--loop", action="store_true",
                        help="Loop back to start when reaching end of demo")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HDF5 Action Server")
    print("=" * 60)
    print(f"HDF5 file:     {args.hdf5}")
    print(f"Port:          {args.port}")
    print(f"Action chunk:  {args.action_chunk}")
    print(f"Output format: {args.output_format}")
    print(f"Step mode:     {args.step_mode}")
    print(f"Loop:          {args.loop}")
    print("=" * 60)
    
    server = HDF5ActionServer(
        hdf5_path=args.hdf5,
        port=args.port,
        action_chunk=args.action_chunk,
        output_format=args.output_format,
        step_mode=args.step_mode,
        loop=args.loop,
    )
    
    server.run()


if __name__ == "__main__":
    main()
