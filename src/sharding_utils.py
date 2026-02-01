import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

def create_mesh_sharding():
    """
    Creates a simple Data Parallel sharding (Replica Sharding).
    """
    devices = jax.devices()
    n_devices = len(devices)
    print(f"[System] Found {n_devices} devices: {devices}")
    
    # Metal/Single Device Optimization: Skip Mesh/Sharding API if trivial
    if n_devices == 1:
        print("[System] Single device detected. Bypassing Sharding Mesh (using default placement).")
        # Return None or a dummy object?
        # main_scaled uses it as argument to shard_batch.
        # We can return None.
        return None, n_devices
    
    # Create a simple mesh where 'data' axis maps to all devices
    mesh = Mesh(devices, axis_names=('data',))
    
    # Sharding Spec: Split first dim (batch) across 'data' axis
    sharding = NamedSharding(mesh, PartitionSpec('data',))
    
    return sharding, n_devices

import equinox as eqx

def replicate_state(tree):
    """
    Replicates the model state (parameters) across devices using SPMD (NamedSharding).
    If single device/Metal, just moves to device.
    """
    devices = jax.devices()
    
    if len(devices) == 1:
        # Simple move to device
        return eqx.filter_jit(lambda x: x)(tree) # Ensure on device? Or just leave it.
        # jax.device_put recursively?
        # return jax.tree.map(lambda x: jax.device_put(x, devices[0]), tree)
        # Actually eqx modules are PyTrees.
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, devices[0]) if eqx.is_array(x) else x, tree)

    # 1. Define Mesh (Must match create_mesh_sharding)
    mesh = Mesh(devices, axis_names=('data',))
    
    # 2. Define Replicated Sharding (PartitionSpec is empty tuple -> Replicated)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    
    # 3. Filter arrays
    arrays, static = eqx.partition(tree, eqx.is_array)
    
    # 4. Device Put with Sharding Spec
    replicated_arrays = jax.device_put(arrays, replicated_sharding)
    
    # 5. Combine back
    return eqx.combine(replicated_arrays, static)

def shard_batch(batch, sharding):
    """
    Shards a batch of data.
    If sharding is None (Single Device), just move to device.
    """
    if sharding is None:
        devices = jax.devices()
        return jax.device_put(batch, devices[0])
        
    # Simply put the array on the devices with the sharding spec
    return jax.device_put(batch, sharding)
