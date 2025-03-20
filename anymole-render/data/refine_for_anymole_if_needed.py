import torch
import numpy as np
import glob


all_motions =glob.glob('characters/*/*.npz')

for single_motion in all_motions:
    motion = np.load(single_motion)
    
    assert motion.files == ['local_quats', 'root_pos', 'pre_quats', 'local_pos', 'parent_idx', 'joint_names'], "Wrong motion data"
    
    motion2refine = ['local_quats', 'root_pos']
    refined_data = {key: motion[key] for key in motion.files}

    ### motion refinment stage ###
    original_len = refined_data[motion2refine[0]].shape[0]
    assert refined_data[motion2refine[1]].shape[0] == original_len, "motion length error root_pos"
    
    refinedframe = (original_len - 61) // 30

    if (original_len%30==0):
        lfrm = refined_data[motion2refine[0]][-1:]
        refined_data[motion2refine[0]] = np.concatenate([refined_data[motion2refine[0]], lfrm], axis=0)
        lfrm = refined_data[motion2refine[1]][-1:]
        refined_data[motion2refine[1]] = np.concatenate([refined_data[motion2refine[1]], lfrm], axis=0)
    else:
        refined_data[motion2refine[0]] = refined_data[motion2refine[0]][:refinedframe*30+61]
        refined_data[motion2refine[1]] = refined_data[motion2refine[1]][:refinedframe*30+61]
    
    np.savez(single_motion.replace(".npz", "_gt.npz"), **refined_data)


    for i in range(refinedframe):
        for j in range(1, 30):
            if j < 15:
                # Forward: Set frames to the current key frame
                refined_data[motion2refine[0]][60 + 30 * i + j] = refined_data[motion2refine[0]][60 + 30 * i]
            else:
                # Backward: Set frames to the next key frame
                refined_data[motion2refine[0]][60 + 30 * i + j] = refined_data[motion2refine[0]][60 + 30 * (i + 1)]
    
        np.savez(single_motion.replace(".npz", "_input.npz"), **refined_data)
        ### inbetweening source stage ###

