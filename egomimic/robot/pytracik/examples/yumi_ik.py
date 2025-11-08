""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20220811osaka

"""

if __name__ == '__main__':
    import os
    import numpy as np
    from trac_ik import TracIK

    urdf_path = os.path.join(os.path.dirname(__file__), "urdf/yumi.urdf")

    yumi_rgt_arm_iksolver = TracIK(base_link_name="yumi_body",
                                   tip_link_name="yumi_link_7_r",
                                   urdf_path=urdf_path, )
    yumi_lft_arm_iksolver = TracIK(base_link_name="yumi_body",
                                   tip_link_name="yumi_link_7_l",
                                   urdf_path=urdf_path, )
    seed_jnt = np.array([-0.34906585, -1.57079633, -2.0943951, 0.52359878, 0.,
                         0.6981317, 0.])
    tgt_pos = np.array([.3, -.4, .1])
    tgt_rotmat = np.array([[0.5, 0., 0.8660254],
                           [0., 1., 0.],
                           [-0.8660254, 0., 0.5]])
    result = yumi_rgt_arm_iksolver.ik(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt)
    print("The ik solution is", result)

    pos_fk, rot_fk = yumi_rgt_arm_iksolver.fk(result)
    print("The fk result is", pos_fk, rot_fk)
