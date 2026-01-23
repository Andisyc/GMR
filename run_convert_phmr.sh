# python scripts/convert_phmr_to_smplx.py --input /hhd4/lizhe/stcomp/GMR/motion_data/Phmr/results.pkl --output /hhd4/lizhe/stcomp/GMR/motion_data/Phmr/dance_z_r90.pkl 
# python scripts/convert_phmr_to_smplx.py --input /hhd4/lizhe/stcomp/GMR/phmr_result/boxing/results.pkl --output /hhd4/lizhe/stcomp/GMR/motion_data/Phmr/boxing_2.pkl --person_id 2
python scripts/convert_phmr_to_smplx.py --input /hhd4/lizhe/stcomp/GMR/phmr_result/dance_sh_long/results.pkl --output /hhd4/lizhe/stcomp/GMR/motion_data/Phmr/dance_sh_long_2.pkl --person_id 2
# python scripts/smplx_to_robot_dataset.py --src_folder /hhd4/lizhe/stcomp/GMR/motion_data/Phmr/smplx/ --tgt_folder /hhd4/lizhe/stcomp/robot_data/UnitreeG1/phmr/ --robot unitree_g1 --num_cpus 24
# python scripts/smplx_to_robot_dataset.py --src_folder /hhd4/lizhe/stcomp/motion_data/AMASS/CNRS --tgt_folder /hhd4/lizhe/stcomp/robot_data/UnitreeG1/CNRS --robot unitree_g1 --num_cpus 24
