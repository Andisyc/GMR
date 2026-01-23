# python convert_pkl_to_csv.py /hhd4/lizhe/stcomp/robot_data/AMASS/ACCAD/Female1General_c3d/A6-_lift_box_t2_stageii.pkl -o ./csv/A6-_lift_box_t2_stageii.csv -d 25
# python convert_pkl_to_csv.py /hhd4/lizhe/stcomp/robot_data/hightorque_hi/Phmr/results_converted_x90.pkl -o ./csv/phmr_dance.csv -d 25
# python convert_pkl_to_csv.py /hhd4/lizhe/stcomp/robot_data/UnitreeG1/phmr/results_converted_x90.pkl  -o ./csv/g1_phmr_dance.csv -d 29
# python ./scripts/convert_pkl_to_csv.py /hhd4/lizhe/stcomp/robot_data/UnitreeG1/boxing.pkl.pkl  -o ./csv/g1_boxing_1.csv -d 29
# python convert_pkl_to_csv.py /hhd4/lizhe/stcomp/robot_data/UnitreeG1/CNRS/283/01_R_2_stageii.pkl   -o ./csv/g1_CNRS_01_R_2_stageii.csv -d 29

### Convert oakink2 data (all together)
# python ./scripts/convert_pkl_to_csv.py /hhd4/lizhe/stcomp/GMR/motion_data/oakink2/scene_01__A001++seq__6e11d637b13834d44e77__2023-04-27-19-22-25.npz.pkl  -o ./csv_data/scene_01__A001++seq__6e11d637b13834d44e77__2023-04-27-19-22-25.csv -d 41

### Convert oakink2 data (gmr+dex)
python ./scripts/convert_pkl_to_csv.py /hhd4/lizhe/stcomp/GMR/motion_data/oakink2/scene_01__A001++seq__6e11d637b13834d44e77__2023-04-27-19-22-25_robot.pkl  -o ./csv_data/scene_01__A001++seq__6e11d637b13834d44e77__2023-04-27-19-22-25_robot.csv -d 41