# 2. Loop through all .pkl.pkl files in the target directory
for file in /hhd4/lizhe/stcomp/robot_data/UnitreeG1/dance_sh_dy/*.pkl.pkl; do
    
    # Extract the filename without the path and extension (e.g., "dance_sh_10")
    base_name=$(basename "$file" .pkl.pkl)
    
    # Create the output path (e.g., "videos/dance_sh_10.mp4")
    output_video="videos/${base_name}.mp4"
    
    echo "Processing: $base_name -> $output_video"

    # Run the command
    xvfb-run -a -s "-screen 0 1024x768x24" python scripts/vis_robot_motion.py \
        --robot unitree_g1 \
        --robot_motion_path "$file" \
        --record_video \
        --video_path "$output_video"
        
done
