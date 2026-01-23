#!/bin/bash
# Script to submit scene_diversity jobs sequentially with 1 minute delay between each

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SBATCH_DIR="${SCRIPT_DIR}/sbatch"

# Array of files from 8_7_5 to 1_3_75 (in order from bottom to top of table)
declare -a sbatch_files=(
    "scene_diversity_8_7_5.sh"
    "scene_diversity_8_3_75.sh"
    "scene_diversity_4_60.sh"
    "scene_diversity_4_30.sh"
    "scene_diversity_4_15.sh"
    "scene_diversity_4_7_5.sh"
    "scene_diversity_4_3_75.sh"
    "scene_diversity_2_60.sh"
    "scene_diversity_2_30.sh"
    "scene_diversity_2_15.sh"
    "scene_diversity_2_7_5.sh"
    "scene_diversity_2_3_75.sh"
    "scene_diversity_1_60.sh"
    "scene_diversity_1_30.sh"
    "scene_diversity_1_15.sh"
    "scene_diversity_1_7_5.sh"
    "scene_diversity_1_3_75.sh"
)

echo "Submitting scene_diversity jobs sequentially (1 minute delay between each)..."
echo "======================================================================"
echo "Total jobs to submit: ${#sbatch_files[@]}"
echo "Estimated total time: ~$(( ${#sbatch_files[@]} - 1 )) minutes"
echo "======================================================================"
echo ""

# Counter
job_num=0
success_count=0
fail_count=0

# Submit each job with delay
for sbatch_file in "${sbatch_files[@]}"; do
    ((job_num++))
    file_path="${SBATCH_DIR}/${sbatch_file}"
    
    if [ ! -f "$file_path" ]; then
        echo "[$job_num/${#sbatch_files[@]}] WARNING: File not found: ${file_path}"
        ((fail_count++))
        continue
    fi
    
    echo "[$job_num/${#sbatch_files[@]}] Submitting ${sbatch_file}..."
    output=$(sbatch "$file_path" 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Extract job ID from output
        job_id=$(echo "$output" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
        if [ -n "$job_id" ]; then
            echo "  ✓ SUCCESS - Job ID: $job_id"
        else
            echo "  ✓ SUCCESS - Job submitted"
        fi
        ((success_count++))
    else
        echo "  ✗ FAILED"
        echo "  Error: $output"
        ((fail_count++))
    fi
    
    # Wait 1 minute before next submission (except for the last one)
    if [ $job_num -lt ${#sbatch_files[@]} ]; then
        echo "  Waiting 30 seconds before next submission..."
        sleep 30
        echo ""
    fi
done

echo "======================================================================"
echo "Submission complete!"
echo "  Successful: $success_count"
echo "  Failed: $fail_count"
echo "  Total: $((success_count + fail_count))"
echo "======================================================================"
