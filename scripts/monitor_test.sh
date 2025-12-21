#!/bin/bash
# Monitor PATH B GPU test progress

LOG_FILE="validation_results/path_b_onthefly_test.log"

echo "===================================================================="
echo "PATH B GPU TEST MONITOR"
echo "===================================================================="
echo ""

# Check if test is running
if ps aux | grep -q "[v]asil_exact_gpu"; then
    echo "✅ Test process is RUNNING"
else
    echo "⚠️  Test process NOT RUNNING (may have completed or failed)"
fi

echo ""
echo "Current Progress:"
echo "-------------------------------------------------------------------"

# Show countries completed
echo ""
echo "Countries processed:"
tail -500 "$LOG_FILE" | grep -E "Building cache for|Built in" | tail -20

echo ""
echo "-------------------------------------------------------------------"
echo "Latest activity (last 10 lines):"
tail -10 "$LOG_FILE"

echo ""
echo "-------------------------------------------------------------------"
echo "Check for results:"
if grep -q "RESULTS" "$LOG_FILE"; then
    echo "✅ RESULTS FOUND!"
    tail -100 "$LOG_FILE" | grep -A 30 "RESULTS"
else
    echo "⏳ Test still running... (results not yet available)"
fi

echo ""
echo "===================================================================="
echo "Monitor command: watch -n 5 'bash scripts/monitor_test.sh'"
echo "===================================================================="
