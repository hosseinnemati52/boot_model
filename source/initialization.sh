#!/bin/bash

# Read values from init_steps_data.txt
n=$(grep 'n_sampling:' init_steps_data.txt | awk '{print $2}')      # Extract the value of n_samp
m=$(grep 'm_checking:' init_steps_data.txt | awk '{print $2}')  # Extract the value of m_checking

# Loop from 1 to n
for ((i=1; i<=n; i++))
do
    # Step 1: Run "mechanically_relax.exe"
    ./mechanically_relax.exe
    
    # Step 2: Run "equilibration_sim.exe". Inside this function, the final distributions are written.
    ./equilibration_sim.exe

    # Step 3: Run the Python script "cond_check.py". Inside this, the file "condition.txt" is written.
    python3 eq_cond_check.py    

    # Step 4: Conditional logic
    if [[ "$i" -gt "$m" ]]; then
        # Read the condition from eq_condition.txt
        condition=$(cat eq_condition.txt)
        
        if [[ "$condition" -eq 1 ]]; then
            echo "Equilibration condition met (condition=1 and i>$m)!";
            
            # Run the script "make_phi_F.py"
            python3 make_phi_F.py;

            # Run the file "mechanically_relax.exe"
            ./mechanically_relax.exe;

            # Exit the loop
            break;
        else
            # Default action when condition != 1
            python3 make_phi_F.py;
        fi
    else
        # Default action when i <= m
        python3 make_phi_F.py;
    fi
done

