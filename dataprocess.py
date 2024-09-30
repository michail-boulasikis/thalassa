import os
import sys
import pandas as pd

# Folder with CSV files
data_dir = 'generated-data'
frontend_measurements = pd.DataFrame()
backend_measurements = pd.DataFrame()
for csv_file in os.listdir(data_dir):
    dot_split = csv_file.split('.')
    # {eq_name}.{type_of_analysis}.{frontend/backend}.csv
    if len(dot_split) == 4:
        eq_name, type_of_analysis, frontend_or_backend, ending = dot_split

        # Add measurements to the corresponding DataFrame (the CSV file has one row)
        df = pd.read_csv(f'{data_dir}/{csv_file}')
        df['eq_name'] = eq_name
        if frontend_or_backend == 'frontend':
            frontend_measurements = pd.concat([frontend_measurements, df])
        elif frontend_or_backend == 'backend':
            backend_measurements = pd.concat([backend_measurements, df])
        else:
            print(f'Unknown type of analysis: {frontend_or_backend}')
            sys.exit(1)

# Drop the first column (index)
frontend_measurements = frontend_measurements.drop(columns=['Unnamed: 0'])
backend_measurements = backend_measurements.drop(columns=['Unnamed: 0'])
frontend_measurements.loc[:, 'simplify_derivatives'] = (frontend_measurements.loc[:, 'simplify_derivatives'] +
                                                        frontend_measurements.loc[:, 'substitute_free_functions'])
frontend_measurements.loc[:, 'verify_discretizations'] = (frontend_measurements.loc[:, 'verify_discretizations'] +
                                                          frontend_measurements.loc[:, 'discretize_pde_system'])
frontend_measurements.loc[:, 'compute_stencil_coefficients'] = (frontend_measurements.loc[:, 'compute_stencil_coefficients'] +
                                                                frontend_measurements.loc[:, 'prepare_ir'])
frontend_measurements = frontend_measurements.drop(columns=['substitute_free_functions', 'verify_discretizations', 'prepare_ir'])
frontend_measurements.columns = [col.replace('_', ' ').capitalize() for col in frontend_measurements.columns]
# Sort by the number of terms in the equation
frontend_measurements = frontend_measurements.sort_values(by='Total', ascending=True)
frontend_measurements.to_csv('frontend_measurements.csv', index=False)

# For the titles, remove the '_' and capitalize the first letter
